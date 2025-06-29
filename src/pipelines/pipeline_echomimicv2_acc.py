import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method
from src.pipelines.step_func import (
    origin_by_velocity_and_sample,
    psuedo_velocity_wrt_noisy_and_timestep,
)


@dataclass
class EchoMimicV2PipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class EchoMimicV2Pipeline(DiffusionPipeline):

    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        audio_guider,
        pose_encoder,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            audio_guider=audio_guider,
            pose_encoder=pose_encoder,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            # audio_feature_mapper=audio_feature_mapper
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents_bp(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        context_frame_length,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            # context_frame_length,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents_seg = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )
        latents = latents_seg

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        print(f"latents shape:{latents.shape}, video_length:{video_length}")
        return latents

    def prepare_latents_smooth(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        context_frame_length,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            # context_frame_length,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents_seg = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )

        latents = latents_seg

        latents = torch.clamp(latents_seg, -1.5, 1.5)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        print(f"latents shape:{latents.shape}, video_length:{video_length}")

        return latents

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        audio_path,
        poses_tensor,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=0,
        context_batch_size=1,
        interpolation_factor=1,
        audio_sample_rate=16000,
        fps=25,
        audio_margin=2,
        start_idx=0,
        progress_callback=None,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        whisper_feature = self.audio_guider.audio2feat(audio_path)
        whisper_chunks = self.audio_guider.feature2chunks(
            feature_array=whisper_feature, fps=fps
        )
        audio_frame_num = whisper_chunks.shape[0]
        audio_fea_final = torch.Tensor(whisper_chunks).to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        audio_fea_final = audio_fea_final.unsqueeze(0)

        video_length = min(video_length, audio_frame_num)

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents_smooth(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            audio_fea_final.dtype,
            device,
            generator,
            context_frames,
        )

        pose_enocder_tensor = self.pose_encoder(poses_tensor)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b , 4, h, w)

        context_scheduler = get_context_scheduler(context_schedule)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        context_queue = list(
            context_scheduler(
                0,
                num_inference_steps,
                latents.shape[2],
                context_frames,
                context_stride,
                context_overlap,
            )
        )

        # Report initial progress if callback is provided
        if progress_callback is not None:
            progress_callback(15)  # Initial setup is complete

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t_i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image
                if t_i == 0:
                    self.reference_unet(
                        ref_image_latents,
                        torch.zeros_like(t),
                        encoder_hidden_states=None,
                        return_dict=False,
                    )
                    reference_control_reader.update(
                        reference_control_writer,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                    )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                global_context = []
                for j in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            j * context_batch_size : (j + 1) * context_batch_size
                        ]
                    )

                ## refine
                for context in global_context:
                    new_context = [
                        [0 for _ in range(len(context[c_j]))]
                        for c_j in range(len(context))
                    ]
                    for c_j in range(len(context)):
                        for c_i in range(len(context[c_j])):
                            new_context[c_j][c_i] = (
                                context[c_j][c_i] + t_i * 3
                            ) % video_length

                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in new_context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )

                    audio_latents_cond = torch.cat(
                        [audio_fea_final[:, c] for c in new_context]
                    ).to(device)

                    audio_latents = torch.cat(
                        [torch.zeros_like(audio_latents_cond), audio_latents_cond], 0
                    )
                    pose_latents_cond = torch.cat(
                        [pose_enocder_tensor[:, :, c] for c in new_context]
                    ).to(device)
                    pose_latents = torch.cat(
                        [torch.zeros_like(pose_latents_cond), pose_latents_cond], 0
                    )

                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    b, c, f, h, w = latent_model_input.shape

                    pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        audio_cond_fea=(
                            audio_latents
                            if do_classifier_free_guidance
                            else audio_latents_cond
                        ),
                        face_musk_fea=(
                            pose_latents
                            if do_classifier_free_guidance
                            else pose_latents_cond
                        ),
                        return_dict=False,
                    )[0]

                    alphas_cumprod = self.scheduler.alphas_cumprod.to(
                        latent_model_input.device
                    )
                    x_pred = origin_by_velocity_and_sample(
                        pred, latent_model_input, alphas_cumprod, t
                    )
                    pred = psuedo_velocity_wrt_noisy_and_timestep(
                        latent_model_input,
                        x_pred,
                        alphas_cumprod,
                        t,
                        torch.ones_like(t) * (-1),
                    )

                    for j, c in enumerate(new_context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred / counter
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # Update progress via callback if provided
                if progress_callback is not None and (
                    t_i % max(1, len(timesteps) // 10) == 0
                ):
                    # Calculate progress from 15% to 75% based on denoising steps
                    progress_percent = int(15 + (t_i / len(timesteps)) * 60)
                    progress_callback(progress_percent)

                if t_i == len(timesteps) - 1 or (
                    (t_i + 1) > num_warmup_steps
                    and (t_i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

            reference_control_reader.clear()
            reference_control_writer.clear()

        if progress_callback is not None:
            progress_callback(75)  # Denoising complete

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)
        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        if progress_callback is not None:
            progress_callback(79)  # Decoding complete

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if progress_callback is not None:
            progress_callback(80)  # Processing complete

        if not return_dict:
            return images

        return EchoMimicV2PipelineOutput(videos=images)
