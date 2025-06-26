import os
import random
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from moviepy.editor import VideoFileClip, AudioFileClip
from datetime import datetime
from torchao.quantization import quantize_, int8_weight_only
import gc

# Initialize CUDA and system settings
total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
print(f"\033[32mCUDA Version: {torch.version.cuda}\033[0m")
print(f"\033[32mPytorch Version: {torch.__version__}\033[0m")
print(f"\033[32mGPU Model: {torch.cuda.get_device_name()}\033[0m")
print(f"\033[32mVRAM Size: {total_vram_in_gb:.2f}GB\033[0m")
print(f"\033[32mPrecision: float16\033[0m")

dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check FFMPEG availability
ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path is None:
    print(
        "Please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static"
    )
elif ffmpeg_path not in os.getenv("PATH"):
    print("Adding ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def generate(
    image_input,
    audio_input,
    pose_input,
    width=768,
    height=768,
    length=120,
    steps=30,
    sample_rate=16000,
    cfg=2.5,
    fps=24,
    context_frames=12,
    context_overlap=3,
    quantization_input=False,
    seed=-1,
    progress_callback=None,
):
    """
    Generate a video based on the input parameters.

    Args:
        image_input (str): Path to the reference image
        audio_input (str): Path to the audio file
        pose_input (str): Path to the pose directory
        width (int): Output video width
        height (int): Output video height
        length (int): Number of frames
        steps (int): Number of denoising steps
        sample_rate (int): Audio sample rate
        cfg (float): Classifier free guidance scale
        fps (int): Frames per second
        context_frames (int): Number of context frames
        context_overlap (int): Context frame overlap
        quantization_input (bool): Whether to use INT8 quantization
        seed (int): Random seed (-1 for random)
        progress_callback (callable): Optional callback function to report progress

    Returns:
        tuple: (video_output_path, seed)
    """
    try:
        # Print input parameters for easier debugging
        print("\n" + "-" * 40)
        print("\033[94m[GENERATION START]\033[0m")
        print(f"Pose input: \033[92m{pose_input}\033[0m")
        print(f"Reference image: \033[92m{image_input}\033[0m")
        print(f"Audio input: \033[92m{audio_input}\033[0m")
        print(f"Width: {width}, Height: {height}, Length: {length}, Steps: {steps}")
        print(
            f"FPS: {fps}, Context frames: {context_frames}, Context overlap: {context_overlap}"
        )
        print("-" * 40 + "\n")

        # Verify that input files exist
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        if not os.path.exists(pose_input):
            raise FileNotFoundError(f"Pose directory not found: {pose_input}")
        if not os.path.isdir(pose_input):
            raise NotADirectoryError(f"Pose input is not a directory: {pose_input}")

        pose_files = os.listdir(pose_input)
        if not pose_files:
            raise FileNotFoundError(f"Pose directory is empty: {pose_input}")
        print(f"Found {len(pose_files)} files in pose directory")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("outputs")
        save_dir.mkdir(exist_ok=True, parents=True)

        # Initialize models
        vae = AutoencoderKL.from_pretrained("./pretrained_weights/sd-vae-ft-mse").to(
            device, dtype=dtype
        )
        if quantization_input:
            quantize_(vae, int8_weight_only())
            print("Using INT8 quantization")

        reference_unet = UNet2DConditionModel.from_pretrained(
            "./pretrained_weights/sd-image-variations-diffusers",
            subfolder="unet",
            use_safetensors=False,
        ).to(dtype=dtype, device=device)
        reference_unet.load_state_dict(
            torch.load("./pretrained_weights/reference_unet.pth", weights_only=True)
        )
        if quantization_input:
            quantize_(reference_unet, int8_weight_only())

        if not os.path.exists("./pretrained_weights/motion_module_acc.pth"):
            raise FileNotFoundError("Motion module not found")

        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            "./pretrained_weights/sd-image-variations-diffusers",
            "./pretrained_weights/motion_module_acc.pth",
            subfolder="unet",
            unet_additional_kwargs={
                "use_inflated_groupnorm": True,
                "unet_use_cross_frame_attention": False,
                "unet_use_temporal_attention": False,
                "use_motion_module": True,
                "cross_attention_dim": 384,
                "motion_module_resolutions": [1, 2, 4, 8],
                "motion_module_mid_block": True,
                "motion_module_decoder_only": False,
                "motion_module_type": "Vanilla",
                "motion_module_kwargs": {
                    "num_attention_heads": 8,
                    "num_transformer_block": 1,
                    "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                    "temporal_position_encoding": True,
                    "temporal_position_encoding_max_len": 32,
                    "temporal_attention_dim_div": 1,
                },
            },
        ).to(dtype=dtype, device=device)
        denoising_unet.load_state_dict(
            torch.load(
                "./pretrained_weights/denoising_unet_acc.pth", weights_only=True
            ),
            strict=False,
        )

        pose_net = PoseEncoder(
            320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
        ).to(dtype=dtype, device=device)
        pose_net.load_state_dict(
            torch.load("./pretrained_weights/pose_encoder.pth", weights_only=True)
        )

        audio_processor = load_audio_model(
            model_path="./pretrained_weights/audio_processor/tiny.pt", device=device
        )

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
            steps_offset=1,
            prediction_type="v_prediction",
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )

        pipe = EchoMimicV2Pipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            audio_guider=audio_processor,
            pose_encoder=pose_net,
            scheduler=scheduler,
        ).to(device, dtype=dtype)

        # Set random seed
        if seed is not None and seed > -1:
            generator = torch.manual_seed(seed)
        else:
            seed = random.randint(100, 1000000)
            generator = torch.manual_seed(seed)

        save_name = f"{save_dir}/{timestamp}"

        ref_image_pil = Image.open(image_input).resize((width, height))
        audio_clip = AudioFileClip(audio_input)

        length = min(
            length, int(audio_clip.duration * fps), len(os.listdir(pose_input))
        )
        start_idx = 0

        # Process pose frames
        pose_list = []
        print(f"\033[94m[INFO] Processing pose frames from {pose_input}\033[0m")
        for index in range(start_idx, start_idx + length):
            try:
                tgt_musk = np.zeros((width, height, 3)).astype("uint8")
                tgt_musk_path = os.path.join(pose_input, f"{index}.npy")

                # Verbose error if file doesn't exist
                if not os.path.exists(tgt_musk_path):
                    raise FileNotFoundError(f"Pose frame not found: {tgt_musk_path}")

                detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()

                # Check if the file has the expected structure
                if "draw_pose_params" not in detected_pose:
                    raise ValueError(
                        f"Invalid pose file structure in {tgt_musk_path}: 'draw_pose_params' not found"
                    )

                imh_new, imw_new, rb, re, cb, ce = detected_pose["draw_pose_params"]
                im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
                im = np.transpose(np.array(im), (1, 2, 0))
                tgt_musk[rb:re, cb:ce, :] = im

                # Print progress every 20 frames
                if index % 20 == 0 or index == start_idx:
                    print(
                        f"\033[94m[INFO] Processed pose frame {index}/{start_idx + length - 1}\033[0m"
                    )

            except Exception as e:
                print(
                    f"\033[91m[ERROR] Failed to process pose frame {index}: {str(e)}\033[0m"
                )
                raise

            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert("RGB")
            pose_list.append(
                torch.Tensor(np.array(tgt_musk_pil))
                .to(dtype=dtype, device=device)
                .permute(2, 0, 1)
                / 255.0
            )

        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        audio_clip = audio_clip.set_duration(length / fps)

        # Generate video
        if progress_callback:
            progress_callback(10)  # Loading models and preparing data completed

        video = pipe(
            ref_image_pil,
            audio_input,
            poses_tensor[:, :, :length, ...],
            width,
            height,
            length,
            steps,
            cfg,
            generator=generator,
            audio_sample_rate=sample_rate,
            context_frames=context_frames,
            fps=fps,
            context_overlap=context_overlap,
            start_idx=start_idx,
            progress_callback=progress_callback,
        ).videos

        if progress_callback:
            progress_callback(80)  # Video generation completed

        final_length = min(video.shape[2], poses_tensor.shape[2], length)
        video_sig = video[:, :, :final_length, :, :]

        # Save video without audio
        save_videos_grid(
            video_sig,
            save_name + "_woa_sig.mp4",
            n_rows=1,
            fps=fps,
        )

        if progress_callback:
            progress_callback(90)  # Video without audio saved

        # Add audio to video
        video_clip_sig = VideoFileClip(save_name + "_woa_sig.mp4")
        video_clip_sig = video_clip_sig.set_audio(audio_clip)
        video_clip_sig.write_videofile(
            save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2
        )

        if progress_callback:
            progress_callback(100)  # Final video with audio saved

        video_output = save_name + "_sig.mp4"

        print(f"\033[92m[SUCCESS] Generation completed: {video_output}\033[0m")
        print("-" * 40 + "\n")

        return video_output, seed
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()

        print("\n" + "=" * 80)
        print(f"\033[91m[ERROR] Generation failed!\033[0m")
        print(f"\033[93mError message: {str(e)}\033[0m")
        print(f"\033[97mError details:\n{error_details}\033[0m")
        print("=" * 80 + "\n")

        # Re-raise the exception to be caught by the caller
        raise
