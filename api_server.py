import os
import shutil
import glob
import uuid
import asyncio
import aiohttp
import requests
from datetime import datetime
from typing import Dict, Optional, List, Union
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, HttpUrl
import threading
import time

# Import the generate function from app_acc_core.py
from app_acc_core import generate

# Global dictionary to store generation tasks progress
tasks_progress: Dict[str, Dict] = {}


class GenerationTask:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "pending"
        self.progress = 0
        self.result_path = None
        self.error = None

    def update_progress(self, progress: int):
        self.progress = progress
        self.status = "processing"

    def complete(self, result_path: str):
        self.status = "completed"
        self.progress = 100
        self.result_path = result_path

    def fail(self, error: str):
        self.status = "failed"
        self.error = error

    def to_dict(self):
        result = {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
        }
        if self.result_path:
            result["result_path"] = self.result_path
        if self.error:
            result["error"] = self.error
        return result


class UrlInput(BaseModel):
    image_url: Optional[HttpUrl] = None
    audio_url: Optional[HttpUrl] = None
    pose_input: str
    width: int = 768
    height: int = 768
    length: int = 120
    steps: int = 30
    sample_rate: int = 16000
    cfg: float = 2.5
    fps: int = 24
    context_frames: int = 12
    context_overlap: int = 3
    quantization_input: bool = False
    seed: int = -1


def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination


async def download_file(url: str, destination: str) -> str:
    """Download a file from URL and save it to the destination."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=400, detail=f"Failed to download file from {url}"
                )

            with open(destination, "wb") as f:
                f.write(await response.read())

    return destination


def find_mp4_files(directory):
    """Recursively find all .mp4 files in the directory."""
    mp4_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                # Get the relative path from the base directory
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                mp4_files.append(rel_path)
    return mp4_files


def cleanup_temp_files(paths: List[str], delay: int = 300):
    """Clean up temporary files after a delay (in seconds)."""

    def _cleanup():
        time.sleep(delay)  # Wait for the specified delay
        for path in paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    print(f"Failed to clean up {path}: {e}")

    # Start cleanup in a separate thread
    threading.Thread(target=_cleanup, daemon=True).start()


def run_generation(
    task: GenerationTask,
    image_path: str,
    audio_path: str,
    pose_input: str,
    width: int,
    height: int,
    length: int,
    steps: int,
    sample_rate: int,
    cfg: float,
    fps: int,
    context_frames: int,
    context_overlap: int,
    quantization_input: bool,
    seed: int,
):
    """Run the generation process in a background thread."""
    try:
        # Create the generated_videos directory if it doesn't exist
        os.makedirs("generated_videos", exist_ok=True)

        # Ensure pose_input has the correct path
        # Check if pose_input is one of the predefined pose names
        if pose_input in ["fight", "good", "ultraman", "salute"]:
            pose_path = f"assets/halfbody_demo/pose/{pose_input}"
        elif pose_input in ["01", "02", "03", "04"]:
            pose_path = f"assets/halfbody_demo/pose/{pose_input}"
        else:
            # If it's already a path, use it as is
            pose_path = pose_input

        # Verify that the pose path exists
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose path not found: {pose_path}")

        # Run the generation
        def progress_callback(progress):
            # Ensure progress is an integer between 0 and 100
            progress = max(0, min(int(progress), 100))
            task.update_progress(progress)
            tasks_progress[task.task_id] = task.to_dict()

        video_output, _ = generate(
            image_path,
            audio_path,
            pose_path,  # Use the corrected pose path instead of pose_input
            width,
            height,
            length,
            steps,
            sample_rate,
            cfg,
            fps,
            context_frames,
            context_overlap,
            quantization_input,
            seed,
            progress_callback=progress_callback,
        )

        # Move the output to the generated_videos directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = f"generated_videos/{timestamp}_{os.path.basename(video_output)}"
        shutil.move(video_output, final_output)

        # Update task status
        task.complete(final_output)
        tasks_progress[task.task_id] = task.to_dict()

        # Schedule cleanup of temporary files
        cleanup_temp_files([image_path, audio_path])

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()

        # Print error with visual emphasis in terminal
        print("\n" + "=" * 80)
        print(f"\033[91m[ERROR] Generation Failed for task {task.task_id}:\033[0m")
        print(f"\033[93mError message: {str(e)}\033[0m")
        print(f"\033[97mError details:\n{error_details}\033[0m")
        print("=" * 80 + "\n")

        # Update task status
        task.fail(str(e))
        tasks_progress[task.task_id] = task.to_dict()

        # Clean up temp files even if generation fails
        cleanup_temp_files([image_path, audio_path])


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist before mounting
os.makedirs("output", exist_ok=True)
os.makedirs("generated_videos", exist_ok=True)

# Mount static directories
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount(
    "/generated_videos",
    StaticFiles(directory="generated_videos"),
    name="generated_videos",
)


@app.post("/generate")
async def generate_api(
    background_tasks: BackgroundTasks,
    image_input: UploadFile = File(None),
    audio_input: UploadFile = File(None),
    pose_input: str = Form(...),
    width: int = Form(768),
    height: int = Form(768),
    length: int = Form(120),
    steps: int = Form(30),
    sample_rate: int = Form(16000),
    cfg: float = Form(2.5),
    fps: int = Form(24),
    context_frames: int = Form(12),
    context_overlap: int = Form(3),
    quantization_input: bool = Form(False),
    seed: int = Form(-1),
    image_url: str = Form(None),
    audio_url: str = Form(None),
):
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    task = GenerationTask(task_id)
    tasks_progress[task_id] = task.to_dict()

    # Save uploaded files with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"temp/{timestamp}_image.png"
    audio_path = f"temp/{timestamp}_audio.wav"

    try:
        # Handle image input (file upload or URL)
        if image_input:
            save_upload_file(image_input, image_path)
        elif image_url:
            await download_file(image_url, image_path)
        else:
            raise HTTPException(
                status_code=400, detail="Either image file or URL must be provided"
            )

        # Handle audio input (file upload or URL)
        if audio_input:
            save_upload_file(audio_input, audio_path)
        elif audio_url:
            await download_file(audio_url, audio_path)
        else:
            raise HTTPException(
                status_code=400, detail="Either audio file or URL must be provided"
            )

        # Start generation in background
        threading.Thread(
            target=run_generation,
            args=(
                task,
                image_path,
                audio_path,
                pose_input,
                width,
                height,
                length,
                steps,
                sample_rate,
                cfg,
                fps,
                context_frames,
                context_overlap,
                quantization_input,
                seed,
            ),
            daemon=True,
        ).start()

        # Return the task ID for tracking progress
        return {"task_id": task_id, "status": "processing"}

    except Exception as e:
        # Clean up temp files if there's an error
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

        task.fail(str(e))
        tasks_progress[task_id] = task.to_dict()
        return JSONResponse(
            status_code=500, content={"error": str(e), "task_id": task_id}
        )


@app.post("/generate_from_url")
async def generate_from_url_api(url_input: UrlInput):
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    task = GenerationTask(task_id)
    tasks_progress[task_id] = task.to_dict()

    # Save uploaded files with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"temp/{timestamp}_image.png"
    audio_path = f"temp/{timestamp}_audio.wav"

    try:
        # Handle image input from URL
        if not url_input.image_url:
            raise HTTPException(status_code=400, detail="Image URL must be provided")
        await download_file(str(url_input.image_url), image_path)

        # Handle audio input from URL
        if not url_input.audio_url:
            raise HTTPException(status_code=400, detail="Audio URL must be provided")
        await download_file(str(url_input.audio_url), audio_path)

        # Start generation in background
        threading.Thread(
            target=run_generation,
            args=(
                task,
                image_path,
                audio_path,
                url_input.pose_input,
                url_input.width,
                url_input.height,
                url_input.length,
                url_input.steps,
                url_input.sample_rate,
                url_input.cfg,
                url_input.fps,
                url_input.context_frames,
                url_input.context_overlap,
                url_input.quantization_input,
                url_input.seed,
            ),
            daemon=True,
        ).start()

        # Return the task ID for tracking progress
        return {"task_id": task_id, "status": "processing"}

    except Exception as e:
        # Clean up temp files if there's an error
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

        task.fail(str(e))
        tasks_progress[task_id] = task.to_dict()
        return JSONResponse(
            status_code=500, content={"error": str(e), "task_id": task_id}
        )


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a generation task."""
    if task_id not in tasks_progress:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = tasks_progress[task_id]

    # If the task is completed, include the result URL
    if task_info["status"] == "completed" and "result_path" in task_info:
        task_info["result_url"] = (
            f"/generated_videos/{os.path.basename(task_info['result_path'])}"
        )

    return task_info


@app.get("/task/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """Stream the progress of a generation task."""
    if task_id not in tasks_progress:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        previous_progress = -1
        heartbeat_counter = 0

        try:
            while True:
                # Send heartbeat every 15 seconds to keep connection alive
                heartbeat_counter += 1
                if heartbeat_counter >= 30:  # 30 x 0.5s = 15s
                    yield f'data: {{"heartbeat": true}}\n\n'
                    heartbeat_counter = 0

                if task_id not in tasks_progress:
                    yield f'data: {{"error": "Task not found"}}\n\n'
                    break

                task_info = tasks_progress[task_id]
                current_progress = task_info.get("progress", 0)

                # Send update if progress changed or status is completed/failed
                if current_progress != previous_progress or task_info["status"] in [
                    "completed",
                    "failed",
                ]:
                    # Convert task_info to a properly formatted JSON string
                    import json

                    task_info_json = json.dumps(task_info)
                    yield f"data: {task_info_json}\n\n"
                    previous_progress = current_progress

                    # If task is done, stop streaming
                    if task_info["status"] in ["completed", "failed"]:
                        break

                # Use a shorter sleep interval to be more responsive
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            # Handle client disconnection gracefully
            print(f"Client disconnected from stream for task {task_id}")
            return
        except Exception as e:
            print(f"Error in event stream for task {task_id}: {str(e)}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevents proxy buffering which can cause timeout issues
        },
    )


@app.get("/outputs")
async def list_outputs_api():
    """List all generated videos from the generated_videos directory."""
    # Create the directory if it doesn't exist
    os.makedirs("generated_videos", exist_ok=True)

    # Get files from generated_videos directory
    files = find_mp4_files("generated_videos")

    # Sort files by modification time (newest first)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(os.path.join("generated_videos", f)),
        reverse=True,
    )

    # Convert to proper paths for frontend
    relative_files = [f"generated_videos/{f}" for f in files]

    return {"files": relative_files}


@app.get("/video/{video_id}")
async def get_video(video_id: str):
    """Get a specific video by its ID (filename without extension)."""
    # Look for the video in generated_videos directory
    video_path = None
    for root, _, files in os.walk("generated_videos"):
        for file in files:
            if file.endswith(".mp4") and video_id in file:
                video_path = os.path.join(root, file)
                break
        if video_path:
            break

    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
    )


@app.on_event("startup")
async def startup_event():
    """Create necessary directories on startup."""
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("generated_videos", exist_ok=True)


@app.get("/debug/list_directory")
async def list_directory(directory_path: str):
    """
    Debug endpoint to list the contents of a directory.
    This helps in troubleshooting path-related issues.
    """
    try:
        # Check if the directory exists
        if not os.path.exists(directory_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Directory not found: {directory_path}"},
            )

        # Check if it's actually a directory
        if not os.path.isdir(directory_path):
            return JSONResponse(
                status_code=400,
                content={"error": f"Path is not a directory: {directory_path}"},
            )

        # Get directory contents
        contents = os.listdir(directory_path)

        # Get detailed info for each item
        items = []
        for item in contents:
            item_path = os.path.join(directory_path, item)
            item_info = {
                "name": item,
                "is_dir": os.path.isdir(item_path),
                "size": (
                    os.path.getsize(item_path) if os.path.isfile(item_path) else None
                ),
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(item_path)
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
            items.append(item_info)

        return {
            "directory": directory_path,
            "contents": items,
            "total_items": len(items),
        }
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()

        # Log error to console
        print(
            f"\033[91m[ERROR] Error listing directory {directory_path}: {str(e)}\033[0m"
        )
        print(error_traceback)

        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to list directory: {str(e)}",
                "traceback": error_traceback,
            },
        )
