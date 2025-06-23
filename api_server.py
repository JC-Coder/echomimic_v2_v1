import os
import shutil
import glob
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import the generate function from app.py
from app_acc import generate


def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
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


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount only the output directory
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.post("/generate")
async def generate_api(
    image_input: UploadFile = File(...),
    audio_input: UploadFile = File(...),
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
):
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    # Save uploaded files with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"temp/{timestamp}_image.png"
    audio_path = f"temp/{timestamp}_audio.wav"

    save_upload_file(image_input, image_path)
    save_upload_file(audio_input, audio_path)

    try:
        video_output, _ = generate(
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
        )

        return FileResponse(
            video_output,
            media_type="video/mp4",
            filename=os.path.basename(video_output),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up temp files
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/outputs")
async def list_outputs_api():
    """List all generated videos from the output directory."""
    # Get files from output directory
    files = find_mp4_files("output")

    # Sort files by modification time (newest first)
    files = sorted(
        files, key=lambda f: os.path.getmtime(os.path.join("output", f)), reverse=True
    )

    # Convert to proper paths for frontend
    relative_files = [f"output/{f}" for f in files]

    return {"files": relative_files}


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)
