from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import tempfile
from pathlib import Path
from models.image_model import virtus
from models.video_model import scarlet

router = APIRouter()

# Dataset directory for sample files
DATASET_DIR = Path("app/datasets")

async def handle_image(file: UploadFile):
    """Process and analyze the uploaded image file."""
    try:
        image = Image.open(file.file).convert("RGB")
        label, confidence = virtus(image)
        return JSONResponse({
            "type": "image",
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image error: {e}")

async def handle_video(file: UploadFile):
    """Process and analyze the uploaded video file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        label, confidence = scarlet(tmp_path)

        return JSONResponse({
            "type": "video",
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video error: {e}")

@router.post("/")
async def playground(file: UploadFile = File(...)):
    """
    Analyze an uploaded image or video file to detect deepfakes.

    - **Accepted image formats**: .jpg, .jpeg, .png, .bmp, .webp
    - **Accepted video formats**: .mp4, .mov, .avi, .webm, .mkv
    - Returns the type (image or video)[0], predicted label (real/fake)[1], and confidence score [2] as json response.
    """
    filename = file.filename.lower()

    if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return await handle_image(file)

    elif filename.endswith((".mp4", ".mov", ".avi", ".webm", ".mkv")):
        return await handle_video(file)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


@router.post("/test")
async def test_sample(file: UploadFile = File(...)):
    """
    Test with sample files sent from frontend.
    
    - Accepts the actual sample file uploaded from frontend
    - Analyzes it just like a regular file upload
    - Returns the predicted label and confidence
    """
    try:
        print(f"Received sample file: {file.filename}")
        
        filename = file.filename.lower()
        
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            print(f"Processing sample image: {file.filename}")
            return await handle_image(file)
            
        elif filename.endswith((".mp4", ".mov", ".avi", ".webm", ".mkv")):
            print(f"Processing sample video: {file.filename}")
            return await handle_video(file)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
            
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing sample: {str(e)}")

