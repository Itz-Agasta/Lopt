from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
from models.image_model import virtus
from models.video_model import scarlet

router = APIRouter()

DATASET_DIR = Path("app/datasets")

class FileRequest(BaseModel):
    filename: str

@router.post("/")
def try_prediction(payload: FileRequest):
    """
    Run prediction on a stored image or video from the datasets folder.

    - Send JSON like { "filename": "fake_image_1.jpg" }
    - It must exist under `app/datasets/`.
    - Returns the predicted label (real/fake) and confidence.
    """
    filename = payload.filename
    file_path = DATASET_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in datasets.")

    try:
        ext = file_path.suffix.lower()
        if ext in ".jpg":  # Dataset only contain .Jpg images.
            image = Image.open(file_path).convert("RGB")
            label, confidence = virtus(image)
            return JSONResponse({
                "type": "image",
                "label": label,
                "confidence": confidence
            })

        elif ext == ".mp4":   # Dataset only contain .mp4 videos.
            label, confidence = scarlet(str(file_path))
            return JSONResponse({
                "type": "video",
                "label": label,
                "confidence": confidence
            })

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
