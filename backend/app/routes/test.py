from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from PIL import Image
from models.image_model import virtus
from models.video_model import scarlet

router = APIRouter()


DATASET_DIR = Path("app/datasets")

@router.get("/")
async def try_prediction(filename: str = Query(..., description="Name of the image or video file to test.")):
    """
    Run prediction on a stored image or video from the datasets folder.

    - Send a filename like 'fake_image_1.jpg' or 'real_video_2.mp4'.
    - It must exist under `app/datasets/`.
    - Returns the predicted label (real/fake) and confidence.
    """
    file_path = DATASET_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in datasets.")

    try:
        if filename.lower().endswith(".jpg"):
            image = Image.open(file_path).convert("RGB")
            label, confidence = virtus(image)
            return JSONResponse({
                "type": "image",
                "label": label,
                "confidence": confidence
            })

        elif filename.lower().endswith(".mp4"):
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