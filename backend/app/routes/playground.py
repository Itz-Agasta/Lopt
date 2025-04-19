from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tempfile
from models.image_model import virtus
from models.video_model import scarlet
import os
import io

VIRTUS_TEST_DATASET = {
    "real": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../../../models/image/Dataset/test/real'
                                         )),
    "fake": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../../../models/image/Dataset/test/fake'
                                         ))
}
SCARLET_TEST_DATASET = {
    "real": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../../../models/video/Dataset/test/real'
                                         )),
    "fake": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../../../models/video/Dataset/test/fake'
                                         ))
}

router = APIRouter()

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

async def handel_testing(dataset: str, pos: int):
        """Loads the required file and returns a UploadFile object"""
        with open(f"{dataset}/00{pos}.mp4", "rb") as f:
            content = io.BytesIO(f.read())
            image = UploadFile(filename=f"test-{pos}.mp4", file=content)
        return image

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

@router.post("/try")
async def test(pos: int, type: str):
    """
    Analyzes an image or video in the test dataset based on pos and type.

    - **pos**: int in the range 0-5
    - **type**: either 'image real', 'image fake' or 'video real', 'video fake'
    - Returns the type (image or video)[0], predicted label (real/fake)[1], and confidence score [2] as json response.
    """
    types = type.split(" ")
    print(types)
    if pos > 5 :
        raise ValueError(f"Value of pos can only be between 0 and 5")
    if type not in ["video real", "image real", "image fake", "video fake"]:
        raise ValueError(f"Value of type can only be 'image real', 'image fake' or 'video real', 'video fake' not {type}")
    if types[0] == "image":
        image = await handel_testing(VIRTUS_TEST_DATASET[types[1]], pos)
        return await handle_image(image)
    elif types[0] == "video":
        video = await handel_testing(SCARLET_TEST_DATASET[types[1]], pos)
        return await handle_video(video)