from fastapi import APIRouter, UploadFile
from playground import handle_image, handle_video
import os
import io

VIRTUS_TEST_DATASET = {
    "real": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../Datasets/test/images/real'
                                         )),
    "fake": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../Datasets/test/images/fake'
                                         ))
}
SCARLET_TEST_DATASET = {
    "real": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../Datasets/test/videos/real'
                                         )),
    "fake": os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../Datasets/test/videos/fake'
                                         ))
}

router = APIRouter()


async def handel_testing(dataset: str, pos: int):
        """Loads the required file and returns a UploadFile object"""
        with open(f"{dataset}/00{pos}.mp4", "rb") as f:
            content = io.BytesIO(f.read())
            image = UploadFile(filename=f"test-{pos}.mp4", file=content)
        return image


@router.post("/")
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