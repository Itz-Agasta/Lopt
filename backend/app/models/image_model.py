from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch.nn.functional as F


MODEL_PATH = "agasta/virtus"  # https://huggingface.co/agasta/virtus

# Load model and preprocessing pipeline once & cache it to ~/.cache/huggingface/transformers
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH) # https://huggingface.co/docs/transformers/en/model_doc/auto
extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model.eval()

def virtus(pil_image: Image.Image) -> tuple[str, float]:
    """
    Analyze a PIL image and return label and confidence.

    Returns:
        label (str): "fake" or "real"
        confidence (float): confidence score in percentage (0–100)
    """
    inputs = extractor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)
    label = model.config.id2label[predicted_class.item()]
    confidence_percent = round(confidence.item() * 100, 2)

    return label.lower(), confidence_percent