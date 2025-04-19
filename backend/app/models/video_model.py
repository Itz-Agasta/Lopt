from transformers import VideoMAEImageProcessor, AutoModelForVideoClassification
from decord import VideoReader, cpu
import torch.nn.functional as F
import torch
import os

MODEL_PATH = "agasta/scarlet"
MAX_SIZE = 50

#Load the model and video extraction pipeline
model = AutoModelForVideoClassification.from_pretrained(MODEL_PATH) 
extractor = VideoMAEImageProcessor.from_pretrained("models\config") 
model.eval()


def scarlet(video_path: str) -> tuple[str, float]:
    """
        Analyzes a video and return label and confidence.
        
        Description:
            Uses decord library to read the video file and extract 8 evenly spaced 
            frames form the video. The formed numpy list of the video frames are then 
            passed to the VideoMAEFeatureExtractor (from the VideoMAE-small model) to
            get inputs to scarlet.


        Returns:
            label (str): "fake" or "real"
            confidence (float): confidence score in percentage (0-100)
    """
    video_size = os.path.getsize(video_path) / (1024*1024)
    if video_size > MAX_SIZE:
        raise ValueError(f"Video size too large ({video_size:.2f} MB).")
    
    num_frames = 8
    vr = VideoReader(video_path, ctx=cpu(0))  # Load video
    total_frames = len(vr)  # Total number of frames
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]  # Sample evenly spaced frames
    frames = [vr[i].asnumpy() for i in indices] #make the final numpy array of frames

    #pass the video file to extractor to get inputs
    inputs = extractor(images= frames, return_tensors="pt")

    with torch.no_grad():
        results = model(**inputs)
        logits = results.logits
        probs = F.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)
    label = model.config.id2label[predicted_class.item()]
    confidence_percent = round(confidence.item() * 100, 2)

    return label.lower(), confidence_percent