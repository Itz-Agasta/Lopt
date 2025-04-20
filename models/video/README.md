# Scarlet 
> Video-based Deepfake Detection using TimeSformer and FaceForensics++ (C23)

**Scarlet** is a transformer-based deepfake detection model built on top of the `facebook/timesformer-base-finetuned-k400` architecture. It is trained on the FaceForensics++ dataset (C23 compressed videos) and fine-tuned for binary classification of real vs fake videos.

## Architecture
Scarlet leverages:
- **TimeSformer** (Time-Space Transformer) for video classification.
- A custom binary classifier head replacing the original 400-class Kinetics output layer.
- 8 sampled frames per video clip.
- Input resolution: 224x224.
- Batch size: 2 (Kaggle-compatible).

## Performance

### **Classification Report:**
          precision    recall  f1-score   support

    Real       0.95      0.97      0.96       407
    Fake       0.97      0.95      0.96       393

accuracy                           0.96       800
macro avg 0.96 0.96 0.96 800 weighted avg 0.96 0.96 0.96 800

**Confusion Matrix:**

![image](https://github.com/user-attachments/assets/df26564f-adb0-4a26-b591-fe1eec063422)

**Best Val Metrics:**
- ✅ Accuracy: **96.0%**
- ✅ F1 Score: **0.96**
- ✅ ROC AUC: *Available in metrics.json*

## Dataset
Scarlet was trained on:
- **Real Videos**: `FaceForensics++_C23/original`
- **Fake Videos**: `FaceForensics++_C23/Deepfakes`
- Split: 600 training, 800 validation (balanced real/fake)

## raining Setup
- Model: `facebook/timesformer-base-finetuned-k400`
- Loss Function: `CrossEntropyLoss`
- Optimizer: `AdamW` (`lr=3e-5`, `weight_decay=1e-2`)
- Epochs: `24`
- Early checkpoint saving for best validation accuracy.

## Checkpoints
- Best model saved as: `Scarlet.pt`
- Metrics: `outputs/metrics.json`
- Confusion Matrix: `outputs/confusion_matrix.png`
- Full report: `outputs/classification_report.txt`

##  Directory Structure
```bash
models/
└── video/
    ├── Scarlet.pt                # Trained model
    ├── Scarlet.ipynb             # Training notebook
    ├── metrics.json              # Accuracy, F1, AUC
    ├── classification_report.txt # Detailed evaluation
    ├── confusion_matrix.png      # Heatmap image
    └── README.md                
```

##  Notes

-  Optimized for Kaggle GPUs (T4, 16GB).
-   Can be fine-tuned further using FF++ HQ or DFDC.
-  Designed as a part of Lopt, a unified deepfake detection framework.

