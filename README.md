# Explainable Medical Image Classification using DINOv3

An explainable deep learning framework for pneumonia detection and subtype classification from chest X-ray images using DINOv3 Vision Transformers and attention-based localization. The framework combines transformer-based feature extraction, medical attention localization, and threshold optimization to improve both classification performance and interpretability in AI-assisted healthcare diagnosis systems.

## Project Overview

This project introduces a two-stage explainable medical image classification pipeline for chest X-ray analysis.

The framework performs:

- Stage 1 --> Binary classification  
  `NORMAL vs PNEUMONIA`

- Stage 2 --> Pneumonia subtype classification  
  `BACTERIAL vs VIRAL`

The model uses a DINOv3 Vision Transformer backbone combined with a Medical Attention Localization Module to generate clinically meaningful attention maps while improving classification accuracy.

## Key Features

- DINOv3 Vision Transformer backbone
- Two-stage hierarchical classification pipeline
- Attention-guided medical localization
- Explainability-driven predictions
- Threshold optimization for recall improvement
- Attention heatmap visualization
- Transformer-based global contextual learning
- False negative reduction strategy
- Three-class pneumonia evaluation
- Confusion matrix and metric analysis

## Motivation

Pneumonia is one of the most common respiratory diseases and delayed diagnosis can significantly increase clinical risk. Traditional CNN-based systems often behave like black boxes and lack interpretability. This project focuses on building a transparent and explainable AI system that not only performs accurate classification but also highlights clinically relevant lung regions responsible for predictions.

## Architecture

The framework consists of:

### 1. DINOv3 Transformer Backbone

The chest X-ray image is divided into patches and processed using transformer self-attention mechanisms to capture long-range spatial dependencies across lung regions.

### 2. Medical Attention Localization Module

A trainable attention mechanism learns clinically relevant spatial regions directly during training instead of relying on post-hoc explainability methods such as Grad-CAM.

### 3. Threshold Optimization

Threshold tuning is introduced during Stage 1 classification to reduce false negatives and improve recall performance for pneumonia detection.

## Dataset

Dataset used:

- Chest X-ray Pneumonia Dataset (Kaggle)

Classes:

- NORMAL
- BACTERIAL PNEUMONIA
- VIRAL PNEUMONIA

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Image Size | 224 × 224 |
| Loss Function | Cross Entropy |
| Epochs | 25 |

## Data Augmentation

The following augmentations were applied during training:

- Random horizontal flip
- Random rotation
- Affine transformations
- Brightness adjustment
- Image normalization

These augmentations improved generalization and reduced overfitting.

## Final Results

### Proposed DINOv3 Framework

| Metric | Value |
|---|---|
| Accuracy | 86.04% |
| Balanced Accuracy | 86.56% |
| Precision | 88.02% |
| Recall | 86.04% |
| Weighted F1-score | 86.23% |

## Comparison with ResNet50 Baseline

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| ResNet50 + Grad-CAM | 79.64 | 84.27 | 79.60 | 79.72 |
| DINOv3 + Localization | 86.04 | 88.02 | 86.04 | 86.23 |

The proposed transformer-based framework significantly outperformed the CNN baseline across all evaluation metrics.

## Explainability

Unlike Grad-CAM-based approaches that generate explanations after prediction, the proposed attention localization module learns disease-relevant spatial attention during training itself.

The generated attention maps:

- Highlight pneumonia-affected lung regions
- Reduce attention to irrelevant background areas
- Improve clinical transparency
- Produce anatomically meaningful visual explanations

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DINOv3 Vision Transformers
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## Running the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/project-name.git
cd project-name
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Inference

```bash
python inference.py
```

---

## Sample Outputs

- Attention heatmaps
- Confusion matrices
- Pneumonia localization maps
- Classification reports
- Threshold analysis plots

## Future Work

Future improvements include:

- Multi-disease thoracic abnormality detection
- Training on NIH ChestX-ray dataset
- Lightweight transformer optimization
- Advanced explainability techniques
- Attention-guided diagnostic reasoning

## Authors

- Shivangi Mittal
- Saumya Pandey
- Tasmiya Fathima

UMass Amherst

## References

Key references include:

- DINO self-supervised transformers
- Vision Transformers (ViT)
- Grad-CAM and Grad-CAM++
- CheXNet
- ResNet architectures

Full references are available in the project report.
