# Facial Emotion Recognition

Multi-approach comparison of deep learning and traditional ML methods for facial emotion classification.

**Author:** Mary Akshara Allam

## Overview

This project classifies facial expressions into **5 emotion categories**: Happy, Fear, Sadness, Surprise, and Anger. It compares several architectures and training strategies:

| Script | Architecture | Dataset | Input Size | Approach |
|--------|-------------|---------|------------|----------|
| `cnn.py` | Custom CNN (6 conv layers) | CK+48/images | 48x48 grayscale | End-to-end training |
| `cnn_rf.py` | CNN + Random Forest | CK+48/data | 128x128 RGB | CNN feature extraction + RF |
| `vgg16.py` | VGG16-like (from scratch) | data/ | 224x224 RGB | Custom architecture, 6 classes |
| `vgg_canny.py` | VGG-like + Canny edges | images/ | 224x224 RGB | Edge preprocessing + CNN |
| `vgg_RF.py` | VGG16 (pretrained) + RF/XGBoost | CK+48/data | 224x224 RGB | Transfer learning + ensemble |
| `emotion.py` | VGG16 (pretrained) + RF/XGBoost | aligned/ | 224x224 RGB | Transfer learning + ensemble |
| `resnet.py` | ResNet50 + VGG19 comparison | CK+48/images | 224x224 RGB | Transfer learning |
| `emotion_detection.py` | CNN + VGG19 + ResNet50 + XGBoost | CK+48/images | 48x48 & 224x224 | Full pipeline comparison |
| `test_emotion.py` | Loads trained CNN | CK+48/test_images | 48x48 grayscale | Inference only |

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
emotion/                        # Main project directory (run scripts from here)
├── emotion_detection.py        # Full pipeline: CNN + VGG19 + ResNet50 + XGBoost
├── resnet.py                   # ResNet50 vs VGG19 comparison
├── cnn.py                      # Custom CNN training
├── cnn_rf.py                   # CNN features + Random Forest
├── vgg16.py                    # VGG16-like architecture from scratch
├── vgg_canny.py                # VGG-like with Canny edge preprocessing
├── vgg_RF.py                   # Pretrained VGG16 + RF/XGBoost
├── emotion.py                  # Pretrained VGG16 + RF/XGBoost (aligned dataset)
├── test_emotion.py             # Test predictions with saved model
│
├── CK+48/                      # CK+ Extended dataset (48x48)
│   ├── images/                 # 607 images, 5 emotions (Anger, Fear, Happy, Sadness, Surprise)
│   ├── data/                   # FER-style dataset, 6 emotions (adds Neutral)
│   │   ├── train/              # ~28,273 training images
│   │   └── validation/         # Validation split
│   ├── test_images/            # Small test subset
│   ├── cnn_CK+48_5emo.h5      # Saved CNN model (5 emotions)
│   └── cnn_CK+48_7emo.h5      # Saved CNN model (7 emotions)
│
├── data/                        # FER-style dataset, 6 emotions
│   ├── train/                   # ~28,273 images (Angry, Fear, Happy, Neutral, Sad, Surprise)
│   └── validation/
│
├── images/                      # FER2013-style dataset, 7 emotions
│   ├── train/                   # ~33,465 images (adds Disgust)
│   ├── validation/
│   └── test_images/
│
├── aligned/                     # Pre-aligned face dataset, 7 classes (numeric labels)
│   ├── train/
│   └── test/
│
├── RF_model.sav                 # Saved Random Forest model
├── model_XG.sav                 # Saved XGBoost model
└── *.png / *.jpg                # Generated plots and visualizations
```

## Datasets

| Dataset | Location | Images | Emotions | Notes |
|---------|----------|--------|----------|-------|
| **CK+48** | `CK+48/images/` | 607 | 5 | Lab-controlled, 48x48 grayscale |
| **CK+48 Extended** | `CK+48/data/` | ~28K | 6 | Adds Neutral |
| **FER-style** | `data/` | ~28K | 6 | Same distribution as CK+48/data |
| **FER2013-style** | `images/` | ~33K | 7 | Adds Disgust |
| **Aligned Faces** | `aligned/` | ~12K | 7 | Numerically labeled (1-7) |
| **Test Subset** | `CK+48/test_images/` | ~800 | 5 | Quick validation |

> **Note:** `CK+48/data/` and `data/` contain similar data. Consider consolidating to save disk space.

## Usage

All scripts should be run from the `emotion/` directory:

```bash
cd emotion
```

### Train the full comparison pipeline (CNN + VGG19 + ResNet50 + XGBoost)

```bash
python emotion_detection.py
```

### Train individual models

```bash
python cnn.py          # Custom CNN on CK+48
python vgg16.py        # VGG16-like from scratch
python vgg_canny.py    # VGG with Canny edge preprocessing
python resnet.py       # ResNet50 vs VGG19 comparison
```

### Train ensemble models (deep features + traditional ML)

```bash
python cnn_rf.py       # CNN features + Random Forest
python vgg_RF.py       # VGG16 features + RF/XGBoost
python emotion.py      # VGG16 features + RF/XGBoost (aligned data)
```

### Run inference on test images

```bash
python test_emotion.py
```

## Models and Approaches

### Custom CNN (`cnn.py`, `emotion_detection.py`)
- 6 convolutional layers with ELU activation and He initialization
- BatchNormalization after each conv layer
- Progressive dropout (0.35 to 0.6)
- Input: 48x48 grayscale images

### VGG19 Transfer Learning (`resnet.py`, `emotion_detection.py`)
- Pretrained ImageNet weights
- Last 4 layers unfrozen for fine-tuning
- Custom dense head: 1024 -> 1024 -> num_classes
- Input: 224x224 RGB images

### ResNet50 Transfer Learning (`resnet.py`, `emotion_detection.py`)
- Pretrained ImageNet weights with frozen base
- BatchNormalization between dense layers
- Custom head: 2048 -> 1024 -> num_classes
- Input: 224x224 RGB images

### Ensemble Methods (`emotion.py`, `vgg_RF.py`, `cnn_rf.py`)
- Deep CNN/VGG16 used as feature extractor (all layers frozen)
- Extracted features fed to Random Forest or XGBoost
- Useful when dataset is too small for end-to-end fine-tuning

### Canny Edge Preprocessing (`vgg_canny.py`, `emotion_detection.py`)
- Applies Canny edge detection (sigma=2) as preprocessing
- Converts edge maps to RGB for VGG-compatible input
- Experimental approach to focus on facial contours

## Output

Each training script generates:
- **Accuracy/Loss plots** showing training vs. validation curves
- **Confusion matrices** for per-class performance analysis
- **Classification reports** with precision, recall, and F1-score
- **Saved models** (.h5 for Keras, .sav for sklearn/xgboost)
