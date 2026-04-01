# Facial Emotion Recognition

Multi-approach comparison of deep learning and traditional ML methods for facial emotion classification, with a mobile-friendly web app for real-time inference.

## Overview

This project classifies facial expressions into **6 emotion categories**: Angry, Fear, Happy, Neutral, Sad, and Surprise. It compares several architectures and training strategies, and includes a FastAPI-powered web app with live webcam support.

### Training Scripts

| Script | Architecture | Input Size | Approach |
|--------|-------------|------------|----------|
| `cnn.py` | Custom CNN (6 conv layers) | 48x48 grayscale | End-to-end training |
| `cnn_rf.py` | CNN + Random Forest | 224x224 RGB | CNN feature extraction + RF |
| `vgg16.py` | VGG16-like (from scratch) | 224x224 RGB | Custom architecture, 6 classes |
| `vgg_canny.py` | VGG-like + Canny edges | 224x224 RGB | Edge preprocessing + CNN |
| `vgg_RF.py` | VGG16 (pretrained) + RF/XGBoost | 224x224 RGB | Transfer learning + ensemble |
| `emotion.py` | VGG16 (pretrained) + RF/XGBoost | 224x224 RGB | Transfer learning + ensemble |
| `resnet.py` | ResNet50 + VGG19 comparison | 224x224 RGB | Transfer learning |
| `emotion_detection.py` | CNN + VGG19 + ResNet50 + XGBoost | 48x48 & 224x224 | Full pipeline comparison |
| `test_emotion.py` | Loads trained CNN | 48x48 grayscale | Inference only |

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
├── config.py                   # Shared configuration (paths, categories, defaults)
├── utils.py                    # Shared utility functions (models, preprocessing, plotting)
├── inference.py                # Unified prediction API (EmotionPredictor class)
│
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
├── app/                        # Web application
│   ├── api.py                  # FastAPI backend (REST + WebSocket)
│   └── frontend/               # Mobile-friendly web UI
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── models/                     # Saved trained models + metadata
│
├── utils/                      # Data utilities
│   └── data_audit.py           # Dataset deduplication & audit tool
│
├── data/                       # Canonical dataset (6 emotions)
│   ├── train/                  # Training images by class
│   └── validation/             # Validation images by class
│
├── CK+48/                      # CK+ Extended dataset
│   ├── images/                 # 607 images, 5 emotions
│   └── data/                   # FER-style, 6 emotions
│
├── images/                     # FER2013-style dataset, 7 emotions
│
└── aligned/                    # Pre-aligned face dataset
```

## Usage

All scripts should be run from the `emotion/` directory:

```bash
cd emotion
```

### Train the full comparison pipeline

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

### Train ensemble models

```bash
python cnn_rf.py       # CNN features + Random Forest
python vgg_RF.py       # VGG16 features + RF/XGBoost
python emotion.py      # VGG16 features + RF/XGBoost (aligned data)
```

### Run the web app

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in a browser. Supports:
- **Image upload**: Drag-and-drop or file picker
- **Live camera**: Real-time webcam emotion detection via WebSocket
- **Mobile**: Responsive design with front/back camera flip

### Run inference programmatically

```python
from inference import EmotionPredictor

predictor = EmotionPredictor("models/cnn_6emo.h5", "models/cnn_6emo_meta.json")
result = predictor.predict(image)
# {"angry": 0.05, "fear": 0.02, "happy": 0.85, "neutral": 0.03, "sad": 0.03, "surprise": 0.02}
```

### Audit and deduplicate datasets

```bash
python -m utils.data_audit           # Report only
python -m utils.data_audit --merge   # Merge unique images into data/
```

## Model Improvements

The following improvements have been applied across all training scripts:

- **Class balancing**: `compute_class_weight('balanced')` used in all `model.fit()` calls
- **L2 regularization**: `kernel_regularizer=l2(1e-4)` on all Dense layers
- **Early stopping**: `EarlyStopping` + `ReduceLROnPlateau` callbacks on all models
- **Dimensionality reduction**: VGG16 features reduced from 25,088 to 512 dims via `GlobalAveragePooling2D`
- **Normalization fix**: All `ImageDataGenerator` instances include `rescale=1./255`
- **Stratified splits**: Reproducible train/test splits with `random_state=42`
- **Cross-validation**: 5-fold `StratifiedKFold` for RF/XGBoost classifiers
- **Error handling**: Bare `except: pass` replaced with specific exception handling

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve frontend |
| GET | `/health` | Health check |
| POST | `/predict/image` | Upload image, get emotion predictions |
| WS | `/predict/stream` | Stream webcam frames for real-time predictions |

## Output

Each training script generates:
- **Accuracy/Loss plots** showing training vs. validation curves
- **Confusion matrices** for per-class performance analysis
- **Classification reports** with precision, recall, and F1-score
- **Saved models** (.h5 for Keras, .sav for sklearn/xgboost)
- **Cross-validation scores** with mean and standard deviation (for ensemble models)
