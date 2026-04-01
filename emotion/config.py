#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared configuration for all emotion recognition training scripts.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Emotion categories (canonical lowercase names)
CATEGORIES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CATEGORIES)

# Image sizes
IMG_SIZE_CNN = (48, 48)
IMG_SIZE_TRANSFER = (224, 224)
INPUT_SHAPE_GRAY = (*IMG_SIZE_CNN, 1)
INPUT_SHAPE_RGB = (*IMG_SIZE_TRANSFER, 3)

# Training defaults
RANDOM_SEED = 42
BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Augmentation defaults (moderate values)
AUGMENTATION_PARAMS = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "rescale": 1.0 / 255,
}
