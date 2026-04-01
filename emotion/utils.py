#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utility functions for emotion recognition training scripts.
Consolidates duplicated code from across the codebase.
"""

import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.feature
from skimage.color import rgb2gray, gray2rgb

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Flatten, Dense, Conv2D, MaxPooling2D,
    Dropout, BatchNormalization, GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from config import CATEGORIES, RANDOM_SEED


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def custom_img_prep(img):
    """Apply Canny edge detection preprocessing to an image.

    Converts to grayscale, applies Canny edge detection with sigma=2,
    then converts back to RGB for model compatibility.
    """
    img_2d = rgb2gray(img)
    canny_edges = skimage.feature.canny(img_2d, sigma=2)
    canny_edges = canny_edges.astype(int)
    canny_edges[canny_edges == 1] = 255
    canny_edges = canny_edges.astype(float)
    return gray2rgb(canny_edges)


def load_images_from_directory(data_dir, target_size, color_mode="rgb"):
    """Load images and labels from a directory with class subdirectories.

    Args:
        data_dir: Path to directory containing class subdirectories.
        target_size: Tuple (width, height) for resizing.
        color_mode: "rgb" for color images, "gray" for grayscale.

    Returns:
        images: numpy array of images.
        labels: numpy array of string labels.
    """
    images = []
    labels = []

    for directory_path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(directory_path):
            continue
        label = os.path.basename(directory_path)
        for img_path in sorted(glob.glob(os.path.join(directory_path, "*.*"))):
            try:
                if color_mode == "gray":
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
                if color_mode == "rgb" and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
            except (cv2.error, ValueError, OSError) as e:
                print(f"Warning: skipping {img_path}: {e}")

    return np.array(images), np.array(labels)


# ---------------------------------------------------------------------------
# Model building utilities
# ---------------------------------------------------------------------------

def build_cnn(input_shape, num_classes, show_summary=True):
    """Build the custom CNN architecture with L2 regularization.

    Architecture: 6 conv layers with ELU activation, BatchNorm,
    progressive dropout (0.35 -> 0.6), and L2-regularized dense layers.
    """
    reg = l2(1e-4)
    model_in = Input(shape=input_shape, name="input_CNN")

    conv2d_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same',
                       kernel_initializer='he_normal', name='conv2d_1')(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1')(conv2d_1)
    conv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same',
                       kernel_initializer='he_normal', name='conv2d_2')(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2')(conv2d_2)

    maxpool2d_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1')(batchnorm_2)
    dropout_1 = Dropout(0.35, name='dropout_1')(maxpool2d_1)

    conv2d_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same',
                       kernel_initializer='he_normal', name='conv2d_3')(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3')(conv2d_3)
    conv2d_4 = Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same',
                       kernel_initializer='he_normal', name='conv2d_4')(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4')(conv2d_4)

    maxpool2d_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2')(batchnorm_4)
    dropout_2 = Dropout(0.4, name='dropout_2')(maxpool2d_2)

    conv2d_5 = Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same',
                       kernel_initializer='he_normal', name='conv2d_5')(dropout_2)
    batchnorm_5 = BatchNormalization(name='batchnorm_5')(conv2d_5)
    conv2d_6 = Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same',
                       kernel_initializer='he_normal', name='conv2d_6')(batchnorm_5)
    batchnorm_6 = BatchNormalization(name='batchnorm_6')(conv2d_6)

    maxpool2d_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3')(batchnorm_6)
    dropout_3 = Dropout(0.5, name='dropout_3')(maxpool2d_3)

    flatten = Flatten(name='flatten')(dropout_3)

    dense_1 = Dense(256, activation='elu', kernel_initializer='he_normal',
                     kernel_regularizer=reg, name='dense1')(flatten)
    batchnorm_7 = BatchNormalization(name='batchnorm_7')(dense_1)
    dropout_4 = Dropout(0.6, name='dropout_4')(batchnorm_7)

    model_out = Dense(num_classes, activation="softmax", name="output_CNN")(dropout_4)

    model = Model(inputs=model_in, outputs=model_out)

    if show_summary:
        model.summary()

    return model


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_class_weights(y_labels):
    """Compute balanced class weights from label array.

    Args:
        y_labels: 1D numpy array of integer class labels.

    Returns:
        Dictionary mapping class indices to weights.
    """
    classes = np.unique(y_labels)
    weights = compute_class_weight('balanced', classes=classes, y=y_labels)
    return dict(zip(classes.astype(int), weights))


def get_callbacks(monitor='val_loss', patience_stop=10, patience_lr=5):
    """Get standard training callbacks (EarlyStopping + ReduceLROnPlateau).

    Args:
        monitor: Metric to monitor for both callbacks.
        patience_stop: Epochs to wait before stopping.
        patience_lr: Epochs to wait before reducing learning rate.

    Returns:
        List of Keras callbacks.
    """
    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_model_history(model_name, history, epochs=None):
    """Plot training accuracy and loss curves.

    Args:
        model_name: Name for the plot title.
        history: Keras history dictionary (history.history).
        epochs: Total epochs for x-axis ticks (auto-detected if None).
    """
    if epochs is None:
        epochs = len(history.get('accuracy', history.get('loss', [])))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(history['accuracy'])), history['accuracy'], 'r')
    plt.plot(np.arange(1, len(history['val_accuracy']) + 1), history['val_accuracy'], 'g')
    plt.xticks(np.arange(0, epochs + 1, max(1, epochs // 10)))
    plt.title(f'{model_name} - Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['loss']) + 1), history['loss'], 'r')
    plt.plot(np.arange(1, len(history['val_loss']) + 1), history['val_loss'], 'g')
    plt.xticks(np.arange(0, epochs + 1, max(1, epochs // 10)))
    plt.title(f'{model_name} - Training Loss vs. Validation Loss')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='best')

    plt.show()


def predict_one_image(img, model, target_size=(224, 224), categories=None):
    """Predict emotion for a single image.

    Args:
        img: Input image (numpy array, BGR from cv2).
        model: Trained Keras model.
        target_size: (width, height) expected by the model.
        categories: List of category names for label mapping.

    Returns:
        class_num: Predicted class index.
        confidence: Prediction confidence.
    """
    if categories is None:
        categories = CATEGORIES
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (1, *target_size, 3 if len(img.shape) == 3 else 1))
    img = img / 255.
    pred = model.predict(img)
    class_num = np.argmax(pred, axis=1)
    return class_num, np.max(pred)


# ---------------------------------------------------------------------------
# VGG16 feature extraction with GlobalAveragePooling
# ---------------------------------------------------------------------------

def build_vgg16_feature_extractor(input_size=224):
    """Build VGG16 feature extractor with GlobalAveragePooling2D.

    Uses GlobalAveragePooling2D instead of Flatten to reduce
    feature dimensionality from 25,088 to 512.

    Args:
        input_size: Input image dimension (square).

    Returns:
        Keras Model that outputs 512-dim feature vectors.
    """
    from tensorflow.keras.applications.vgg16 import VGG16

    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(input_size, input_size, 3))
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=x)
    return model
