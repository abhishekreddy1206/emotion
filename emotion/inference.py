#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified inference module for emotion recognition models.

Provides a single EmotionPredictor class that handles model loading,
image preprocessing, face detection, and prediction. Used by the
FastAPI backend and can be imported by any script.

Usage:
    from inference import EmotionPredictor

    predictor = EmotionPredictor("models/cnn_6emo.h5", "models/cnn_6emo_meta.json")
    result = predictor.predict(image)
    # result = {"angry": 0.05, "fear": 0.02, "happy": 0.85, ...}

    # For webcam frames with face detection:
    results = predictor.predict_from_frame(frame)
    # results = [{"bbox": [x, y, w, h], "emotions": {...}}, ...]
"""

import json
import cv2
import numpy as np
import tensorflow as tf

from config import CATEGORIES


class EmotionPredictor:
    """Unified emotion prediction interface.

    Handles model loading, preprocessing, and prediction for any
    trained emotion recognition model. Reads metadata sidecar JSON
    to determine input size, color mode, and category labels.
    """

    def __init__(self, model_path, metadata_path=None):
        """Initialize the predictor.

        Args:
            model_path: Path to a saved Keras .h5 model or TFLite model.
            metadata_path: Path to JSON metadata file (optional).
                If not provided, defaults to standard categories and
                infers input shape from the model.
        """
        self.model_path = model_path

        if metadata_path:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.categories = self.metadata["categories"]
            self.input_size = tuple(self.metadata["input_size"])
        else:
            self.categories = CATEGORIES
            self.input_size = None

        # Load model
        if model_path.endswith(".tflite"):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
            if self.input_size is None:
                shape = self.input_details[0]['shape']
                self.input_size = (shape[1], shape[2], shape[3])
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.interpreter = None
            if self.input_size is None:
                shape = self.model.input_shape
                self.input_size = (shape[1], shape[2], shape[3])

        self.is_grayscale = self.input_size[-1] == 1
        self.target_hw = (self.input_size[0], self.input_size[1])

        # Load face detector (Haar cascade, bundled with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _preprocess(self, image):
        """Preprocess a single face image for model input.

        Args:
            image: numpy array (BGR from cv2 or RGB).

        Returns:
            Preprocessed image array ready for model.predict().
        """
        if self.is_grayscale:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, self.target_hw, interpolation=cv2.INTER_CUBIC)
            image = np.expand_dims(image, axis=-1)
        else:
            image = cv2.resize(image, self.target_hw, interpolation=cv2.INTER_CUBIC)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def predict(self, image):
        """Predict emotions for a single face image.

        Args:
            image: numpy array (face crop, BGR or grayscale).

        Returns:
            Dictionary mapping emotion names to confidence scores.
        """
        processed = self._preprocess(image)

        if self.interpreter:
            self.interpreter.set_tensor(self.input_details[0]['index'], processed)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            predictions = self.model.predict(processed, verbose=0)[0]

        return dict(zip(self.categories, predictions.tolist()))

    def _detect_faces(self, frame):
        """Detect faces in a frame using Haar cascade.

        Args:
            frame: Full image/frame (BGR numpy array).

        Returns:
            List of (bbox, face_crop) tuples.
            bbox is [x, y, w, h] in pixel coordinates.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        results = []
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            results.append(([int(x), int(y), int(w), int(h)], face_crop))

        return results

    def predict_from_frame(self, frame):
        """Detect faces and predict emotions for each face in a frame.

        Args:
            frame: Full image/video frame (BGR numpy array).

        Returns:
            List of dicts with "bbox" and "emotions" keys.
        """
        faces = self._detect_faces(frame)
        return [
            {"bbox": bbox, "emotions": self.predict(face)}
            for bbox, face in faces
        ]


def save_model_metadata(filepath, model_name, input_size, categories,
                        preprocessing_info, accuracy=None):
    """Save model metadata as a JSON sidecar file.

    Args:
        filepath: Path for the JSON file.
        model_name: Name/identifier for the model.
        input_size: List [height, width, channels].
        categories: List of emotion category names.
        preprocessing_info: String describing preprocessing steps.
        accuracy: Optional validation accuracy.
    """
    from datetime import datetime

    metadata = {
        "model_name": model_name,
        "input_size": list(input_size),
        "categories": categories,
        "preprocessing": preprocessing_info,
        "accuracy": accuracy,
        "date_trained": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata saved to {filepath}")


def export_tflite(keras_model, output_path):
    """Export a Keras model to TensorFlow Lite format.

    Args:
        keras_model: Trained Keras model.
        output_path: Path for the .tflite output file.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {output_path} ({len(tflite_model) / 1024:.0f} KB)")
