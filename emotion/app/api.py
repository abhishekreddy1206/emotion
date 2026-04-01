#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI backend for emotion recognition.

Endpoints:
    POST /predict/image   - Upload an image, get emotion predictions
    WS   /predict/stream  - Stream webcam frames, get real-time predictions
    GET  /                - Serve the frontend

Usage:
    cd emotion/
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import io
import base64

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import EmotionPredictor
from config import MODEL_SAVE_DIR, CATEGORIES

app = FastAPI(title="Emotion Recognition API", version="1.0.0")

# CORS for mobile browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Load the predictor (lazy initialization)
predictor = None


def get_predictor():
    """Lazy-load the emotion predictor."""
    global predictor
    if predictor is not None:
        return predictor

    # Try to find a model in the models directory
    model_path = None
    meta_path = None

    if os.path.isdir(MODEL_SAVE_DIR):
        for f in os.listdir(MODEL_SAVE_DIR):
            if f.endswith(".h5") or f.endswith(".tflite"):
                model_path = os.path.join(MODEL_SAVE_DIR, f)
                meta_candidate = os.path.splitext(model_path)[0] + "_meta.json"
                if os.path.exists(meta_candidate):
                    meta_path = meta_candidate
                break

    # Fallback to legacy model locations
    if model_path is None:
        base = os.path.dirname(os.path.dirname(__file__))
        legacy_paths = [
            os.path.join(base, "CK+48", "images", "cnn_CK+48_5emo.h5"),
            os.path.join(base, "CK+48", "cnn_CK+48_5emo.h5"),
            os.path.join(base, "CK+48", "cnn_CK+48_7emo.h5"),
        ]
        for p in legacy_paths:
            if os.path.exists(p):
                model_path = p
                break

    if model_path is None:
        raise RuntimeError(
            "No trained model found. Run a training script first, "
            f"or place a .h5 model in {MODEL_SAVE_DIR}/"
        )

    predictor = EmotionPredictor(model_path, meta_path)
    print(f"Loaded model: {model_path}")
    return predictor


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to a numpy array (BGR)."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Emotion Recognition API</h1><p>Frontend not found. Place index.html in app/frontend/</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict emotions from an uploaded image.

    Detects faces in the image and returns emotion predictions for each.
    If no faces are detected, returns prediction for the full image.
    """
    pred = get_predictor()
    data = await file.read()
    image = decode_image(data)

    # Try face detection first
    results = pred.predict_from_frame(image)

    if not results:
        # No faces detected, predict on full image
        emotions = pred.predict(image)
        results = [{"bbox": None, "emotions": emotions}]

    return {"predictions": results, "faces_detected": len(results)}


@app.websocket("/predict/stream")
async def predict_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time webcam emotion prediction.

    Accepts base64-encoded JPEG frames, returns JSON predictions.
    """
    await websocket.accept()
    pred = get_predictor()

    try:
        while True:
            data = await websocket.receive_text()

            # Decode base64 frame
            if "," in data:
                data = data.split(",", 1)[1]

            frame_bytes = base64.b64decode(data)
            frame = decode_image(frame_bytes)

            results = pred.predict_from_frame(frame)

            await websocket.send_json({
                "predictions": results,
                "faces_detected": len(results),
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
