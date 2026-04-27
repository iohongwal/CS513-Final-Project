from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import cv2
import joblib
import numpy as np
import mediapipe as mp
from skimage.feature import hog, local_binary_pattern
from mediapipe.tasks          import python as mp_python
from mediapipe.tasks.python   import vision as mp_vision

# ---------------------------
# Model artefact paths
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", "notebooks/output_model/best_model_F41.pkl"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", "notebooks/output_model/scaler_F41.pkl"))
LE_PATH = Path(os.getenv("LE_PATH", "notebooks/output_model/label_encoder_F41.pkl"))

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LE_PATH)

# ---------------------------
# MediaPipe FaceLandmarker
# ---------------------------
mp_tasks = mp.tasks
mp_vision = mp.tasks.vision
#mp_image = mp.Image

FACELANDMARKER_MODEL = "notebooks/face_landmarker_models/face_landmarker.task"
FACELANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
if not os.path.exists(FACELANDMARKER_MODEL):
    print(f"Downloading FaceLandmarker model → {FACELANDMARKER_MODEL} …")
    urllib.request.urlretrieve(FACELANDMARKER_URL, FACELANDMARKER_MODEL)
    print(f"  Done  ({os.path.getsize(FACELANDMARKER_MODEL)/1e6:.1f} MB)")
else:
    print(f"Model found: {FACELANDMARKER_MODEL}  ({os.path.getsize(FACELANDMARKER_MODEL)/1e6:.1f} MB)")

options = mp_vision.FaceLandmarkerOptions(
    base_options= mp_python.BaseOptions(model_asset_path=FACELANDMARKER_MODEL),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

# ---------------------------
# Geometry features (F25–F41)
# ---------------------------

def _extract_geometry(gray: np.ndarray) -> np.ndarray:
    """
    Compute the 17 geometry features used in the F41 setup.
    Replace the placeholder with your exact F25–F41 logic.
    """
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(mp_img)

    if not result.face_landmarks:
        return np.full(17, np.nan, dtype=float)

    lm = result.face_landmarks[0]

    # TODO: replace this with your actual F25–F41 feature computation
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    geom = np.concatenate([xs[:9], ys[:8]])  # placeholder 17 dims

    return geom.astype(float)

# ---------------------------
# Texture features (F01–F24)
# ---------------------------

def _extract_texture(gray: np.ndarray) -> np.ndarray:
    """Compute HOG + LBP features (total 24 dims)."""
    gray_resized = cv2.resize(gray, (48, 48))

    hog_vec = hog(
        gray_resized,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        orientations=9,
        feature_vector=True,
    )

    lbp = local_binary_pattern(gray_resized, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=15, range=(0, 15), density=True)

    tex = np.concatenate([hog_vec, hist])[:24]
    return tex.astype(float)

# ---------------------------
# Emoji mapping + public API
# ---------------------------

EMOJI_MAP = {
    "angry": "😠",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😮",
}

def predict_expression_and_emoji(bgr_image: np.ndarray) -> Dict:
    """
    Run full F41 pipeline on a BGR image and return label + emoji.

    Returns
    -------
    {
      "emotion": str,
      "emoji": str,
      "probabilities": {class: prob}
    }
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    geom = _extract_geometry(gray)
    tex = _extract_texture(gray)

    if np.isnan(geom).any():
        return {
            "emotion": "no_face",
            "emoji": "❓",
            "probabilities": {},
        }

    features = np.concatenate([tex, geom])  # 24 + 17 = 41 dims
    features_scaled = scaler.transform([features])

    pred_int = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    emotion = label_encoder.inverse_transform([pred_int])[0]

    emoji = EMOJI_MAP.get(emotion, "❓")

    return {
        "emotion": emotion,
        "emoji": emoji,
        "probabilities": {
            cls: float(p) for cls, p in zip(label_encoder.classes_, proba)
        },
    }