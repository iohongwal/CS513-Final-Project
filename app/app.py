from __future__ import annotations

import numpy as np
from PIL import Image
import gradio as gr

from .inference import predict_expression_and_emoji


def _predict_from_image(img: Image.Image):
    if img is None:
        return "no_image ❓", {}

    # PIL (RGB) → numpy (BGR)
    rgb = np.array(img)
    bgr = rgb[..., ::-1]

    result = predict_expression_and_emoji(bgr)
    emotion = result["emotion"]
    emoji = result["emoji"]
    probs = result["probabilities"]

    return f"{emotion} {emoji}", probs


demo = gr.Interface(
    fn=_predict_from_image,
    inputs=gr.Image(
        type="pil",
        sources=["upload", "webcam"],
        label="Upload face photo or use webcam",
    ),
    outputs=[
        gr.Text(label="Predicted Expression + Emoji"),
        gr.Label(label="Class Probabilities", num_top_classes=5),
    ],
    title="Facial Expression → Emoji",
    description=(
        "Upload a face image; the app predicts the emotion and returns an emoji."
    ),
)

if __name__ == "__main__":
    demo.launch()