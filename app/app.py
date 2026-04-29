from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = (Path(__file__).resolve().parents[1])
_AVIF_PLUGIN_NAME = "PIL.AvifImagePlugin"
if _AVIF_PLUGIN_NAME not in sys.modules:
    sys.modules[_AVIF_PLUGIN_NAME] = types.ModuleType(_AVIF_PLUGIN_NAME)

def build_demo():
    _removed_sys_path_entries = [
        entry for entry in sys.path if Path(entry or ".").resolve() == _PROJECT_ROOT
    ]
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or ".").resolve() != _PROJECT_ROOT
    ]
    original_image_init = Image.init
    Image.init = lambda: None
    try:
        import gradio as gr
    finally:
        Image.init = original_image_init
        for entry in reversed(_removed_sys_path_entries):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    return gr.Interface(
        fn=_predict_from_image,
        inputs=gr.Image(
            type="pil",
            sources=["upload", "webcam"],
            label="Upload face photo or use webcam",
        ),
        outputs=[
            gr.Textbox(label="Predicted Expression + Emoji"),
            gr.Label(label="Class Probabilities", num_top_classes=5),
        ],
        title="Facial Expression → Emoji",
        description=(
            "Upload a face image; the app predicts the emotion and returns an emoji."
        ),
    )

demo = None


def _predict_expression_and_emoji(bgr_image: np.ndarray):
    from .inference import predict_expression_and_emoji

    return predict_expression_and_emoji(bgr_image)


def _predict_from_image(img: Image.Image):
    if img is None:
        return "no_image ❓", {}

    # PIL (RGB) → numpy (BGR)
    rgb = np.array(img)
    bgr = rgb[..., ::-1]

    result = _predict_expression_and_emoji(bgr)
    if result.get("error"):
        return f"unavailable ⚠️ {result['error']}", {}

    emotion = result["emotion"]
    emoji = result["emoji"]
    probs = result["probabilities"]

    return f"{emotion} {emoji}", probs

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()