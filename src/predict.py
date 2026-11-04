"""Load the trained FruitScan CNN and predict an uploaded image."""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import tensorflow as tf
from PIL import Image

from src.preprocessing import prepare_image


@lru_cache(maxsize=2)
def load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Ask Participant 1 to train the CNN and copy "
            "fruit_model.keras into models/."
        )
    return tf.keras.models.load_model(path)


def load_class_names(class_names_path: str) -> list[str]:
    path = Path(class_names_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Class list not found: {path}. Ask Participant 1 to provide class_names.json."
        )
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must contain a non-empty JSON list.")
    return names


def predict_image(
    image: Image.Image,
    model_path: str = "models/fruit_model.keras",
    class_names_path: str = "models/class_names.json",
    top_k: int = 3
) -> dict:
    model = load_model(model_path)
    class_names = load_class_names(class_names_path)
    batch = prepare_image(image)
    probabilities = model.predict(batch, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "The number of model outputs does not match class_names.json. "
            "Use the class list generated with this exact model."
        )

    top_indices = np.argsort(probabilities)[::-1][:min(top_k, len(class_names))]
    top = [
        {"class": class_names[int(i)], "confidence": float(probabilities[i])}
        for i in top_indices
    ]
    best = top[0]
    return {
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "top_3": top
    }
