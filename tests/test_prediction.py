"""Integration smoke test. Requires trained model artifacts in models/."""
from pathlib import Path
import pytest
from PIL import Image

from src.predict import predict_image

MODEL = Path("models/fruit_model.keras")
CLASSES = Path("models/class_names.json")


@pytest.mark.skipif(
    not MODEL.exists() or not CLASSES.exists(),
    reason="Train the model first and copy artifacts into models/."
)
def test_prediction_result_shape():
    image = Image.new("RGB", (100, 100), color=(220, 30, 30))
    result = predict_image(image)
    assert {"predicted_class", "confidence", "top_3"} <= result.keys()
    assert isinstance(result["predicted_class"], str)
    assert 0.0 <= result["confidence"] <= 1.0
    assert 1 <= len(result["top_3"]) <= 3
