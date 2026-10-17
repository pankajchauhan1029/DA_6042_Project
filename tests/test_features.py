from pathlib import Path

import numpy as np
from PIL import Image

from src.features import extract_features


def test_feature_vector_is_finite(tmp_path: Path):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = 255
    path = tmp_path / "sample.png"
    Image.fromarray(image).save(path)
    features = extract_features(path)
    assert features.ndim == 1
    assert features.size > 10
    assert np.isfinite(features).all()


def test_frequency_only_features_are_smaller(tmp_path: Path):
    image = np.random.default_rng(4).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    Image.fromarray(image).save(path)
    all_features = extract_features(path, mode="all")
    frequency_features = extract_features(path, mode="frequency")
    assert frequency_features.size < all_features.size
    assert np.isfinite(frequency_features).all()
