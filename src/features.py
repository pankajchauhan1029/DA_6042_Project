from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def extract_features(image_path, size=128, mode="all"):
    """Extract deterministic, explainable image features for a single image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)

    centered = gray - gray.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    max_radius = np.sqrt(2) * size / 2
    total_energy = float(np.sum(spectrum ** 2)) + 1e-8
    high_mask = radius >= size * 0.18
    high_frequency_ratio = float(np.sum(spectrum[high_mask] ** 2) / total_energy)

    frequency_bands = []
    for lower, upper in [(0.00, 0.08), (0.08, 0.18), (0.18, 0.32), (0.32, 0.50), (0.50, 1.01)]:
        mask = (radius / max_radius >= lower) & (radius / max_radius < upper)
        values = spectrum[mask]
        frequency_bands.append(float(np.mean(values)) if values.size else 0.0)

    if mode == "frequency":
        return np.asarray([high_frequency_ratio, *frequency_bands], dtype=np.float32)

    color_features = []
    for channel in range(3):
        color_features.extend([float(rgb[:, :, channel].mean()), float(rgb[:, :, channel].std())])

    lap = laplace(gray)
    edge_features = [float(np.var(lap)), float(np.mean(np.abs(lap)))]
    return np.asarray([high_frequency_ratio, *frequency_bands, *color_features, *edge_features], dtype=np.float32)


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_labeled_folder(root, mode="all"):
    """Expected folder structure: root/real/* and root/fake/*."""
    root = Path(root)
    paths, labels, features = [], [], []
    for class_name, label in [("real", 0), ("fake", 1)]:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder: {class_dir}")
        images = list_images(class_dir)
        if not images:
            raise ValueError(f"No supported images found in {class_dir}")
        for path in images:
            try:
                features.append(extract_features(path, mode=mode))
                paths.append(str(path))
                labels.append(label)
            except (OSError, ValueError) as error:
                print(f"Skipping unreadable image {path}: {error}")
    if len(set(labels)) != 2:
        raise ValueError(f"Both real and fake images are required in {root}")
    return np.vstack(features), np.asarray(labels, dtype=np.int64), paths
