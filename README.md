# FreqCheck Lite: Simple Frequency-Based Image Forensics

**Team members**
- Pitta Abhiram — 230104070
- Pankaj Chauhan — 230104069

## What this project does

This is a small, explainable image-forensics project. It extracts a few image statistics—including high-frequency energy from the 2D Fourier transform—and trains a Random Forest to distinguish real images from generated/fake images. It is intentionally kept small enough to run on a normal laptop without a GPU.

This is **not the original FreqNet neural-network architecture**. It is a lightweight baseline inspired by the idea that frequency information can help image forensics. Its usefulness must be established with experiments; it is not guaranteed to generalize to unseen generators.

## Dataset

Prepare a dataset with the following structure. Put images you have permission to use in the folders; do not commit a large dataset to GitHub.

```text
dataset/
  train/
    real/
    fake/
  test/
    real/
    fake/
```

Use images from the same image size range where possible. Keep test images separate from training images. For a stronger test, use a different generator/source in the test split than the training split when the dataset supports it. The project expects `real` and `fake` labels only.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train

```bash
python train.py --train_dir dataset/train --model_path artifacts/freqcheck.joblib
```

The script prints validation metrics using a held-out portion of the training directory and saves the model plus feature metadata. Do not use the final test set to choose model settings.

## Evaluate

```bash
python evaluate.py --test_dir dataset/test --model_path artifacts/freqcheck.joblib --output metrics.json
```

Metrics include accuracy, precision, recall, F1, ROC-AUC, average precision, and the confusion matrix. ROC-AUC/AP are reported as unavailable if the evaluation set contains only one class.

## What the features mean

- **High-frequency ratio:** fraction of Fourier magnitude energy outside a central low-frequency region.
- **Radial frequency statistics:** average spectral magnitude in coarse distance-from-centre bands.
- **Color statistics:** mean and standard deviation of RGB channels.
- **Edge statistics:** variance of a Laplacian response, a rough measure of local sharpness/texture.

These are simple signals, not proof that an image is fake. Compression, resizing, camera pipelines, and dataset bias can all change them.

## Small experiment plan

1. Train a model using frequency features only.
2. Train a model using all features.
3. Compare both on the same held-out test set.
4. Evaluate a robustness copy of the test images saved as JPEG at quality 75. Keep the original test set unchanged.
5. Report the actual metrics, dataset source, split method, and limitations. Do not write results before running the code.

## Limitations

This is a classical machine-learning baseline, not a large deep-learning model. Performance depends strongly on dataset quality and split design. It may learn dataset-specific compression or source differences instead of reliable evidence of synthesis. Results from one dataset should not be described as general deepfake detection.
