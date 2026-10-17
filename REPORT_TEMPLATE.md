# Project Report: FreqCheck Lite

**Team:** Pitta Abhiram (230104070), Pankaj Chauhan (230104069)

## Objective
Build a small, explainable real-versus-fake image classifier using frequency, color, and edge statistics. The aim is to study whether simple frequency features are useful, not to claim a universal deepfake detector.

## Method
Images are resized to a fixed resolution. Features include a high-frequency Fourier energy ratio, spectral-band averages, RGB mean/standard deviation, and Laplacian texture statistics. A Random Forest predicts real/fake labels. The frequency-only and all-feature settings are compared.

## Dataset and split
- Dataset/source: [fill in]
- Number of real/fake training images: [fill in]
- Number of real/fake test images: [fill in]
- Split method and source/generator separation: [fill in]

## Results
Fill this table only after running the scripts.

| Model | Accuracy | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|
| Frequency-only | [measure] | [measure] | [measure] | [measure] |
| All features | [measure] | [measure] | [measure] | [measure] |

## Discussion
[Describe what the measured results show. Mention whether adding color/edge features helped compared with frequency-only features.]

## Limitations
Results may reflect compression, image source, resizing, or dataset biases rather than synthesis itself. Performance on one dataset does not prove performance on unseen generators. No result should be reported until the dataset experiment has been run.
