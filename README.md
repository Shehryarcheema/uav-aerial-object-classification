# Multi-Class Object Classification on UAV Aerial Imagery

A comparative study of classical and deep learning approaches for multi-class
object classification on UAV aerial imagery, built on the VisDrone-MOT benchmark
(sequence `uav0000297_00000_v`).

Companion code for the MSc research report *"Multi-Class Object Classification
on UAV Aerial Imagery: A Comparative Study of Classical and Deep Learning
Approaches"* — De Montfort University, CSIP5403 Research Methods and Applied AI.

## Highlights

- **8,842 object crops** extracted across 146 video frames and 7 active
  categories, with a 41.6× maximum class imbalance (car: 4,329 vs. bicycle: 104).
- **Track-aware partitioning** that prevents identity leakage across frames —
  naïve random splitting inflates macro-F1 by **+52.0pp**, the largest leakage
  effect documented in this class of study.
- Three-model comparison: **HOG + Linear SVM**, **HOG + Random Forest**, and
  **fine-tuned ResNet18 with Focal Loss**.
- ResNet18 achieves the highest **macro-F1 (0.4973)** — the only model with
  non-zero F1 for bicycle (0.538), motor (0.719), and people (0.463).
- Mechanistic interpretability via **t-SNE** feature-space analysis and
  **Grad-CAM** visual explanations for every observed failure mode.

## Results

| Model | Accuracy | Top-2 Acc | Macro-F1 | Weighted-F1 | Train time |
| --- | --- | --- | --- | --- | --- |
| HOG + Linear SVM | 0.8122 | 0.8792 | 0.3876 | 0.7622 | 35.8 min |
| HOG + Random Forest | 0.8285 | 0.8534 | 0.3924 | 0.7695 | 1.8 min |
| **ResNet18 + Focal Loss** | 0.7986 | **0.9688** | **0.4973** | 0.8454 | ~5 min |

Key insight: classical models win on aggregate accuracy by exploiting the
majority classes but collapse to zero F1 on minority classes. ResNet18 trades
~3–4pp of accuracy for genuine minority-class recognition — the right
trade-off for UAV surveillance, where rare safety-critical objects must not
be missed. Its Top-2 accuracy of 96.9% confirms strong soft discriminability.

## Methodology

1. **Dataset construction** — crops extracted from VisDrone-MOT sequence
   `uav0000297_00000_v` (2000×1500px, ~40–60m altitude); CLAHE illumination
   normalization, 16px minimum-dimension filter, zero-confidence detections
   removed.
2. **Track-aware split** — all crops sharing a `track_id` go exclusively to
   train (80%) or test (20%), guaranteeing zero identity overlap
   (6,632 train / 2,210 test samples).
3. **Classical pipelines** — HOG descriptors (9 orientation bins, 8×8 cells,
   1,764-D) → random over-sampling to 21,434 balanced samples → LinearSVC
   (C=0.05) or Random Forest (200 trees, depth 20).
4. **Deep pipeline** — ImageNet-pretrained ResNet18, two-phase fine-tuning
   (frozen-backbone head warmup, then layer4 + head at lr=4e-5), Focal Loss
   (γ=2.0) with inverse-frequency class weights and weighted random sampling.
5. **Interpretability** — t-SNE on HOG vs. penultimate-layer features;
   Grad-CAM on correctly classified and misclassified crops.

## Suggested repository layout

```
├── data/            # dataset extraction & preprocessing scripts
├── features/        # HOG extraction, oversampling, scaling
├── models/          # SVM, Random Forest, ResNet18 training
├── evaluation/      # metrics, confusion matrices, leakage experiments
├── visualization/   # t-SNE, Grad-CAM, dataset analysis figures
├── notebooks/       # Colab walkthrough
└── README.md
```

## Getting started

Experiments were run on Google Colab Pro (T4 GPU) with PyTorch 2.x and
scikit-learn 1.x. Adapt the paths in the notebooks to your environment and run
top to bottom: data extraction → features → training → evaluation.

## Limitations & future work

- Single-sequence evaluation: bus (n=6) and van (n=1) test samples are
  statistically insufficient for reliable F1 — reported for reproducibility
  only.
- Future work: multi-sequence training, higher input resolution (128–224px),
  and end-to-end detectors (e.g., YOLOv8) exploiting full-frame spatial
  context instead of pre-cropped regions.

## Citation

If you use this work, please cite the accompanying report:

> SHEHRYAR (P2952028). *Multi-Class Object Classification on UAV Aerial
> Imagery: A Comparative Study of Classical and Deep Learning Approaches.*
> De Montfort University, CSIP5403 — Research Methods and Applied AI, 2026.

The VisDrone benchmark: Zhu et al., "Detection and Tracking Meet Drones
Challenge," IEEE TPAMI, 2022.

## License

MIT — see [LICENSE](LICENSE) for details.
