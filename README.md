# Conformal Triage Routing for Breast Cancer Detection

CM3070 Machine Learning and Neural Networks final project, University of London. Author: Alexander Brakas

> **Research prototype. Not a medical device and not clinically validated.**

## Overview

A mammography triage prototype that decides, for each scan, whether the model may act on it or must defer to a radiologist.

- A ResNet-18 + DINOv2 ViT-S/14 hybrid (plus a CNN baseline and a DINOv2-only ablation) produces class probabilities.
- Probabilities are calibrated with temperature scaling and wrapped in **class-conditional (Mondrian) conformal prediction**, with asymmetric error tolerances: 1% for malignant, 5% for benign.
- A scan is **automated** only if its prediction set contains exactly one label. Otherwise it is **deferred** to a radiologist queue with a Grad-CAM heatmap and **no predicted class**, so the model's guess cannot anchor the reader.

## Quick check (no retraining needed)

The three telemetry CSVs from the reported run are included, so the evaluation figures and statistics can be reproduced directly:

```
python generate_charts.py
python generate_roc.py
python bootstrap_auc.py
```

These write the routing, accuracy, false-negative and ROC charts to `evaluation_charts/` and print the ROC-AUC values with 95% bootstrap confidence intervals.

## Repository contents

| File | Purpose |
|---|---|
| `datasort.py` | Builds the `ImageFolder` dataset (`benign/`, `malignant/`) from the CBIS-DDSM CSVs and JPEGs |
| `model.py` | Architectures: `BaselineCNN`, `HybridDualTopology`, `ViTOnlyBranch` |
| `train_pipeline.py` | Trains all three models, fits temperature scaling, runs conformal calibration, saves artifacts |
| `triage_daemon.py` | Routes the held-out scans (automated pass or deferral), writes heatmaps and a telemetry CSV |
| `generate_charts.py` | Routing, accuracy and false-negative charts from the telemetry CSVs |
| `generate_roc.py` | ROC curves and AUC from the logged probabilities |
| `bootstrap_auc.py` | 95% bootstrap confidence intervals for ROC-AUC and paired AUC differences |
| `benchmark_hardware.py` | Parameter count and inference latency per architecture |
| `*_triage_telemetry.csv` | Per-scan results of the reported run, one file per model |
| `requirements.txt` | Pinned package versions |

## Setup

Developed on Windows with an NVIDIA RTX 3060 (CUDA 12.1). Requires Python 3.11 or newer. CPU also works, but training is slow.

1. Install PyTorch for your system from https://pytorch.org/get-started/locally/ (developed with torch 2.5.1 and torchvision 0.20.1).
2. Install the remaining packages:

```
pip install -r requirements.txt
```

The first training or triage run downloads pretrained weights (ResNet-18 via torchvision, DINOv2 ViT-S/14 via timm), so an internet connection is needed.

## Data

The dataset is not included. Download CBIS-DDSM from the Kaggle mirror (https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) and place it in the repository root as:

```
data/
    csv/     the CBIS-DDSM metadata CSVs
    jpeg/    one folder per series (named by its UID), each holding its JPEG images
```

`datasort.py` reads `mass_case_description_train_set.csv` and `calc_case_description_train_set.csv`, labels each image from the `pathology` column (`MALIGNANT` becomes malignant, everything else benign) and copies the images into `ImageFolder` layout:

```
clean_training_set/
    benign/
    malignant/
```

The test-set CSVs are not used: only training-partition images are sorted, and all train / evaluation / calibration splits are drawn from them.

## Running

Run everything from the repository root; all paths are relative to it.

1. **Build the dataset:** `python datasort.py` creates `clean_training_set/` from `data/`.
2. **Train and calibrate:** `python train_pipeline.py` writes `deploy_artifacts/{model}_weights.pth`, `_calibration_scores.npz` and `_meta.json` for `baseline_cnn`, `hybrid_topology` and `vit_only`.
3. **Triage the held-out scans:** set `model_name` at the top of the `__main__` block in `triage_daemon.py`, then run `python triage_daemon.py`. Repeat once per model. This writes `{model}_triage_telemetry.csv` plus heatmaps in `human_review_queue/` (deferrals) and `automated_pass_audit/` (automated passes).
4. **Evaluation:** `python generate_charts.py`, `python generate_roc.py` and `python bootstrap_auc.py`.
5. **Hardware benchmark:** `python benchmark_hardware.py`.

The two heatmap folders are shared by all models and file names do not include the model, so running the daemon for a second model overwrites heatmaps with the same scan ID. The ViT-only model writes none. To get a clean set for one model, empty both folders and run that model last.

## Model weights

Trained weights are not included (the files are 45 to 132 MB). `train_pipeline.py` recreates all artifacts in `deploy_artifacts/`, after which `triage_daemon.py` can be run. The reported results themselves are preserved in the three telemetry CSVs (see Quick check).

## Results

Single run on the 247 held-out scans (109 malignant, 138 benign).

| | Baseline CNN | ViT-only (DINOv2) | Hybrid |
|---|---|---|---|
| Deferred | 82.6% | 73.7% | 83.0% |
| Automated accuracy | 81.4% | 80.0% | 81.0% |
| Malignant scans automated as benign | 0 / 109 | 1 / 109 | 0 / 109 |
| Malignant coverage (target 99%) | 100.0% | 99.1% | 100.0% |
| Benign coverage (target 95%) | 94.2% | 91.3% | 94.2% |
| ROC-AUC [95% CI] | 70.9% [64.0, 76.9] | 77.9% [71.8, 83.1] | 71.3% [64.7, 77.3] |
| Latency, RTX 3060 (ms per tensor) | 1.32 | 4.13 | 5.37 |
| Parameters | 11,171,266 | 21,907,202 | 33,081,286 |

- The hybrid did not outperform its own CNN branch (AUC difference +0.3, 95% CI -4.6 to +5.0). The DINOv2-only model discriminated best (+6.9 over the baseline, 95% CI +0.1 to +13.5), which is suggestive rather than decisive.
- Only 5 to 10% of scans were cleared as benign, so the safety floor bounds how much can be automated.
- AUC is computed from the logged `P_Malignant`; intervals come from 2,000 paired bootstrap resamples (seed 0). Latency is the mean of 100 passes after 20 warm-up passes, with GPU synchronisation before and after timing.

## Reproducibility

- Data splits are seeded: 80/10/10 train / evaluation / calibration with seed 42, and the early-stopping holdout with seed 7. The global seed is 22.
- Re-running `triage_daemon.py` on the saved hybrid weights reproduced the reported routing and probabilities.
- Retraining may differ slightly because of GPU non-determinism.

## Known limitations

- The split is at image level, so images from one patient may appear in both training and evaluation sets.
- Results come from a single run and one split, and the held-out set also informed development; 109 malignant scans cannot verify a 1% miss rate.
- Input resolution is fixed at 224 x 224, images are JPEG conversions of the original DICOM files, and only the last block of each backbone is trained.
- Heatmaps are coarse (7 x 7, upsampled) and have not been validated against lesion annotations.

## Data and model sources

- CBIS-DDSM: Lee et al., 2017, *Scientific Data* 4:170177 (Kaggle mirror by awsaf49).
- ResNet-18 ImageNet weights via torchvision; DINOv2 (Oquab et al., 2023) via timm.

## Licence

The code in this repository is released under the MIT License (see `LICENSE`).

The CBIS-DDSM data (Kaggle mirror) is licensed CC BY-SA 3.0. The three telemetry CSVs contain labels
derived from it and are therefore shared under CC BY-SA 3.0, not MIT. The dataset itself and the
pretrained ResNet-18 (torchvision) and DINOv2 (timm) weights are not included and remain under
their own terms.
