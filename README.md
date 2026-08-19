# Hybrid Biomedical Image Analysis — An Auditable Data Science Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-red)
![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5--VL%20%7C%20Llama3.2-black)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, auditable pipeline for **fluorescence-microscopy nuclei segmentation**. It keeps
language generation *downstream* of measurable image evidence: preprocessing and EDA, a
**Qwen2.5-VL** visual description, classical **Otsu** segmentation with quantitative region
features, a **PyTorch U-Net**, and a numbers-first LLM interpretation stage — with structured JSON
kept as the source of truth at every step.

**Author:** Ofonagoro Chisom Francis · MSc Applied Data Science, University of Hertfordshire

---

## Overview

The central idea is that predictive accuracy and *trustworthiness* are separate properties. Every
narrative claim traces back to a measured number, and failures stay visible instead of being hidden
behind fluent prose.

| Stage | What it does |
|---|---|
| **Task 1 — VLM description** | Preprocessing + EDA, then a Qwen2.5-VL description with a naive vs. prompt-engineered comparison. |
| **Task 2 — Classical + numbers-first** | Otsu + morphology + `regionprops_table`; a text-only LLM interprets *only* the measured features. |
| **Task 3 — U-Net** | A compact PyTorch U-Net (BCE+Dice), evaluated against the Otsu baseline. |
| **Task 4 — Hybrid pipeline** | U-Net -> region features -> text LLM -> validated JSON + narrative, aggregated to CSV/JSON. |
| **Extensions** | Loss ablation (BCE / Dice / BCE+Dice) and a robustness experiment (blur / low contrast). |

### Headline results

| Method | Validation Dice | Validation IoU |
|---|---|---|
| Otsu + morphology (baseline) | 0.9781 | 0.9572 |
| **Main U-Net (BCE+Dice, 15 ep.)** | **0.9940** | **0.9881** |
| U-Net — held-out test (12 images) | 0.9936 | 0.9874 |

The U-Net beat Otsu on every one of the 20 validation images. In the 10-epoch loss ablation, **BCE**
gave the best short-run Dice (0.9932). A blur-corruption test shows the first detectable failure is
at **segmentation** (Dice collapses to 0.7357), before feature extraction — downstream language
cannot repair an upstream segmentation error.

---

## Models

| Role | Model | Runtime |
|---|---|---|
| Task 1 vision-language description | `qwen2.5vl:7b` (Qwen2.5-VL 7B) | Ollama |
| Tasks 2 & 4 text-only interpretation | `llama3.2:3b` | Ollama |

> The assessment brief specified `llama3.2-vision`; the module announcement permitted an alternative
> vision model when compatibility problems occur, so **Qwen2.5-VL** is used for Task 1. This is an
> authorised substitution — the VLM outputs reflect Qwen2.5-VL, not `llama3.2-vision`.

---

## Repository structure

```
hybrid-biomedical-image-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── Biomedical_Image_Analysis.ipynb          # end-to-end notebook (Tasks 1-4)
│
├── figures/                                     # figures exported by the notebooks
│
├── outputs/                                     
│   ├── experiment_config.json                   
│   ├── validation_otsu_vs_unet.csv              # per-image Otsu vs U-Net Dice/IoU
│   ├── loss_comparison.csv                      # BCE / Dice / BCE+Dice ablation
│   ├── robustness_results.csv                   # blur / low-contrast corruption metrics
│   ├── hybrid_test_records.csv                  # Task 4 structured records
│   ├── hybrid_test_records.json
│   ├── hybrid_raw_llm_outputs.json              # raw LLM responses (audit trail)
│   └── unet_bce_dice_best.pt                    # trained U-Net weights (~7.7 MB)
│
└── data/
    └── README.md                               
```

---

## Dataset

Paired fluorescence-microscopy images and binary nuclei masks: **112 valid image-mask pairs**,
converted to grayscale and resized to **256 x 256**, split **80 / 20 / 12** train/validation/test
with a fixed **seed of 42**. The image data is **not committed** (size / licensing) — place it where
the notebook's data paths expect it and describe the source in `data/README.md`.

---

## Setup

**Python packages** (`requirements.txt`):

```
torch
torchvision
scikit-image
numpy
pandas
matplotlib
pillow
ollama
```

**Local models** (via [Ollama](https://ollama.com)):

```bash
# install Ollama (see ollama.com), start the server, then:
ollama pull qwen2.5vl:7b
ollama pull llama3.2:3b
```

---

## How to run

Open `notebooks/Biomedical_Image_Analysis.ipynb` in Google Colab (GPU runtime recommended) and run
all cells top to bottom. The notebook installs and starts Ollama, pulls the two models, runs Tasks
1 -> 4, writes every artifact to `assignment3_outputs/`, and bundles `assignment3_submission.zip`.
Locally, install the requirements, start Ollama with both models pulled, then run the notebook.

---

## Reproducibility

- Fixed seed (**42**) and a frozen 80/20/12 split (see `experiment_config.json`).
- Best-validation-epoch selection rather than last-epoch.
- **Temperature 0** for the numbers-first and hybrid LLM calls; temperature 0.2 for the VLM
  repeatability check.
- The test set is reserved for the final pipeline, not for model selection.

## Limitations

Educational prototype only — **not** for clinical use. Evidence is limited by a small dataset
(112 pairs, 20 validation, 12 test), no external cohort, semantic rather than instance-level
segmentation, and an added layer of LLM uncertainty. The most valuable next step would be
independent external validation on a larger, clinically representative, expert-annotated dataset.

---

## References

1. Ronneberger, O., Fischer, P. & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
2. Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms.* IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.
3. Bai, S. et al. (2025). *Qwen2.5-VL Technical Report.* arXiv:2502.13923.
4. Assignment 3 assessment brief, Applied Data Science assessment specification.

---
