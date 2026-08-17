# Hybrid Biomedical Image Analysis

An end-to-end, auditable pipeline for **fluorescence-microscopy nuclei segmentation** that keeps
language generation *downstream* of measurable image evidence. It links preprocessing, a
vision–language model (VLM) description, classical Otsu segmentation with quantitative region
features, a PyTorch **U-Net**, and a "numbers-first" LLM interpretation stage — with structured
JSON kept as the source of truth at every step.

**Author:** Ofonagoro Chisom Francis 
---

## Overview

The guiding idea is that predictive accuracy and *trustworthiness* are different properties. The
pipeline is built so every narrative claim can be traced back to a measured number, and so
failures are visible rather than hidden behind fluent prose.

| Stage | What it does |
|---|---|
| **Task 1 — VLM description** | Preprocessing + EDA, then a direct VLM description, comparing a naive vs. a prompt-engineered request. |
| **Task 2 — Classical + numbers-first** | Otsu + morphology + `regionprops_table`; a text-only LLM interprets *only* the measured features. |
| **Task 3 — U-Net** | A compact PyTorch U-Net (BCE+Dice), evaluated against the Otsu baseline. |
| **Task 4 — Hybrid pipeline** | U-Net → region features → text LLM → validated JSON + short narrative, aggregated to CSV/JSON. |
| **Extensions** | Loss ablation (BCE / Dice / BCE+Dice) and a robustness experiment (blur / low contrast). |

### Headline results

| Method | Validation Dice | Validation IoU |
|---|---|---|
| Otsu + morphology (baseline) | 0.9781 | 0.9572 |
| **Main U-Net (BCE+Dice, 15 ep.)** | **0.9958** | **0.9916** |
| U-Net — held-out test (12 images) | 0.9960 | 0.9921 |

The U-Net beat Otsu on all 20 validation images. A blur-corruption test shows the first
detectable failure is at **segmentation** (Dice collapses to 0.7357), before feature extraction —
downstream language cannot repair an upstream segmentation error.

---

## Repository structure

```
Assignment3/                                                    # repository root
├── README.md
└── Assignment_3/
    ├── Assignment3_Hybrid_Biomedical_Image_Analysis.ipynb      # main notebook (Tasks 1–4)
    │
    ├── figures/                                                # figures
    │
    ├── assignment3_output/                                     # raw run outputs
    │   ├── hybrid_test_records.csv
    │   ├── hybrid_test_records.json
    │   ├── hybrid_raw_llm_outputs.json
    │   └── validation_otsu_vs_unet.csv
    │
    └── assignment3_submission/                                 # submission artifacts
        ├── experiment_config.json                             # run config (seed, split, hyperparams)
        ├── unet_bce_dice_best.pt                              # best U-Net checkpoint
        ├── hybrid_test_records.json
        ├── hybrid_raw_llm_outputs.json
        ├── validation_otsu_vs_unet.csv
        ├── robustness_results.csv
        └── loss_comparison.csv
```

### What the output files contain

| File | Contents |
|---|---|
| `hybrid_test_records.csv` / `.json` | Task 4 structured records per test image: `image_id, n_objects, mean_area, density_class, quality_flag` + narrative. |
| `hybrid_raw_llm_outputs.json` | Raw, unparsed LLM responses, retained for auditability. |
| `validation_otsu_vs_unet.csv` | Per-image Dice/IoU for Otsu vs. U-Net across the 20 validation images. |
| `robustness_results.csv` | Metrics for the blur / low-contrast corruption experiment. |
| `loss_comparison.csv` | Loss ablation results (BCE / Dice / BCE+Dice). |
| `experiment_config.json` | Seed, data split and training hyperparameters for reproducibility. |
| `unet_bce_dice_best.pt` | Trained weights for the best-validation U-Net. |

> `assignment3_output/` holds the notebook's raw outputs; `assignment3_submission/` is the curated
> set handed in (it adds the config, the model checkpoint, and the ablation/robustness CSVs). Some
> files appear in both — the submission copies are the canonical ones.

---

## Dataset

Paired fluorescence-microscopy images and binary nuclei masks. The directory audit found 116
candidate raw images and 112 masks → **112 valid pairs** (4 unmatched raw images dropped). Images
are grayscale, resized to **256 × 256**, split **80 / 20 / 12** train/validation/test with a fixed
**seed of 42**.

The image data itself is **not committed** to this repo (size / licensing). To reproduce, place
the images and masks where the notebook's data paths expect them.

---

## Requirements

- Python 3.10+
- PyTorch + TorchVision
- scikit-image, NumPy, pandas, matplotlib, Pillow
- [Ollama](https://ollama.com) running locally, with the **LLaVA** model pulled (`ollama pull llava`)

> **Note on the VLM.** The brief specifies `llama3.2-vision`, but that model wasn't reliable in the
> runtime, so **LLaVA** was used for the image-description step instead. It fills the same role in
> the pipeline; the VLM outputs reflect LLaVA, not `llama3.2-vision`.

---

## How to run

Open `Assignment_3/Assignment3_Hybrid_Biomedical_Image_Analysis.ipynb` in Google Colab (or
Jupyter), make the dataset available, and run the cells top to bottom. Tasks 1 → 4 execute in
order and write results to `assignment3_output/`. The LLM cells expect a reachable Ollama server,
so run those steps locally or point them at a running Ollama instance.

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
2. Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms.* IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.
3. Liu, H., Li, C., Wu, Q. & Lee, Y.J. (2023). *Visual Instruction Tuning.* arXiv:2304.08485.

---

> **Optional additions:** a `requirements.txt`, a `.gitignore` (exclude the dataset and large
> checkpoints), and a `LICENSE`. Also check your module's policy on public coursework — some
> programmes require repos to stay **private** until after grading.
