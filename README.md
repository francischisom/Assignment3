# Assignment3
# Hybrid Biomedical Image Analysis — An Auditable Data Science Pipeline

An end-to-end pipeline for a fluorescence-microscopy **nuclei segmentation** dataset that
deliberately keeps language generation *downstream* of measurable image evidence. It links
image preprocessing, a vision–language model (VLM) description, classical Otsu segmentation
with quantitative region features, a PyTorch **U-Net**, and a "numbers-first" LLM
interpretation stage — with structured JSON kept as the source of truth at every step.

**Author:** Ofonagoro Chisom Francis · MSc Applied Data Science, University of Hertfordshire
**Assignment:** 3 — Hybrid Biomedical Image Analysis

---

## Overview

The core idea is that predictive accuracy and *trustworthiness* are separate properties. The
pipeline is built so that every narrative claim can be traced back to a measured number, and
so that failures are visible rather than hidden behind fluent prose.

| Stage | What it does |
|---|---|
| **Task 1 — VLM description** | Preprocessing + EDA, then a direct VLM description of each image, with naive vs. prompt-engineered comparison. |
| **Task 2 — Classical + numbers-first** | Otsu thresholding + morphology + `regionprops_table`; a text-only LLM interprets *only* the measured features. |
| **Task 3 — U-Net** | A compact PyTorch U-Net trained with BCE+Dice; evaluated against the Otsu baseline. |
| **Task 4 — Hybrid pipeline** | U-Net → region features → text LLM → validated JSON + short narrative, aggregated to CSV. |
| **Extensions** | Loss ablation (BCE / Dice / BCE+Dice) and a robustness experiment (blur / low-contrast corruption). |

### Headline results

| Method | Validation Dice | Validation IoU |
|---|---|---|
| Otsu + morphology (baseline) | 0.9781 | 0.9572 |
| **Main U-Net (BCE+Dice, 15 ep.)** | **0.9958** | **0.9916** |
| U-Net — held-out test (12 images) | 0.9960 | 0.9921 |

The U-Net beat Otsu on every one of the 20 validation images. A blur-corruption test shows the
first detectable failure is at **segmentation** (Dice collapses to 0.7357), before feature
extraction — downstream language cannot repair an upstream segmentation error.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── hybrid_biomedical_image_analysis.ipynb   # the main Colab notebook (end-to-end)
│
├── src/                        # optional: logic refactored out of the notebook
│   ├── preprocessing.py        #   grayscale, resize (256×256), EDA
│   ├── segmentation.py         #   Otsu + morphology + regionprops features
│   ├── unet.py                 #   U-Net model, training loop, Dice/IoU metrics
│   ├── llm.py                  #   VLM + numbers-first + hybrid prompts (Ollama)
│   └── robustness.py           #   blur / contrast corruption experiment
│
├── data/
│   ├── README.md               # how to obtain the dataset (do NOT commit the images)
│   ├── images/                 # raw microscopy images   (git-ignored)
│   └── masks/                  # ground-truth masks      (git-ignored)
│
├── models/
│   └── unet_best.pt            # best-validation checkpoint (git-ignored or via Releases/LFS)
│
├── results/
│   ├── figures/                # fig1–fig6 (EDA, curves, val panels, robustness)
│   ├── metrics/                # per-image Dice/IoU CSVs, ablation results
│   └── hybrid_records.csv      # Task 4 structured output
│
└── report/
    └── Assignment3_Report_24101661.docx   # (or export a PDF)
```

If you keep everything in a single Colab notebook, you can drop the `src/` folder — just make
sure the notebook is cleanly ordered by task and the `data/`, `results/`, and `report/`
folders are still present.

---

## Dataset

- **Content:** paired fluorescence-microscopy images and binary nuclei masks.
- **Audit:** 116 candidate raw images and 112 masks → **112 valid image–mask pairs** (4 raw images unmatched and dropped).
- **Preprocessing:** converted to grayscale, resized to **256 × 256**.
- **Split:** **80 / 20 / 12** train / validation / test, fixed **seed 42**.

The image data is **not committed** to the repo (size / licensing). Place the files under
`data/images/` and `data/masks/` and describe the source in `data/README.md`.

---

## Requirements

- Python 3.10+
- PyTorch + TorchVision
- scikit-image, NumPy, pandas, matplotlib, Pillow
- [Ollama](https://ollama.com) running locally, with the **LLaVA** model pulled

Example `requirements.txt`:

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

Set up the local LLM (used for the VLM and numbers-first / hybrid text steps):

```bash
# install Ollama first (see ollama.com), then:
ollama pull llava
```

> **Note on the VLM.** The assignment brief specifies `llama3.2-vision`, but that model wasn't
> reliable in the runtime, so **LLaVA** was used for the image-description step instead. It fills
> the same role in the pipeline; the VLM outputs reflect LLaVA, not `llama3.2-vision`.

---

## How to run

**Colab (simplest):** open `notebooks/hybrid_biomedical_image_analysis.ipynb` in Google Colab,
mount the dataset, and run the cells in order. The LLM cells expect a reachable Ollama server —
run those steps locally or point them at a running Ollama instance.

**Local:**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
ollama pull llava           # start Ollama in the background
jupyter notebook notebooks/hybrid_biomedical_image_analysis.ipynb
```

The notebook runs Task 1 → Task 4 top to bottom and writes figures to `results/figures/`,
metrics to `results/metrics/`, and the Task 4 records to `results/hybrid_records.csv`.

---

## Reproducibility

- Fixed seed (**42**) and a frozen 80/20/12 split.
- Best-validation-epoch selection rather than last-epoch.
- **Temperature 0** for the numbers-first and hybrid LLM calls (deterministic); temperature 0.2
  for the VLM repeatability check.
- The test set is reserved for the final pipeline, not for model selection.

## Limitations

Educational prototype only — **not** for clinical use. Evidence is limited by a small dataset
(112 pairs, 20 validation, 12 test), no external cohort, semantic rather than instance-level
segmentation, and an added layer of LLM uncertainty. The single most valuable next step would be
independent external validation on a larger, clinically representative, expert-annotated dataset.

---

## References

1. Ronneberger, O., Fischer, P. & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
2. Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms.* IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.
3. Liu, H., Li, C., Wu, Q. & Lee, Y.J. (2023). *Visual Instruction Tuning.* arXiv:2304.08485.

---

## License

Choose one that suits your needs (e.g. MIT for the code). Add a `LICENSE` file at the repo root.

> **Before making this public:** check your module's policy on publishing coursework. Some
> programmes require repos to stay **private** until after grading. If in doubt, keep it private
> and share access with markers directly.
