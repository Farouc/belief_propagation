# Belief Propagation for Discrete Graphical Models

This repository implements a clean, modular **sum-product Belief Propagation (BP)** pipeline for discrete pairwise graphical models, with experiments on:
- binary image segmentation (BSDS500 + GrabCut-style data)
- Sudoku solving

## Implemented Components

### Core (`src/`)
- `graph.py`: `PairwiseMRF` with unary/pairwise potentials
- `belief_propagation.py`: sum-product BP (supports loopy BP + damping)
- `potentials.py`: helper potential constructors

### Datasets (`datasets/`)
- `bsds500_loader.py`: BSDS500 loader + binary conversion from multi-region annotations
- `grabcut_loader.py`: GrabCut-style image/mask loader with robust matching
- `sudoku_loader.py`: HuggingFace Sudoku loader with deterministic fallback

### Experiments (`experiments/`)
- `synthetic_graph_test.py`
- `image_segmentation_demo.py`
- `segmentation_bsds500.py`
- `segmentation_grabcut.py`
- `sudoku_bp_solver.py`

### Utilities (`utils/`)
- `segmentation_utils.py`: Gaussian unary modeling + contrast-sensitive pairwise MRF build
- `sudoku_utils.py`: Sudoku graph constraints and validation
- `metrics.py`: IoU, pixel accuracy, dataset summaries
- `visualization.py`: plotting helpers

### Notebooks (`notebooks/`)
- `segmentation_bsds500.ipynb`
- `segmentation_grabcut.ipynb`
- `sudoku_experiments.ipynb`

## Project Structure

```text
belief_propagation/
├── src/
├── datasets/
├── experiments/
├── utils/
├── notebooks/
├── results/
├── data/            # local datasets (ignored in git)
├── requirements.txt
└── README.md
```

## Environment Setup

Activate your environment (your local shortcut):

```bash
base
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

`data/` is ignored by git.

### 1) BSDS500

```bash
mkdir -p data/BSDS500
cd data
curl -L --fail https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -o BSR_bsds500.tgz
rm -rf BSDS500/*
tar -xzf BSR_bsds500.tgz -C BSDS500 --strip-components=1
```

Expected layout used by loader:
- `data/BSDS500/data/images`
- `data/BSDS500/data/groundTruth`

### 2) GrabCut-style dataset (recommended source used here)

A working multi-image source is the repository you found:

```bash
git clone --depth 1 https://github.com/irllabs/grabcut.git /tmp/irllabs_grabcut
mkdir -p data/grabcut/images/irllabs data/grabcut/masks/irllabs
python3 - <<'PY'
from pathlib import Path
import shutil
src_img = Path('/tmp/irllabs_grabcut/dataset/img/ASAP')
src_msk = Path('/tmp/irllabs_grabcut/dataset/mask/ASAP')
dst_img = Path('data/grabcut/images/irllabs')
dst_msk = Path('data/grabcut/masks/irllabs')
for img in sorted(src_img.glob('*.jpg')):
    m = src_msk / f'{img.stem}.jpg'
    if m.exists():
        shutil.copy2(img, dst_img / img.name)
        shutil.copy2(m, dst_msk / f'{img.stem}_mask.jpg')
print('done')
PY
```

### 3) Sudoku

No local files required.
The loader tries HuggingFace first, then falls back to built-in puzzles if needed.

Note: Sudoku `0` means an empty cell to fill.

## Run Experiments

### Synthetic sanity check

```bash
python experiments/synthetic_graph_test.py
```

### Segmentation demo

```bash
python experiments/image_segmentation_demo.py
```

### BSDS500 segmentation

```bash
python experiments/segmentation_bsds500.py --data-path data/BSDS500 --max-images 20 --resize 60 60 --max-iters 50
```

### GrabCut segmentation

```bash
python experiments/segmentation_grabcut.py --data-path data/grabcut --max-images 20 --resize 60 60 --max-iters 20
```

### Sudoku BP solver

```bash
python experiments/sudoku_bp_solver.py --n-samples 100 --max-iters 150
```

## Notebooks

Run:
- `notebooks/segmentation_bsds500.ipynb`
- `notebooks/segmentation_grabcut.ipynb`
- `notebooks/sudoku_experiments.ipynb`

They auto-detect common dataset paths, run experiments, and visualize summaries.

## Results

Main outputs are written under `results/`:
- `metrics.csv`
- `summary.json`
- aggregate plots (IoU/accuracy histograms, convergence curves, qualitative panels)

See `results/README.md` for interpretation details.

To keep commits clean, this repo now keeps **lightweight aggregate outputs** and omits heavy per-sample dumps when not needed.

## Reproducibility Notes

- Deterministic neighbor and edge ordering are used.
- Scripts expose seed/config arguments.
- Loopy BP is approximate; convergence depends on graph hardness and hyperparameters.
