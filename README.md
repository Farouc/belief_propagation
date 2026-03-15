# Belief Propagation for Discrete Graphical Models

This project implements a clean, modular **sum-product Belief Propagation (BP)** pipeline for discrete pairwise graphical models.

Current focus:
- core BP implementation for pairwise MRFs
- image segmentation experiments (BSDS500 and GrabCut-style data)
- Sudoku solving with BP constraints

## What Is Implemented

### Core modules (`src/`)
- `graph.py`: `PairwiseMRF` graph container (unary and pairwise potentials)
- `belief_propagation.py`: sum-product BP / loopy BP message passing
- `potentials.py`: helper potential functions

### Dataset loaders (`datasets/`)
- `bsds500_loader.py`: loads BSDS500 images + annotations, optional resizing, binary mask conversion
- `grabcut_loader.py`: loads GrabCut-style image/mask pairs from local folders
- `sudoku_loader.py`: loads Sudoku puzzles from HuggingFace (with deterministic fallback)

### Experiments (`experiments/`)
- `segmentation_bsds500.py`
- `segmentation_grabcut.py`
- `sudoku_bp_solver.py`

### Utilities (`utils/`)
- `segmentation_utils.py`: build segmentation MRFs + decode beliefs to masks
- `sudoku_utils.py`: build Sudoku MRF constraints + validity checks
- `metrics.py`: IoU and pixel accuracy
- `visualization.py`: plot helpers

### Notebooks (`notebooks/`)
- `segmentation_bsds500.ipynb`
- `segmentation_grabcut.ipynb`
- `sudoku_experiments.ipynb`

### Results (`results/`)
- experiment outputs (metrics CSV, summary JSON, qualitative plots)
- interpretation guide: `results/README.md`

## Project Structure

```text
belief_propagation/
├── src/
│   ├── graph.py
│   ├── potentials.py
│   └── belief_propagation.py
├── datasets/
│   ├── bsds500_loader.py
│   ├── grabcut_loader.py
│   └── sudoku_loader.py
├── experiments/
│   ├── synthetic_graph_test.py
│   ├── image_segmentation_demo.py
│   ├── segmentation_bsds500.py
│   ├── segmentation_grabcut.py
│   └── sudoku_bp_solver.py
├── utils/
│   ├── visualization.py
│   ├── segmentation_utils.py
│   ├── sudoku_utils.py
│   └── metrics.py
├── notebooks/
│   ├── segmentation_bsds500.ipynb
│   ├── segmentation_grabcut.ipynb
│   └── sudoku_experiments.ipynb
├── results/
└── data/            # local datasets (ignored by git)
```

## Environment Setup

If you use the local shortcut:

```bash
base
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

`data/` is ignored in git (datasets are too heavy). Use the commands below.

### 1) BSDS500

```bash
mkdir -p data/BSDS500
cd data
curl -L --fail https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -o BSR_bsds500.tgz
rm -rf BSDS500/*
tar -xzf BSR_bsds500.tgz -C BSDS500 --strip-components=1

# Ensure loader-compatible structure:
# data/BSDS500/data/images
# data/BSDS500/data/groundTruth
if [ -d BSDS500/BSDS500/data ] && [ ! -d BSDS500/data ]; then
  mv BSDS500/BSDS500/data BSDS500/data
fi
```

### 2) GrabCut (local image/mask pair format)

Expected structure:

```text
data/grabcut/
├── images/
└── masks/
```

Example (OpenCV extra sample):

```bash
mkdir -p data/grabcut/raw data/grabcut/images data/grabcut/masks
curl -L --fail https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/grabcut/image1652.ppm -o data/grabcut/raw/image1652.ppm
curl -L --fail https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/grabcut/mask1652.ppm -o data/grabcut/raw/mask1652.ppm
python - <<'PY'
from pathlib import Path
from skimage import io
import numpy as np
root = Path('data/grabcut')
img = io.imread(root / 'raw' / 'image1652.ppm')
msk = io.imread(root / 'raw' / 'mask1652.ppm')
if msk.ndim == 3:
    msk = msk[..., 0]
io.imsave(root / 'images' / 'image1652.png', img, check_contrast=False)
io.imsave(root / 'masks' / 'image1652_mask.png', ((msk > 127).astype(np.uint8) * 255), check_contrast=False)
PY
```

### 3) Sudoku

No local files required. The loader tries public HuggingFace Sudoku datasets first and falls back to built-in puzzles if unavailable.

Note: in Sudoku grids, `0` means an empty cell to fill.

## Run Experiments

### Synthetic BP check
```bash
python experiments/synthetic_graph_test.py
```

### Simple segmentation demo
```bash
python experiments/image_segmentation_demo.py
```

### BSDS500 segmentation
```bash
python experiments/segmentation_bsds500.py --data-path data/BSDS500 --max-images 20 --resize 80 80
```

### GrabCut segmentation
```bash
python experiments/segmentation_grabcut.py --data-path data/grabcut --max-images 20 --resize 80 80
```

### Sudoku BP solver
```bash
python experiments/sudoku_bp_solver.py --n-samples 100 --max-iters 150
```

## Result Files

See `results/README.md` for detailed explanation of:
- IoU / pixel accuracy / solve rate
- convergence plots
- per-sample CSV and summary JSON outputs

## Reproducibility Notes

- deterministic ordering is used in graph neighbors/edge iteration
- experiment scripts expose seed arguments
- BP on loopy graphs is approximate; convergence depends on hyperparameters and problem difficulty
