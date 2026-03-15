# Results Guide

This file explains how to read the plots and metrics saved by the experiment scripts.

## Segmentation (`results/segmentation/...`)

### `predictions/sample_XXX_pred.png`
- Binary predicted mask from BP.
- Pixel value `0` = background, `1` = foreground.

### `plots/sample_XXX_comparison.png`
- Three-panel qualitative view:
1. Original image.
2. BP predicted mask.
3. Ground-truth mask.
- Use this to visually inspect where BP over/under segments.

### `plots/convergence.png` (BSDS500)
- Bar chart where:
- x-axis: sample index.
- y-axis: number of BP iterations run.
- color: green = converged, red = not converged by `max_iters`.
- If many bars are red, increase `max_iters` or relax `tol`.

### `plots/runtime_convergence.png` (GrabCut)
- Bar chart where:
- x-axis: sample index.
- y-axis: runtime (seconds).
- color: green = converged, red = not converged.
- Helps identify hard images that take longer.

### `metrics.csv`
Per-sample table:
- `iou`: Intersection-over-Union (higher is better, max 1).
- `accuracy`: pixel-wise accuracy (higher is better, max 1).
- `runtime_sec`: wall-clock runtime for that sample.
- `iterations`: iterations used by BP.
- `converged`: 1 if BP met tolerance, else 0.

### `summary.json`
Dataset-level averages:
- `mean_iou`, `mean_accuracy`, `mean_runtime_sec`, `mean_iterations`, `convergence_rate`.

## Sudoku (`results/sudoku/...`)

### `plots/convergence.png`
- Bar chart where:
- x-axis: puzzle index.
- y-axis: BP iterations.
- color: green = converged, red = did not converge by `max_iters`.

### `examples/puzzle_XXX.txt`
Text report per puzzle:
- Whether it was solved.
- Whether BP converged.
- Runtime.
- Original puzzle and BP prediction grids.

### `metrics.csv`
Per-puzzle table:
- `solved`: 1 if predicted grid is a valid Sudoku and respects clues.
- `clues_consistent`: 1 if given clues were preserved.
- `converged`: 1 if BP met tolerance.
- `iterations`, `runtime_sec`.

### `summary.json`
Aggregate Sudoku metrics:
- `solve_rate`: fraction solved.
- `clue_consistency_rate`: fraction preserving clues.
- `convergence_rate`: fraction converged.
- `mean_iterations`, `mean_runtime_sec`.

## Note on Your Current Results
- The folders `bsds500_smoke` and `grabcut_smoke` are smoke-test runs on synthetic toy data.
- High IoU/accuracy there do **not** reflect real benchmark performance.
- For real evaluation, run on real BSDS500/GrabCut datasets and inspect the same files.
