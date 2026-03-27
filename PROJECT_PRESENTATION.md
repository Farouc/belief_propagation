# Belief Propagation for Discrete Graphical Models

Technical project report for image segmentation and Sudoku with sum-product BP and TRW-BP.

## 1. Executive Summary

This project implements a clean, modular inference pipeline for discrete pairwise Markov Random Fields (MRFs), centered on:

- Sum-product Belief Propagation (BP)
- Tree-Reweighted Belief Propagation (TRW-BP)

The codebase supports:

- synthetic graph sanity checks,
- binary image segmentation (BSDS500 and GrabCut-style data),
- Sudoku solving as a high-constraint combinatorial benchmark.

Main outcomes from the current saved runs:

- **Segmentation (GT-unary/oracle track):**
  - BSDS500: BP IoU = 0.787, TRW IoU = 0.778.
  - GrabCut: BP IoU = 0.774, TRW IoU = 0.772.
- **Segmentation (no-GT unary, EM track):**
  - BSDS500 (label-aligned IoU): BP = 0.596, TRW = 0.596.
  - GrabCut (label-aligned IoU): BP = 0.480, TRW = 0.480.
- **Sudoku (100 puzzles):** BP solve rate = 0.97, TRW solve rate = 0.71.
- **Sudoku difficulty analysis by clue count:** 33 unique clue counts (28..76), but classic hard/medium/easy bins are imbalanced (`2/28/70`), so quantile buckets are more reliable for comparisons.

## 2. Project Scope and Goals

The project objective is to build and evaluate a modular probabilistic inference framework where the same message-passing engine can be reused across distinct domains.

Goals achieved:

1. Implemented reusable `PairwiseMRF` representation and factor storage.
2. Implemented stable loopy BP with damping and convergence tracking.
3. Added TRW-BP variant as a drop-in alternative.
4. Built robust dataset loaders for BSDS500, GrabCut, and Sudoku.
5. Designed domain-specific MRF models for segmentation and Sudoku.
6. Automated experiments with metrics, plots, summaries, and CSV outputs.
7. Added notebooks for reproducible exploration and qualitative analysis.

## 3. Repository Organization

```text
belief_propagation/
  src/
    graph.py
    potentials.py
    belief_propagation.py
    trw_belief_propagation.py
  datasets/
    bsds500_loader.py
    grabcut_loader.py
    sudoku_loader.py
  experiments/
    synthetic_graph_test.py
    image_segmentation_demo.py
    segmentation_bsds500.py
    segmentation_bsds500_trw.py
    segmentation_grabcut.py
    segmentation_grabcut_trw.py
    segmentation_em_comparison.py
    sudoku_bp_solver.py
    sudoku_bp_solver_trw.py
    sudoku_difficulty_analysis.py
  utils/
    segmentation_utils.py
    segmentation_em_utils.py
    sudoku_utils.py
    trw_utils.py
    metrics.py
    visualization.py
  notebooks/
    segmentation_bsds500.ipynb
    segmentation_bsds500_trw.ipynb
    segmentation_grabcut.ipynb
    segmentation_grabcut_trw.ipynb
    sudoku_experiments.ipynb
    sudoku_experiments_trw.ipynb
  results/
    segmentation/
    sudoku/
    sudoku/difficulty_analysis/
```

Design principle: keep inference core (`src`) independent from task-specific data/modeling (`datasets`, `utils`, `experiments`).

## 4. Theoretical Background

### 4.1 Pairwise MRF Model

Let a graph \(G=(V,E)\), variables \(x_i \in \{1,\dots,K_i\}\). A pairwise MRF defines:

\[
P(x) \propto \prod_{i \in V} \psi_i(x_i) \prod_{(i,j) \in E} \psi_{ij}(x_i, x_j)
\]

where:

- \(\psi_i\): unary potential,
- \(\psi_{ij}\): pairwise compatibility.

### 4.2 Sum-Product Belief Propagation

Message from node \(i\) to \(j\):

\[
m_{i \to j}(x_j) \propto \sum_{x_i} \psi_i(x_i)\,\psi_{ij}(x_i,x_j)
\prod_{k \in N(i)\setminus j} m_{k \to i}(x_i)
\]

Belief (approximate marginal) at node \(i\):

\[
b_i(x_i) \propto \psi_i(x_i) \prod_{k \in N(i)} m_{k \to i}(x_i)
\]

MAP label for discrete prediction:

\[
\hat{x}_i = \arg\max_{x_i} b_i(x_i)
\]

### 4.3 Loopy BP and Damping

For non-tree graphs, BP is approximate. The implementation uses synchronous updates and damping:

\[
m^{(t+1)} = (1-\alpha)m^{(t)} + \alpha \tilde{m}^{(t+1)}, \quad \alpha=0.5
\]

Convergence criterion:

\[
\Delta_t = \max_{(i,j)} \| m^{(t+1)}_{i\to j} - m^{(t)}_{i\to j} \|_\infty
\]

stop if \(\Delta_t < \text{tol}\), with default `tol=1e-3`.

### 4.4 Tree-Reweighted BP (TRW-BP)

TRW introduces edge appearance probabilities \(\rho_{ij}\). Implemented update:

\[
m_{i\to j}(x_j) \propto
\sum_{x_i}
\psi_i(x_i)
\psi_{ij}(x_i,x_j)^{\rho_{ij}}
\prod_{k\in N(i)\setminus j}
 m_{k\to i}(x_i)^{\rho_{ki}}
\]

Beliefs:

\[
b_i(x_i) \propto \psi_i(x_i)
\prod_{k\in N(i)} m_{k\to i}(x_i)^{\rho_{ki}}
\]

Current experiments use uniform weights:

- \(\rho_{ij}=0.5\) for every directed edge.

## 5. Core Implementation Details

### 5.1 Graph Abstraction (`PairwiseMRF`)

- Stores per-node state cardinality (supports heterogeneous \(K_i\)).
- Unary storage: `unary_potentials[i]` shape `(K_i,)`.
- Pairwise storage: `pairwise_potentials[(i,j)]` shape `(K_i, K_j)` and transpose cached for reverse direction.
- Deterministic traversal via sorted neighbors and sorted edges.

### 5.2 Numerical Stability Tricks

Used throughout BP/TRW and segmentation utilities:

1. Message normalization with fallback to uniform if sums are invalid.
2. Damping to suppress oscillations in loopy graphs.
3. Covariance regularization for Gaussian unary estimation.
4. Fallback covariance solve path if matrix inversion is unstable.
5. Potential clipping (`np.clip`) in TRW before fractional powers.
6. Log-likelihood stabilization before exponentiation.

### 5.3 Complexity Notes

- BP update complexity per iteration is roughly \(O(|E|K^2)\) for pairwise MRFs.
- Segmentation at 60x60:
  - nodes = 3600,
  - undirected edges = 7080 (4-neighborhood),
  - directed messages = 14160, each over 2 states.
- Sudoku:
  - nodes = 81,
  - undirected edges = 810,
  - degree = 20 for every node,
  - directed messages = 1620, each over 9 states.

## 6. Dataset Pipelines

### 6.1 BSDS500 Loader

Input challenge: BSDS500 annotations are multi-region, not binary.

Implemented conversion pipeline:

1. Load image and segmentation annotation (`.mat` or image mask).
2. Extract segmentation map.
3. For each region label, compute connected components.
4. Select largest connected component globally as background.
5. Set foreground = complement.
6. Resize image and binary mask to 60x60 for tractable BP.

This creates consistent binary masks for the segmentation experiments.

### 6.2 GrabCut Loader

Supports multiple directory layouts and robust image-mask matching.

Mask conversion for OpenCV GrabCut labels:

- 0: background
- 1: foreground
- 2: probable background
- 3: probable foreground

Binary conversion used:

- foreground = `(mask == 1) OR (mask == 3)`
- background = `(mask == 0) OR (mask == 2)`

Additional handling:

- supports standard 0/255 masks,
- skips degenerate masks (single class) if requested,
- deterministic file ordering,
- loads up to 20 samples by default.

### 6.3 Sudoku Loader

- Tries HuggingFace datasets first (multiple candidate datasets).
- Runs loading in subprocess to avoid import conflicts with local `datasets/` package name.
- Parses varied puzzle formats to 9x9 int grid.
- Uses deterministic built-in fallback puzzles when offline.

Important convention:

- `0` means empty cell (to be solved by inference).

## 7. Task-Specific Modeling

## 7.1 Binary Image Segmentation MRF

### Variables and Graph

- one binary variable per pixel: `0=background`, `1=foreground`.
- 4-neighborhood grid graph.

### Unary Potentials (Gaussian Color Models)

Given pixel feature vector \(I_p\):

\[
\psi_p(\text{fg}) \propto P(I_p\mid \text{fg}) P(\text{fg}), \quad
\psi_p(\text{bg}) \propto P(I_p\mid \text{bg}) P(\text{bg})
\]

with class-conditional Gaussian likelihoods:

\[
\log P(I\mid c) = -\frac{1}{2}\left[d\log(2\pi) + \log|\Sigma_c| + (I-\mu_c)^T\Sigma_c^{-1}(I-\mu_c)\right]
\]

Potentials are normalized per pixel to sum to 1.

Practical note:

- In the current segmentation experiments, unary Gaussians are estimated using ground-truth masks by default (`use_gt_unary=True`). This is a supervised/oracle setting for stronger class modeling.

### EM unary potentials (no-GT track)

To avoid label leakage in inference, we added a second segmentation track where unary terms are estimated with an unsupervised 2-component Gaussian mixture (EM) over pixel colors.

EM updates:

\[
\gamma_{nk} =
\frac{\pi_k \mathcal{N}(x_n \mid \mu_k,\Sigma_k)}
{\sum_{j} \pi_j \mathcal{N}(x_n \mid \mu_j,\Sigma_j)}
\]

\[
N_k = \sum_n \gamma_{nk},\quad
\pi_k = \frac{N_k}{N},\quad
\mu_k = \frac{1}{N_k}\sum_n \gamma_{nk}x_n
\]

\[
\Sigma_k =
\frac{1}{N_k}\sum_n \gamma_{nk}(x_n-\mu_k)(x_n-\mu_k)^T + \epsilon I
\]

Unary potentials are set from responsibilities (`bg`, `fg`) after mapping the brighter component to foreground.

### Pairwise Potentials (Contrast-Sensitive Smoothness)

For neighboring pixels \(p,q\):

\[
\text{diff}_{pq} = \|I_p - I_q\|^2,
\quad
w_{pq} = \exp(-\beta\,\text{diff}_{pq})
\]

\[
\psi_{pq} =
\begin{bmatrix}
1 & \exp(-\lambda w_{pq}) \\
\exp(-\lambda w_{pq}) & 1
\end{bmatrix}
\]

with

\[
\beta = \frac{1}{2\,\mathbb{E}[\|I_p-I_q\|^2]}
\]

Interpretation:

- similar neighbors (small diff, larger \(w\)) get stronger same-label preference,
- high-contrast neighbors get weaker smoothing (boundary preservation).

## 7.2 Sudoku as Pairwise Constraint MRF

- 81 nodes (cells), 9 states per node (digits 1..9).
- Edge between cells sharing row/column/3x3 block.

Unary:

- clue cell: one-hot potential on fixed digit,
- empty cell: uniform potential.

Pairwise (inequality constraint):

\[
\psi_{ij}(x_i,x_j)=
\begin{cases}
0 & x_i=x_j \\
1 & x_i\neq x_j
\end{cases}
\]

Final prediction takes per-cell MAP and re-enforces clue cells to original values.

## 8. Experimental Protocol

### 8.1 Reproducibility Controls

- deterministic sorted graph traversal,
- explicit seed argument in scripts,
- deterministic dataset ordering,
- saved JSON config with each run.

### 8.2 Main Hyperparameters

Segmentation (BP):

- max iterations: 50 (GrabCut BP run used 20 in saved summary config),
- tolerance: 1e-3,
- damping: 0.5,
- pairwise smoothness `lambda=2.0`.

Segmentation (TRW):

- same, with uniform `rho=0.5`.

Segmentation EM (no-GT unary):

- EM max iterations: 60,
- EM tolerance: 1e-4,
- EM covariance regularization: 1e-4.

Sudoku:

- max iterations: 150,
- tolerance: 1e-3,
- damping: 0.5,
- TRW uses uniform `rho=0.5`.

### 8.3 Metrics

Segmentation:

\[
\text{IoU} = \frac{|\hat{Y}\cap Y|}{|\hat{Y}\cup Y|}
\]

\[
\text{Pixel Accuracy} = \frac{1}{N}\sum_{p=1}^N \mathbf{1}[\hat{y}_p = y_p]
\]

For unsupervised binary segmentation (EM track), label identity is permutation-ambiguous. We therefore report both:

\[
\text{IoU}_{\text{raw}} = \text{IoU}(\hat{Y},Y),\quad
\text{IoU}_{\text{eval}} = \max\{\text{IoU}(\hat{Y},Y),\ \text{IoU}(1-\hat{Y},Y)\}
\]

and similarly for accuracy (`raw` vs `eval`).

Sudoku:

- solve rate (valid completed grids),
- clue consistency rate,
- convergence rate,
- iterations and runtime statistics.

## 9. Quantitative Results (from saved artifacts)

## 9.1 Segmentation: BP vs TRW

| Dataset | Method | Images | Mean IoU +/- std | Mean Acc +/- std | Conv. Rate | Mean Iters | Mean Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BSDS500 | BP | 20 | 0.7868 +/- 0.1273 | 0.8635 +/- 0.1024 | 0.45 | 37.45 | 8.14 |
| BSDS500 | TRW | 20 | 0.7775 +/- 0.1216 | 0.8584 +/- 0.0981 | 1.00 | 13.85 | 5.96 |
| GrabCut | BP | 20 | 0.7740 +/- 0.2007 | 0.9739 +/- 0.0172 | 0.20 | 18.60 | 8.96 |
| GrabCut | TRW | 20 | 0.7718 +/- 0.1969 | 0.9733 +/- 0.0172 | 1.00 | 12.10 | 10.99 |

Observations:

- TRW strongly improves convergence on segmentation loops (both datasets reach 1.0 convergence rate).
- TRW also reduces average iterations on both segmentation datasets (BSDS500: 37.45 -> 13.85, GrabCut: 18.60 -> 12.10).
- IoU/accuracy differences between BP and TRW are small and slightly favor BP in this configuration.
- GrabCut has one severe outlier (IoU=0 for one sample in both BP and TRW) while most samples are strong.

## 9.2 Sudoku: BP vs TRW

| Method | Puzzles | Solve Rate +/- std | Clue Consistency | Conv. Rate | Mean Iters +/- std | Mean Runtime (s) +/- std |
|---|---:|---:|---:|---:|---:|---:|
| BP | 100 | 0.97 +/- 0.1706 | 1.00 | 0.98 | 30.13 +/- 18.68 | 2.610 +/- 1.527 |
| TRW | 100 | 0.71 +/- 0.4538 | 1.00 | 0.99 | 65.58 +/- 26.52 | 23.899 +/- 9.918 |

Observations:

- BP substantially outperforms TRW for Sudoku in this setup.
- TRW converges slightly more often but to worse fixed points for solving accuracy.
- TRW is much slower due to significantly higher iteration count and per-iteration overhead.

## 9.3 Distribution-Level Findings

From per-sample CSVs:

- BSDS500 BP IoU range: 0.513 to 0.967 (median ~0.792).
- BSDS500 TRW IoU range: 0.517 to 0.967 (median ~0.784).
- GrabCut BP IoU range: 0.000 to 0.938 (median ~0.829).
- GrabCut TRW IoU range: 0.000 to 0.943 (median ~0.820).
- Sudoku BP solved 97/100; TRW solved 71/100.

## 9.4 Segmentation without GT unary (EM track)

Table below reports permutation-aware `eval` metrics (the meaningful unsupervised comparison).

| Dataset | Method | Images | Mean IoU eval +/- std | Mean Acc eval +/- std | Flip Rate | Conv. Rate | Mean Iters | Mean Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BSDS500 | BP | 20 | 0.5956 +/- 0.1675 | 0.7118 +/- 0.1780 | 0.50 | 0.75 | 34.90 | 13.79 |
| BSDS500 | TRW | 20 | 0.5961 +/- 0.1657 | 0.7124 +/- 0.1759 | 0.50 | 1.00 | 12.95 | 10.39 |
| GrabCut | BP | 20 | 0.4803 +/- 0.2667 | 0.8130 +/- 0.1736 | 0.90 | 0.85 | 26.80 | 10.19 |
| GrabCut | TRW | 20 | 0.4799 +/- 0.2666 | 0.8131 +/- 0.1730 | 0.90 | 1.00 | 11.65 | 8.30 |

Observations:

- Removing GT from unary estimation significantly lowers segmentation quality versus oracle-unary runs, as expected.
- BP and TRW yield very similar EM-track quality; TRW mainly improves convergence and iteration counts.
- High flip rates (especially GrabCut) confirm that unsupervised binary label identity is arbitrary and must be handled in evaluation.

## 9.5 Sudoku difficulty-by-clue analysis

Difficulty metadata is not provided directly, so we used clue count as a proxy and extracted clue counts from saved puzzle examples to stay aligned with experiment metrics.

Dataset diversity summary:

- 100 puzzles, clue range: 28..76.
- 33 unique clue counts.
- Classic bins (`hard <=29`, `medium 30..35`, `easy >=36`) are imbalanced: `2 / 28 / 70`.
- Quantile bins are balanced: `38 / 30 / 32`.

Quantile-bucket results (better comparison protocol):

| Bucket | BP solve rate | TRW solve rate | BP mean runtime (s) | TRW mean runtime (s) |
|---|---:|---:|---:|---:|
| hard_q (<=36 clues) | 0.947 | 0.447 | 3.37 | 28.70 |
| medium_q (37..43 clues) | 1.000 | 0.767 | 2.49 | 25.79 |
| easy_q (>=44 clues) | 0.969 | 0.969 | 1.82 | 16.43 |

Interpretation:

- BP is robust across clue-count buckets in this dataset.
- TRW improves with more clues, but remains slower and less reliable in lower-clue regimes.
- For this repository's Sudoku sample, quantile-based difficulty analysis is preferable to classic hard/medium/easy bins.

## 10. How to Read the Plots

### Segmentation plots (`results/segmentation/.../plots`)

1. `best_median_worst.png`
- three representative cases sorted by IoU.
- compare where BP preserves object boundaries vs where it over-smooths.

2. `iou_hist.png`
- distribution shape indicates robustness.
- narrow high-IoU histogram = stable model.
- long low-IoU tail = difficult failure modes.

3. `accuracy_hist.png`
- useful but can be inflated by background dominance.
- always interpret with IoU, not alone.

4. `convergence_curves.png` / `*_trw_convergence_curves.png`
- y-axis is max message change (log scale), x-axis iteration.
- steep drop indicates stable inference.
- flattening above tolerance suggests non-convergent oscillatory regime.

### Sudoku plots (`results/sudoku/plots`)

1. `solve_rate.png` and `sudoku_trw_solve_rate.png`
- solved vs unsolved counts.

2. `iterations_hist.png` / `sudoku_trw_iterations_hist.png`
- lower and tighter is generally better computationally.

3. `convergence.png` / `sudoku_trw_convergence.png`
- per-puzzle iteration bars, color-coded by convergence.
- shows whether hard puzzles dominate runtime tail.

4. `runtime_hist.png`
- practical compute cost profile across puzzle set.

### Sudoku difficulty plots (`results/sudoku/difficulty_analysis/plots`)

1. `solve_rate_vs_clue_count.png`
- clue-count trend curves (BP vs TRW) with variability bands.

2. `runtime_vs_clue_count.png` and `iterations_vs_clue_count.png`
- computational scaling with clue density.

3. `trw_bp_gap_by_clue_count.png`
- direct method gap view (solve-rate difference and runtime ratio).

4. `quantile_pareto_runtime_vs_solve.png`
- runtime/quality Pareto view across quantile difficulty buckets.

## 11. Interpretation and Research Insights

### 11.1 Why segmentation improved vs early poor runs

Main gains came from modeling and preprocessing, not from changing BP core:

1. BSDS500 binary conversion fixed (largest connected component as background).
2. Correct GrabCut mask semantics (0/1/2/3 mapping).
3. Better unary model (Gaussian class-conditionals instead of naive intensity thresholds).
4. Contrast-sensitive pairwise smoothing.
5. Damping and stable normalization in message passing.

### 11.2 Why convergence and quality are not always aligned

- TRW often converges more reliably in loopy grids.
- But convergence to a fixed point does not guarantee better MAP labels.
- With fixed uniform \(\rho\), TRW can over/underweight interactions depending on task structure.

### 11.3 Why Sudoku BP > TRW here

Likely reasons:

1. Sudoku constraints are hard and highly structured (many zero entries in pairwise potentials).
2. Uniform `rho=0.5` may dilute critical exclusion constraints in TRW updates.
3. TRW converges but can settle on marginals less compatible with exact combinatorial consistency.

### 11.4 Important evaluation caveat

For segmentation experiments, default config uses ground-truth masks to estimate unary Gaussian parameters (`use_gt_unary=True`). This is a strong supervised hint and should be reported explicitly in any academic presentation.

Recommended phrasing:

- "Current results evaluate BP/TRW inference quality under supervised unary estimation, not fully unsupervised segmentation."

## 12. Limitations

1. Segmentation unary estimation uses GT labels by default (label leakage risk for fair benchmarking).
2. TRW edge weights are uniform (not optimized/tree-derived).
3. BP and TRW are run in probability domain; log-domain variants may be more stable for harder settings.
4. No post-processing (e.g., connected component cleanup, morphology, CRF refinement) is applied.
5. Sudoku decoding is simple argmax per cell; no global search/backtracking layer.

## 13. Practical Recommendations for Presentation

When presenting to professor/colleagues:

1. Highlight that the main technical contribution is a reusable inference framework across two very different domains.
2. Explain the separation of concerns: data preprocessing vs probabilistic modeling vs inference engine.
3. Emphasize that preprocessing/modeling quality dominates final segmentation performance.
4. Show BP vs TRW as a convergence-quality tradeoff, not a strict "one is always better" result.
5. Explicitly disclose supervised unary estimation in segmentation to keep scientific rigor.
6. Present both segmentation tracks (oracle GT-unary and unsupervised EM-unary) to separate modeling signal from inference behavior.

## 14. Suggested Next Steps

1. Unsupervised unary estimation:
- estimate FG/BG GMMs without GT masks (EM or weak priors), then compare against current supervised setting.

2. Better TRW weighting:
- use non-uniform \(\rho\) from spanning-tree sampling or degree-based heuristics.

3. Inference schedule improvements:
- residual/asynchronous BP updates,
- early stopping with patience,
- adaptive damping.

4. Sudoku hybrid solver:
- BP marginals for candidate pruning + lightweight backtracking for exact solve boost.

5. Fairness protocol:
- run ablations with `--no-gt-unary` and report both supervised-unary and unsupervised-unary tracks.

6. Sudoku difficulty protocol:
- prefer quantile buckets when classic clue bins are strongly imbalanced.

## 15. Reproducibility Commands

Activate environment:

```bash
base
```

Run BP experiments:

```bash
python experiments/segmentation_bsds500.py --data-path data/BSDS500 --max-images 20 --resize 60 60 --max-iters 50
python experiments/segmentation_grabcut.py --data-path data/grabcut --max-images 20 --resize 60 60 --max-iters 50
python experiments/sudoku_bp_solver.py --n-samples 100 --max-iters 150
```

Run TRW experiments:

```bash
python experiments/segmentation_bsds500_trw.py --data-path data/BSDS500 --max-images 20 --resize 60 60 --max-iters 50
python experiments/segmentation_grabcut_trw.py --data-path data/grabcut --max-images 20 --resize 60 60 --max-iters 50
python experiments/sudoku_bp_solver_trw.py --n-samples 100 --max-iters 150
```

Run no-GT EM segmentation experiments:

```bash
python experiments/segmentation_em_comparison.py --dataset bsds500 --inference bp --data-path data/BSDS500 --max-images 20 --resize 60 60
python experiments/segmentation_em_comparison.py --dataset bsds500 --inference trw --data-path data/BSDS500 --max-images 20 --resize 60 60
python experiments/segmentation_em_comparison.py --dataset grabcut --inference bp --data-path data/grabcut --max-images 20 --resize 60 60
python experiments/segmentation_em_comparison.py --dataset grabcut --inference trw --data-path data/grabcut --max-images 20 --resize 60 60
```

Run Sudoku difficulty analysis and research figures:

```bash
python experiments/sudoku_difficulty_analysis.py \
  --bp-metrics results/sudoku/metrics.csv \
  --trw-metrics results/sudoku/sudoku_trw_metrics.csv \
  --examples-root results/sudoku/examples \
  --n-samples 100 \
  --results-dir results/sudoku/difficulty_analysis
```

## 16. Final Takeaway

The project successfully demonstrates that a single modular message-passing framework can support both low-level vision segmentation and structured combinatorial reasoning. The strongest lesson is that **inference quality is tightly coupled to data preprocessing and potential design**: better probabilistic modeling and cleaner labels delivered the largest gains, while TRW mainly improved convergence behavior under loops.
