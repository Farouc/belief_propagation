"""Generate presentation figures for segmentation and Sudoku slides.

All outputs are saved to one folder for direct Overleaf import.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bsds500_loader import load_bsds500_dataset
from datasets.grabcut_loader import load_grabcut_dataset
from src.belief_propagation import BeliefPropagation
from src.trw_belief_propagation import TreeReweightedBeliefPropagation
from utils.segmentation_utils import beliefs_to_mask, build_binary_segmentation_mrf
from utils.trw_utils import compute_uniform_edge_weights

BP_COLOR = "#009688"
TRW_COLOR = "#C62828"
BG_COLOR = "#ECF0F5"
GRID_COLOR = "#B0BEC5"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "savefig.facecolor": BG_COLOR,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": GRID_COLOR,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.size": 11,
        }
    )


def _read_csv(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed: List[Dict[str, float]] = []
    for row in rows:
        parsed.append({k: float(v) for k, v in row.items()})
    return parsed


def _load_binary_image(path: Path) -> np.ndarray:
    arr = np.asarray(io.imread(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(np.uint8)


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    resize: Tuple[int, int] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if resize is None:
        return image, mask

    if image.ndim == 2:
        image_resized = transform.resize(
            image,
            resize,
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )
    else:
        image_resized = transform.resize(
            image,
            (*resize, image.shape[-1]),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )

    mask_resized = transform.resize(
        mask.astype(np.float64),
        resize,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return image_resized.astype(np.float64), (mask_resized > 0.5).astype(np.uint8)


def _load_oracle_segmentation_data(
    dataset: str,
    data_path: Path,
    resize: Tuple[int, int],
    max_images: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if dataset == "bsds500":
        images, masks = load_bsds500_dataset(data_path, max_images=max_images, resize=resize)
        masks = [(np.asarray(m) > 0).astype(np.uint8) for m in masks]
        return images, masks

    if dataset == "grabcut":
        images_raw, masks_raw = load_grabcut_dataset(data_path, max_images=max_images)
        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for img, m in zip(images_raw, masks_raw):
            img_r, m_r = _resize_pair(img, (np.asarray(m) > 0).astype(np.uint8), resize=resize)
            images.append(img_r)
            masks.append(m_r)
        return images, masks

    raise ValueError(f"Unsupported dataset: {dataset}")


def _selected_worst_median_best(rows: Sequence[Dict[str, float]], score_key: str = "iou") -> List[int]:
    order = np.argsort(np.array([r[score_key] for r in rows], dtype=np.float64))
    return [int(rows[int(order[0])]["index"]), int(rows[int(order[len(order) // 2])]["index"]), int(rows[int(order[-1])]["index"])]


def _save_qualitative_grid(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    predictions: Dict[int, np.ndarray],
    selected_indices: Sequence[int],
    metric_rows: Sequence[Dict[str, float]],
    output_path: Path,
    pred_title: str,
) -> None:
    metric_by_index = {int(r["index"]): r for r in metric_rows}
    labels = ["Worst IoU", "Median IoU", "Best IoU"]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12.0))

    for row_id, idx in enumerate(selected_indices):
        img = images[idx]
        gt = masks[idx]
        pred = predictions[idx]
        row = metric_by_index[idx]

        ax0 = axes[row_id, 0]
        if img.ndim == 2:
            ax0.imshow(img, cmap="gray")
        else:
            ax0.imshow(np.clip(img, 0.0, 1.0))
        ax0.set_title(f"{labels[row_id]} | Original")
        ax0.axis("off")
        ax0.set_aspect("equal")

        ax1 = axes[row_id, 1]
        ax1.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax1.set_title("Ground Truth")
        ax1.axis("off")
        ax1.set_aspect("equal")

        ax2 = axes[row_id, 2]
        ax2.imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax2.set_title(f"{pred_title}\nIoU={row['iou']:.3f}, Acc={row['accuracy']:.3f}")
        ax2.axis("off")
        ax2.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_inference_curves_and_selected_preds(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    lambda_: float,
    max_iters: int,
    tol: float,
    damping: float,
    method: str,
) -> Tuple[List[List[float]], Dict[int, np.ndarray]]:
    curves: List[List[float]] = []
    selected_set = set(int(i) for i in selected_indices)
    selected_preds: Dict[int, np.ndarray] = {}

    for idx, (image, gt_mask) in enumerate(zip(images, masks)):
        mrf = build_binary_segmentation_mrf(image=image, mask_for_unary=gt_mask, lambda_=lambda_)

        if method == "bp":
            infer = BeliefPropagation(mrf, max_iters=max_iters, tol=tol, damping=damping)
        elif method == "trw":
            rho = compute_uniform_edge_weights(mrf)
            infer = TreeReweightedBeliefPropagation(
                mrf,
                rho=rho,
                max_iters=max_iters,
                tol=tol,
                damping=damping,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        infer.run()
        curves.append(list(infer.message_deltas))

        if idx in selected_set:
            beliefs = infer.compute_beliefs()
            selected_preds[idx] = beliefs_to_mask(beliefs, gt_mask.shape)

    return curves, selected_preds


def _save_convergence_plot(
    bp_curves: Sequence[Sequence[float]],
    trw_curves: Sequence[Sequence[float]],
    tol: float,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.0))

    for c in bp_curves:
        x = np.arange(1, len(c) + 1)
        ax.plot(x, c, color=BP_COLOR, alpha=0.23, linewidth=1.2)
    for c in trw_curves:
        x = np.arange(1, len(c) + 1)
        ax.plot(x, c, color=TRW_COLOR, alpha=0.23, linewidth=1.2)

    # Mean curves for readability.
    def _mean_curve(curves: Sequence[Sequence[float]]) -> np.ndarray:
        max_len = max(len(c) for c in curves)
        m = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for i, c in enumerate(curves):
            m[i, : len(c)] = np.asarray(c, dtype=np.float64)
        return np.nanmean(m, axis=0)

    bp_mean = _mean_curve(bp_curves)
    trw_mean = _mean_curve(trw_curves)
    ax.plot(np.arange(1, bp_mean.size + 1), bp_mean, color=BP_COLOR, linewidth=3.0, label="BP mean")
    ax.plot(np.arange(1, trw_mean.size + 1), trw_mean, color=TRW_COLOR, linewidth=3.0, label="TRW mean")

    ax.axhline(tol, color="black", linestyle="--", linewidth=1.5, label=f"tol={tol:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max message change")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_em_vs_oracle_bar(
    summary_oracle_bsds_bp: Dict[str, float],
    summary_oracle_bsds_trw: Dict[str, float],
    summary_oracle_grab_bp: Dict[str, float],
    summary_oracle_grab_trw: Dict[str, float],
    summary_em_bsds_bp: Dict[str, float],
    summary_em_bsds_trw: Dict[str, float],
    summary_em_grab_bp: Dict[str, float],
    summary_em_grab_trw: Dict[str, float],
    output_path: Path,
) -> None:
    datasets = ["BSDS500", "GrabCut"]
    x = np.arange(len(datasets))
    w = 0.18

    bp_oracle = [summary_oracle_bsds_bp["mean_iou"], summary_oracle_grab_bp["mean_iou"]]
    trw_oracle = [summary_oracle_bsds_trw["mean_iou"], summary_oracle_grab_trw["mean_iou"]]
    bp_em = [summary_em_bsds_bp["mean_iou_eval"], summary_em_grab_bp["mean_iou_eval"]]
    trw_em = [summary_em_bsds_trw["mean_iou_eval"], summary_em_grab_trw["mean_iou_eval"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(x - 1.5 * w, bp_oracle, w, color=BP_COLOR, label="BP-oracle")
    ax.bar(x - 0.5 * w, trw_oracle, w, color=TRW_COLOR, label="TRW-oracle")
    ax.bar(x + 0.5 * w, bp_em, w, color=BP_COLOR, alpha=0.45, hatch="//", label="BP-EM")
    ax.bar(x + 1.5 * w, trw_em, w, color=TRW_COLOR, alpha=0.45, hatch="//", label="TRW-EM")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Segmentation: oracle unary vs unsupervised EM unary")
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_em_best_median_worst_bsds(
    images_bsds: Sequence[np.ndarray],
    masks_bsds: Sequence[np.ndarray],
    em_rows: Sequence[Dict[str, float]],
    pred_eval_dir: Path,
    output_path: Path,
) -> None:
    order = np.argsort(np.array([r["iou_eval"] for r in em_rows], dtype=np.float64))
    selected = [int(em_rows[int(order[0])]["index"]), int(em_rows[int(order[len(order) // 2])]["index"]), int(em_rows[int(order[-1])]["index"])]

    if not any(em_rows[i]["label_flipped_for_eval"] > 0.5 for i in [int(order[0]), int(order[len(order) // 2]), int(order[-1])]):
        flipped = [r for r in em_rows if r["label_flipped_for_eval"] > 0.5]
        if flipped:
            # Force one flipped example in the middle row.
            selected[1] = int(max(flipped, key=lambda r: r["iou_eval"])["index"])

    labels = ["Worst IoU", "Middle IoU", "Best IoU"]
    row_by_idx = {int(r["index"]): r for r in em_rows}
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12.0))

    for ridx, idx in enumerate(selected):
        img = images_bsds[idx]
        gt = masks_bsds[idx]
        pred = _load_binary_image(pred_eval_dir / f"sample_{idx:03d}_pred_eval.png")
        row = row_by_idx[idx]
        flip_tag = "flipped" if row["label_flipped_for_eval"] > 0.5 else "not flipped"

        ax0 = axes[ridx, 0]
        if img.ndim == 2:
            ax0.imshow(img, cmap="gray")
        else:
            ax0.imshow(np.clip(img, 0.0, 1.0))
        ax0.set_title(f"{labels[ridx]} | Original")
        ax0.axis("off")
        ax0.set_aspect("equal")

        ax1 = axes[ridx, 1]
        ax1.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax1.set_title("Ground Truth")
        ax1.axis("off")
        ax1.set_aspect("equal")

        ax2 = axes[ridx, 2]
        ax2.imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax2.set_title(
            f"EM Prediction\nIoU raw/eval={row['iou_raw']:.3f}/{row['iou_eval']:.3f}\n{flip_tag}"
        )
        ax2.axis("off")
        ax2.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_sudoku_overall_figures(
    summary_bp: Dict[str, float],
    summary_trw: Dict[str, float],
    bp_rows: Sequence[Dict[str, float]],
    trw_rows: Sequence[Dict[str, float]],
    output_dir: Path,
) -> None:
    # Solve rate bar with std.
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    means = [summary_bp["solve_rate"], summary_trw["solve_rate"]]
    stds = [summary_bp["solve_rate_std"], summary_trw["solve_rate_std"]]
    ax.bar(["BP", "TRW"], means, yerr=stds, capsize=6, color=[BP_COLOR, TRW_COLOR], alpha=0.9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Solve rate")
    ax.set_title("Sudoku solve rate (BP vs TRW)")
    plt.tight_layout()
    fig.savefig(output_dir / "sudoku_solve_rate_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Iterations histogram.
    bp_iter = np.array([r["iterations"] for r in bp_rows], dtype=np.float64)
    trw_iter = np.array([r["iterations"] for r in trw_rows], dtype=np.float64)
    bins = np.arange(0, max(int(np.max(bp_iter)), int(np.max(trw_iter))) + 6, 5)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.hist(bp_iter, bins=bins, alpha=0.55, color=BP_COLOR, label="BP", edgecolor="black")
    ax.hist(trw_iter, bins=bins, alpha=0.55, color=TRW_COLOR, label="TRW", edgecolor="black")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Puzzle count")
    ax.set_title("Sudoku iterations distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "sudoku_iterations_hist.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Runtime histogram.
    bp_rt = np.array([r["runtime_sec"] for r in bp_rows], dtype=np.float64)
    trw_rt = np.array([r["runtime_sec"] for r in trw_rows], dtype=np.float64)
    bins_rt = np.linspace(0.0, max(np.max(bp_rt), np.max(trw_rt)), 18)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.hist(bp_rt, bins=bins_rt, alpha=0.55, color=BP_COLOR, label="BP", edgecolor="black")
    ax.hist(trw_rt, bins=bins_rt, alpha=0.55, color=TRW_COLOR, label="TRW", edgecolor="black")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Puzzle count")
    ax.set_title("Sudoku runtime distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "sudoku_runtime_hist.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_clues_from_examples(examples_root: Path, n_samples: int) -> np.ndarray:
    index_to_clues: Dict[int, int] = {}
    for path in sorted(examples_root.rglob("puzzle_*.txt")):
        name = path.stem  # puzzle_000
        idx = int(name.split("_")[1])
        if idx >= n_samples:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if "Puzzle:" not in lines:
            continue
        start = lines.index("Puzzle:") + 1
        grid_lines = lines[start : start + 9]
        values: List[int] = []
        for ln in grid_lines:
            toks = ln.strip().split()
            if len(toks) != 9:
                values = []
                break
            values.extend(int(t) for t in toks)
        if len(values) != 81:
            continue
        index_to_clues[idx] = int(np.sum(np.array(values) > 0))

    return np.array([index_to_clues[i] for i in range(n_samples)], dtype=np.int32)


def _save_sudoku_difficulty_figures(
    bp_rows: Sequence[Dict[str, float]],
    trw_rows: Sequence[Dict[str, float]],
    clues: np.ndarray,
    quantile_table: Sequence[Dict[str, float]],
    output_dir: Path,
) -> None:
    # solve_rate_vs_clue_count.png
    unique_clues = np.sort(np.unique(clues))
    bp_solved = np.array([r["solved"] for r in bp_rows], dtype=np.float64)
    trw_solved = np.array([r["solved"] for r in trw_rows], dtype=np.float64)

    def _stats(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for c in unique_clues:
            idx = np.where(clues == c)[0]
            x = vals[idx]
            means.append(float(np.mean(x)))
            stds.append(float(np.std(x)))
        return np.array(means), np.array(stds)

    bp_m, bp_s = _stats(bp_solved)
    trw_m, trw_s = _stats(trw_solved)

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(unique_clues, bp_m, color=BP_COLOR, linewidth=2.7, marker="o", label="BP")
    ax.fill_between(unique_clues, bp_m - bp_s, bp_m + bp_s, color=BP_COLOR, alpha=0.16)
    ax.plot(unique_clues, trw_m, color=TRW_COLOR, linewidth=2.7, marker="o", label="TRW")
    ax.fill_between(unique_clues, trw_m - trw_s, trw_m + trw_s, color=TRW_COLOR, alpha=0.16)
    ax.set_xlabel("Clue count")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Sudoku solve rate vs clue count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "solve_rate_vs_clue_count.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # quantile_pareto_runtime_vs_solve.png
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    marker_map = {"hard_q": "s", "medium_q": "o", "easy_q": "^"}
    for row in quantile_table:
        method = row["method"]
        bucket = row["difficulty_bucket"]
        color = BP_COLOR if method == "bp" else TRW_COLOR
        ax.scatter(
            row["mean_runtime_sec"],
            row["solve_rate"],
            s=60 + row["count"] * 3,
            color=color,
            marker=marker_map[bucket],
            edgecolors="black",
            linewidths=0.7,
            alpha=0.88,
        )
        ax.annotate(
            f"{method.upper()}-{bucket}",
            (row["mean_runtime_sec"], row["solve_rate"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    ax.set_xlabel("Mean runtime (s)")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Sudoku quantile difficulty Pareto view")
    plt.tight_layout()
    fig.savefig(output_dir / "quantile_pareto_runtime_vs_solve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # trw_bp_gap_by_difficulty.png
    order = ["hard_q", "medium_q", "easy_q"]
    label_order = ["Hard", "Medium", "Easy"]
    row_map = {(r["method"], r["difficulty_bucket"]): r for r in quantile_table}
    solve_gap = np.array(
        [row_map[("trw", b)]["solve_rate"] - row_map[("bp", b)]["solve_rate"] for b in order],
        dtype=np.float64,
    )
    runtime_ratio = np.array(
        [row_map[("trw", b)]["mean_runtime_sec"] / max(row_map[("bp", b)]["mean_runtime_sec"], 1e-12) for b in order],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    x = np.arange(len(order))
    colors = [BP_COLOR if v >= 0 else TRW_COLOR for v in solve_gap]
    axes[0].bar(x, solve_gap, color=colors, alpha=0.88)
    axes[0].axhline(0.0, color="black", linewidth=1.2)
    axes[0].set_ylabel("Solve-rate gap\n(TRW - BP)")
    axes[0].set_title("TRW vs BP gap by quantile difficulty")

    axes[1].bar(x, runtime_ratio, color=[TRW_COLOR] * len(order), alpha=0.8)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Runtime ratio\n(TRW / BP)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(label_order)
    axes[1].set_xlabel("Difficulty bucket (quantiles)")

    plt.tight_layout()
    fig.savefig(output_dir / "trw_bp_gap_by_difficulty.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_iou_hist(
    bp_rows: Sequence[Dict[str, float]],
    trw_rows: Sequence[Dict[str, float]],
    title: str,
    output_path: Path,
    annotate_zero_outlier: bool = False,
) -> None:
    bp_iou = np.array([r["iou"] for r in bp_rows], dtype=np.float64)
    trw_iou = np.array([r["iou"] for r in trw_rows], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, 12)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.hist(bp_iou, bins=bins, alpha=0.55, color=BP_COLOR, edgecolor="black", label="BP")
    ax.hist(trw_iou, bins=bins, alpha=0.55, color=TRW_COLOR, edgecolor="black", label="TRW")
    if annotate_zero_outlier:
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.text(0.01, ax.get_ylim()[1] * 0.9, "outlier at IoU=0", fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Image count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_quantile_table(path: Path) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "method": row["method"],
                    "difficulty_bucket": row["difficulty_bucket"],
                    "count": float(row["count"]),
                    "solve_rate": float(row["solve_rate"]),
                    "mean_runtime_sec": float(row["mean_runtime_sec"]),
                }
            )
    return out


def _load_summary(path: Path) -> Dict[str, float]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def generate(args: argparse.Namespace) -> None:
    _configure_matplotlib()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation metrics/summaries.
    bsds_bp_rows = _read_csv(PROJECT_ROOT / "results" / "segmentation" / "bsds500" / "metrics.csv")
    bsds_trw_rows = _read_csv(PROJECT_ROOT / "results" / "segmentation" / "bsds500" / "bsds500_trw_metrics.csv")
    grab_bp_rows = _read_csv(PROJECT_ROOT / "results" / "segmentation" / "grabcut" / "metrics.csv")
    grab_trw_rows = _read_csv(PROJECT_ROOT / "results" / "segmentation" / "grabcut" / "grabcut_trw_metrics.csv")

    bsds_bp_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "bsds500" / "summary.json")
    bsds_trw_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "bsds500" / "bsds500_trw_summary.json")
    grab_bp_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "grabcut" / "summary.json")
    grab_trw_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "grabcut" / "grabcut_trw_summary.json")

    # Oracle datasets.
    bsds_images, bsds_masks = _load_oracle_segmentation_data(
        "bsds500",
        data_path=Path(args.bsds_path),
        resize=(60, 60),
        max_images=args.max_images,
    )
    grab_images, grab_masks = _load_oracle_segmentation_data(
        "grabcut",
        data_path=Path(args.grabcut_path),
        resize=(60, 60),
        max_images=args.max_images,
    )

    # Best/median/worst indices from BP IoU.
    bsds_selected = _selected_worst_median_best(bsds_bp_rows, score_key="iou")
    grab_selected = _selected_worst_median_best(grab_bp_rows, score_key="iou")

    # Existing BP predictions.
    bsds_bp_pred_dir = PROJECT_ROOT / "results" / "segmentation" / "bsds500" / "predictions"
    grab_bp_pred_dir = PROJECT_ROOT / "results" / "segmentation" / "grabcut" / "predictions"
    bsds_bp_preds = {i: _load_binary_image(bsds_bp_pred_dir / f"sample_{i:03d}_pred.png") for i in bsds_selected}
    grab_bp_preds = {i: _load_binary_image(grab_bp_pred_dir / f"sample_{i:03d}_pred.png") for i in grab_selected}

    # Run inference to obtain convergence curves and TRW selected predictions.
    bsds_bp_curves, _ = _run_inference_curves_and_selected_preds(
        images=bsds_images,
        masks=bsds_masks,
        selected_indices=[],
        lambda_=float(bsds_bp_summary["config"]["lambda"]),
        max_iters=int(bsds_bp_summary["config"]["max_iters"]),
        tol=float(bsds_bp_summary["config"]["tol"]),
        damping=float(bsds_bp_summary["config"]["damping"]),
        method="bp",
    )
    bsds_trw_curves, bsds_trw_preds = _run_inference_curves_and_selected_preds(
        images=bsds_images,
        masks=bsds_masks,
        selected_indices=bsds_selected,
        lambda_=float(bsds_trw_summary["config"]["lambda"]),
        max_iters=int(bsds_trw_summary["config"]["max_iters"]),
        tol=float(bsds_trw_summary["config"]["tol"]),
        damping=float(bsds_trw_summary["config"]["damping"]),
        method="trw",
    )
    grab_bp_curves, _ = _run_inference_curves_and_selected_preds(
        images=grab_images,
        masks=grab_masks,
        selected_indices=[],
        lambda_=float(grab_bp_summary["config"]["lambda"]),
        max_iters=int(grab_bp_summary["config"]["max_iters"]),
        tol=float(grab_bp_summary["config"]["tol"]),
        damping=float(grab_bp_summary["config"]["damping"]),
        method="bp",
    )
    grab_trw_curves, grab_trw_preds = _run_inference_curves_and_selected_preds(
        images=grab_images,
        masks=grab_masks,
        selected_indices=grab_selected,
        lambda_=float(grab_trw_summary["config"]["lambda"]),
        max_iters=int(grab_trw_summary["config"]["max_iters"]),
        tol=float(grab_trw_summary["config"]["tol"]),
        damping=float(grab_trw_summary["config"]["damping"]),
        method="trw",
    )

    # 1,2,4 qualitative oracle figures.
    _save_qualitative_grid(
        images=bsds_images,
        masks=bsds_masks,
        predictions=bsds_bp_preds,
        selected_indices=bsds_selected,
        metric_rows=bsds_bp_rows,
        output_path=output_dir / "best_median_worst_bsds500_bp.png",
        pred_title="BP prediction",
    )
    _save_qualitative_grid(
        images=bsds_images,
        masks=bsds_masks,
        predictions=bsds_trw_preds,
        selected_indices=bsds_selected,
        metric_rows=bsds_trw_rows,
        output_path=output_dir / "best_median_worst_bsds500_trw.png",
        pred_title="TRW prediction",
    )
    _save_qualitative_grid(
        images=grab_images,
        masks=grab_masks,
        predictions=grab_bp_preds,
        selected_indices=grab_selected,
        metric_rows=grab_bp_rows,
        output_path=output_dir / "best_median_worst_grabcut_bp.png",
        pred_title="BP prediction",
    )
    _save_qualitative_grid(
        images=grab_images,
        masks=grab_masks,
        predictions=grab_trw_preds,
        selected_indices=grab_selected,
        metric_rows=grab_trw_rows,
        output_path=output_dir / "best_median_worst_grabcut_trw.png",
        pred_title="TRW prediction",
    )

    # 3 and 4 convergence plots.
    _save_convergence_plot(
        bp_curves=bsds_bp_curves,
        trw_curves=bsds_trw_curves,
        tol=1e-3,
        title="BSDS500 convergence curves (BP vs TRW)",
        output_path=output_dir / "convergence_curves_bsds500.png",
    )
    _save_convergence_plot(
        bp_curves=grab_bp_curves,
        trw_curves=grab_trw_curves,
        tol=1e-3,
        title="GrabCut convergence curves (BP vs TRW)",
        output_path=output_dir / "convergence_curves_grabcut.png",
    )

    # 5 and 6 EM track figures.
    em_bsds_bp_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "bsds500_em_bp" / "bsds500_bp_em_summary.json")
    em_bsds_trw_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "bsds500_em_trw" / "bsds500_trw_em_summary.json")
    em_grab_bp_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "grabcut_em_bp" / "grabcut_bp_em_summary.json")
    em_grab_trw_summary = _load_summary(PROJECT_ROOT / "results" / "segmentation" / "grabcut_em_trw" / "grabcut_trw_em_summary.json")
    _save_em_vs_oracle_bar(
        summary_oracle_bsds_bp=bsds_bp_summary,
        summary_oracle_bsds_trw=bsds_trw_summary,
        summary_oracle_grab_bp=grab_bp_summary,
        summary_oracle_grab_trw=grab_trw_summary,
        summary_em_bsds_bp=em_bsds_bp_summary,
        summary_em_bsds_trw=em_bsds_trw_summary,
        summary_em_grab_bp=em_grab_bp_summary,
        summary_em_grab_trw=em_grab_trw_summary,
        output_path=output_dir / "em_vs_oracle_comparison.png",
    )
    em_bsds_rows = _read_csv(PROJECT_ROOT / "results" / "segmentation" / "bsds500_em_bp" / "bsds500_bp_em_metrics.csv")
    _save_em_best_median_worst_bsds(
        images_bsds=bsds_images,
        masks_bsds=bsds_masks,
        em_rows=em_bsds_rows,
        pred_eval_dir=PROJECT_ROOT / "results" / "segmentation" / "bsds500_em_bp" / "predictions_eval",
        output_path=output_dir / "em_best_median_worst_bsds500.png",
    )

    # 7,8,9 Sudoku overall.
    sudoku_bp_summary = _load_summary(PROJECT_ROOT / "results" / "sudoku" / "summary.json")
    sudoku_trw_summary = _load_summary(PROJECT_ROOT / "results" / "sudoku" / "sudoku_trw_summary.json")
    sudoku_bp_rows = _read_csv(PROJECT_ROOT / "results" / "sudoku" / "metrics.csv")
    sudoku_trw_rows = _read_csv(PROJECT_ROOT / "results" / "sudoku" / "sudoku_trw_metrics.csv")
    _save_sudoku_overall_figures(
        summary_bp=sudoku_bp_summary,
        summary_trw=sudoku_trw_summary,
        bp_rows=sudoku_bp_rows,
        trw_rows=sudoku_trw_rows,
        output_dir=output_dir,
    )

    # 10,11,12 Sudoku difficulty figures.
    clues = _parse_clues_from_examples(PROJECT_ROOT / "results" / "sudoku" / "examples", n_samples=min(len(sudoku_bp_rows), len(sudoku_trw_rows)))
    quantile_table = _load_quantile_table(PROJECT_ROOT / "results" / "sudoku" / "difficulty_analysis" / "difficulty_quantile_table.csv")
    _save_sudoku_difficulty_figures(
        bp_rows=sudoku_bp_rows,
        trw_rows=sudoku_trw_rows,
        clues=clues,
        quantile_table=quantile_table,
        output_dir=output_dir,
    )

    # 13,14 IoU distribution appendix.
    _save_iou_hist(
        bp_rows=bsds_bp_rows,
        trw_rows=bsds_trw_rows,
        title="BSDS500 IoU distribution (BP vs TRW)",
        output_path=output_dir / "iou_hist_bsds500.png",
        annotate_zero_outlier=False,
    )
    _save_iou_hist(
        bp_rows=grab_bp_rows,
        trw_rows=grab_trw_rows,
        title="GrabCut IoU distribution (BP vs TRW)",
        output_path=output_dir / "iou_hist_grabcut.png",
        annotate_zero_outlier=True,
    )

    print(f"Slide figures saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bsds-path", type=str, default="data/BSDS500")
    parser.add_argument("--grabcut-path", type=str, default="data/grabcut")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results/slides_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
