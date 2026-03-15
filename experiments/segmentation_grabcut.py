"""Run GrabCut binary segmentation experiments with loopy belief propagation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

# Allow running as: python experiments/segmentation_grabcut.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.grabcut_loader import load_grabcut_dataset
from src.belief_propagation import BeliefPropagation
from utils.metrics import (
    compute_iou,
    pixel_accuracy,
    save_metrics_csv,
    summarize_segmentation_metrics,
)
from utils.segmentation_utils import beliefs_to_mask, build_binary_segmentation_mrf


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    resize: Tuple[int, int] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image/mask pair while preserving binary mask structure."""
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


def _display_image(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")


def _save_distribution_plot(values: List[float], title: str, xlabel: str, output_path: Path) -> None:
    """Save histogram of a scalar metric."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(10, max(5, len(values))), color="#55A868", edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_curves(curves: List[List[float]], output_path: Path) -> None:
    """Save per-sample and mean BP message-delta convergence curves."""
    if not curves:
        return

    max_len = max(len(c) for c in curves)
    curve_matrix = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, curve in enumerate(curves):
        curve_matrix[i, : len(curve)] = np.asarray(curve, dtype=np.float64)

    mean_curve = np.nanmean(curve_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for curve in curves:
        ax.plot(np.arange(1, len(curve) + 1), curve, color="#C7D4B6", alpha=0.45, linewidth=1.2)

    ax.plot(
        np.arange(1, mean_curve.size + 1),
        mean_curve,
        color="#2E8B57",
        linewidth=2.5,
        label="Mean message change",
    )
    ax.set_title("GrabCut BP Convergence Curves")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max message difference")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_selected_examples(samples: List[Dict[str, object]], output_path: Path) -> None:
    """Save best/median/worst qualitative panel based on IoU."""
    if not samples:
        return

    ious = np.array([float(sample["iou"]) for sample in samples], dtype=np.float64)
    order = np.argsort(ious)
    chosen = [order[-1], order[len(order) // 2], order[0]]
    labels = ["Best IoU", "Median IoU", "Worst IoU"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for row_idx, sample_idx in enumerate(chosen):
        sample = samples[int(sample_idx)]
        image = np.asarray(sample["image"])
        gt = np.asarray(sample["gt"])
        pred = np.asarray(sample["pred"])

        _display_image(axes[row_idx, 0], image, f"{labels[row_idx]} | Image")
        axes[row_idx, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Ground Truth")
        axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title(
            f"Prediction\nIoU={float(sample['iou']):.3f}, Acc={float(sample['accuracy']):.3f}"
        )
        axes[row_idx, 2].axis("off")

    fig.suptitle("GrabCut Qualitative Comparison")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    """Execute BP segmentation benchmark over GrabCut subset."""
    resize = tuple(args.resize) if args.resize is not None else None
    images, masks = load_grabcut_dataset(args.data_path, max_images=args.max_images)

    results_dir = Path(args.results_dir)
    predictions_dir = results_dir / "predictions"
    plots_dir = results_dir / "plots"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    curves: List[List[float]] = []
    samples: List[Dict[str, object]] = []

    for idx, (image_raw, mask_raw) in enumerate(zip(images, masks)):
        image, gt_mask = _resize_pair(image_raw, (mask_raw > 0).astype(np.uint8), resize)
        mask_for_unary = gt_mask if args.use_gt_unary else None

        t0 = time.perf_counter()
        mrf = build_binary_segmentation_mrf(
            image=image,
            mask_for_unary=mask_for_unary,
            lambda_=args.lambda_,
        )
        bp = BeliefPropagation(
            mrf,
            max_iters=args.max_iters,
            tol=args.tol,
            damping=args.damping,
        )
        converged = bp.run()
        beliefs = bp.compute_beliefs()
        pred_mask = beliefs_to_mask(beliefs, gt_mask.shape)
        elapsed = time.perf_counter() - t0

        iou = compute_iou(pred_mask, gt_mask)
        acc = pixel_accuracy(pred_mask, gt_mask)

        io.imsave(
            predictions_dir / f"sample_{idx:03d}_pred.png",
            (pred_mask * 255).astype(np.uint8),
            check_contrast=False,
        )

        row = {
            "index": idx,
            "iou": iou,
            "accuracy": acc,
            "runtime_sec": elapsed,
            "iterations": float(bp.num_iters),
            "converged": float(converged),
            "fg_ratio": float(np.mean(pred_mask)),
        }
        rows.append(row)
        curves.append(list(bp.message_deltas))
        samples.append(
            {
                "image": image,
                "gt": gt_mask,
                "pred": pred_mask,
                "iou": iou,
                "accuracy": acc,
            }
        )

        print(
            f"[{idx + 1}/{len(images)}] IoU={iou:.4f} Acc={acc:.4f} "
            f"Runtime={elapsed:.2f}s Iter={bp.num_iters} Converged={converged}"
        )

    save_metrics_csv(rows, results_dir / "metrics.csv")

    iou_scores = [float(row["iou"]) for row in rows]
    acc_scores = [float(row["accuracy"]) for row in rows]
    metric_summary = summarize_segmentation_metrics(iou_scores, acc_scores)

    _save_distribution_plot(iou_scores, "IoU Distribution (GrabCut)", "IoU", plots_dir / "iou_hist.png")
    _save_distribution_plot(
        acc_scores,
        "Pixel Accuracy Distribution (GrabCut)",
        "Accuracy",
        plots_dir / "accuracy_hist.png",
    )
    _save_convergence_curves(curves, plots_dir / "convergence_curves.png")
    _save_selected_examples(samples, plots_dir / "best_median_worst.png")

    summary = {
        "num_images": len(rows),
        "mean_iou": metric_summary["mean_iou"],
        "std_iou": metric_summary["std_iou"],
        "mean_accuracy": metric_summary["mean_accuracy"],
        "std_accuracy": metric_summary["std_accuracy"],
        "mean_runtime_sec": float(np.mean([row["runtime_sec"] for row in rows])),
        "mean_iterations": float(np.mean([row["iterations"] for row in rows])),
        "convergence_rate": float(np.mean([row["converged"] for row in rows])),
        "config": {
            "data_path": str(args.data_path),
            "max_images": args.max_images,
            "resize": list(resize) if resize is not None else None,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "damping": args.damping,
            "lambda": args.lambda_,
            "use_gt_unary": args.use_gt_unary,
            "seed": args.seed,
        },
    }

    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="data/grabcut")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--resize", type=int, nargs=2, default=[60, 60])
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results/segmentation/grabcut")
    parser.add_argument("--no-gt-unary", dest="use_gt_unary", action="store_false")
    parser.set_defaults(use_gt_unary=True)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    np.random.seed(args.seed)

    summary = run_experiment(args)
    print("\nGrabCut experiment summary")
    print(f"Images: {summary['num_images']}")
    print(f"Mean IoU: {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Mean runtime (s): {summary['mean_runtime_sec']:.3f}")
    print(f"Mean iterations: {summary['mean_iterations']:.2f}")
    print(f"Convergence rate: {summary['convergence_rate']:.3f}")


if __name__ == "__main__":
    main()
