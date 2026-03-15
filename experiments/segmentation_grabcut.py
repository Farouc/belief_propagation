"""Run GrabCut binary segmentation experiments with loopy belief propagation."""

from __future__ import annotations

import argparse
import csv
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
from utils.metrics import compute_iou, pixel_accuracy
from utils.segmentation_utils import beliefs_to_mask, build_binary_segmentation_mrf


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    resize: Tuple[int, int] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image/mask pair while preserving binary mask structure."""
    if resize is None:
        return image, mask

    image_resized = transform.resize(
        image,
        resize,
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


def _save_triplet_plot(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save qualitative visualization for one sample."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    if image.ndim == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("BP Prediction")
    axes[1].axis("off")

    axes[2].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_runtime_plot(rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save runtime and convergence behavior plot."""
    indices = [int(row["index"]) for row in rows]
    runtimes = [float(row["runtime_sec"]) for row in rows]
    converged = [bool(row["converged"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2E8B57" if c else "#C44E52" for c in converged]
    ax.bar(indices, runtimes, color=colors)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Runtime (sec)")
    ax.set_title("GrabCut Runtime / Convergence")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
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

    for idx, (image_raw, mask_raw) in enumerate(zip(images, masks)):
        image, gt_mask = _resize_pair(image_raw, (mask_raw > 0).astype(np.uint8), resize)

        t0 = time.perf_counter()
        mrf = build_binary_segmentation_mrf(image=image, smoothness=args.smoothness)
        bp = BeliefPropagation(mrf, max_iters=args.max_iters, tol=args.tol)
        converged = bp.run()
        beliefs = bp.compute_beliefs()
        pred_mask = beliefs_to_mask(beliefs, gt_mask.shape)
        elapsed = time.perf_counter() - t0

        iou = compute_iou(pred_mask, gt_mask)
        acc = pixel_accuracy(pred_mask, gt_mask)

        pred_path = predictions_dir / f"sample_{idx:03d}_pred.png"
        io.imsave(pred_path, (pred_mask * 255).astype(np.uint8), check_contrast=False)

        plot_path = plots_dir / f"sample_{idx:03d}_comparison.png"
        _save_triplet_plot(
            image=image,
            prediction=pred_mask,
            ground_truth=gt_mask,
            output_path=plot_path,
            title=(
                f"Sample {idx:03d} | IoU={iou:.3f} | Acc={acc:.3f} | "
                f"Conv={converged} | Iter={bp.num_iters}"
            ),
        )

        row = {
            "index": idx,
            "iou": iou,
            "accuracy": acc,
            "runtime_sec": elapsed,
            "iterations": float(bp.num_iters),
            "converged": float(converged),
        }
        rows.append(row)

        print(
            f"[{idx + 1}/{len(images)}] IoU={iou:.4f} Acc={acc:.4f} "
            f"Runtime={elapsed:.2f}s Iter={bp.num_iters} Converged={converged}"
        )

    csv_path = results_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "iou", "accuracy", "runtime_sec", "iterations", "converged"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _save_runtime_plot(rows, plots_dir / "runtime_convergence.png")

    summary = {
        "num_images": len(rows),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "mean_runtime_sec": float(np.mean([row["runtime_sec"] for row in rows])),
        "mean_iterations": float(np.mean([row["iterations"] for row in rows])),
        "convergence_rate": float(np.mean([row["converged"] for row in rows])),
        "config": {
            "data_path": str(args.data_path),
            "max_images": args.max_images,
            "resize": list(resize) if resize is not None else None,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "smoothness": args.smoothness,
            "seed": args.seed,
        },
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="data/grabcut")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--resize", type=int, nargs=2, default=[80, 80])
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--smoothness", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results/segmentation/grabcut")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    np.random.seed(args.seed)

    summary = run_experiment(args)
    print("\nGrabCut experiment summary")
    print(f"Images: {summary['num_images']}")
    print(f"Mean IoU: {summary['mean_iou']:.4f}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f}")
    print(f"Mean runtime (s): {summary['mean_runtime_sec']:.3f}")
    print(f"Mean iterations: {summary['mean_iterations']:.2f}")
    print(f"Convergence rate: {summary['convergence_rate']:.3f}")


if __name__ == "__main__":
    main()
