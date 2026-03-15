"""Run BSDS500 segmentation with Tree-Reweighted Belief Propagation (TRW-BP)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bsds500_loader import load_bsds500_dataset
from src.trw_belief_propagation import TreeReweightedBeliefPropagation
from utils.metrics import compute_iou, pixel_accuracy, save_metrics_csv, summarize_segmentation_metrics
from utils.segmentation_utils import beliefs_to_mask, build_binary_segmentation_mrf
from utils.trw_utils import compute_uniform_edge_weights


def _save_convergence_plot(curves: List[List[float]], output_path: Path) -> None:
    if not curves:
        return

    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, curve in enumerate(curves):
        mat[i, : len(curve)] = np.asarray(curve, dtype=np.float64)

    mean_curve = np.nanmean(mat, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for curve in curves:
        ax.plot(np.arange(1, len(curve) + 1), curve, color="#B8B8D1", alpha=0.4)
    ax.plot(
        np.arange(1, mean_curve.size + 1),
        mean_curve,
        color="#6A3D9A",
        linewidth=2.4,
        label="TRW mean message change",
    )
    ax.set_title("TRW Convergence Curves (BSDS500)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max message difference")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    resize = tuple(args.resize) if args.resize is not None else None
    images, masks = load_bsds500_dataset(
        args.data_path,
        max_images=args.max_images,
        resize=resize,
    )

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    curves: List[List[float]] = []

    for idx, (image, gt_mask_raw) in enumerate(zip(images, masks)):
        gt_mask = (np.asarray(gt_mask_raw) > 0).astype(np.uint8)
        mask_for_unary = gt_mask if args.use_gt_unary else None

        t0 = time.perf_counter()
        mrf = build_binary_segmentation_mrf(
            image=image,
            mask_for_unary=mask_for_unary,
            lambda_=args.lambda_,
        )
        rho = compute_uniform_edge_weights(mrf)
        trw = TreeReweightedBeliefPropagation(
            mrf,
            rho=rho,
            max_iters=args.max_iters,
            tol=args.tol,
            damping=args.damping,
        )
        converged = trw.run()
        beliefs = trw.compute_beliefs()
        pred_mask = beliefs_to_mask(beliefs, gt_mask.shape)
        elapsed = time.perf_counter() - t0

        iou = compute_iou(pred_mask, gt_mask)
        acc = pixel_accuracy(pred_mask, gt_mask)

        rows.append(
            {
                "index": idx,
                "iou": iou,
                "accuracy": acc,
                "runtime_sec": elapsed,
                "iterations": float(trw.num_iters),
                "converged": float(converged),
                "fg_ratio": float(np.mean(pred_mask)),
            }
        )
        curves.append(list(trw.message_deltas))

        print(
            f"[TRW {idx + 1}/{len(images)}] IoU={iou:.4f} Acc={acc:.4f} "
            f"Runtime={elapsed:.2f}s Iter={trw.num_iters} Converged={converged}"
        )

    save_metrics_csv(rows, results_dir / "bsds500_trw_metrics.csv")
    _save_convergence_plot(curves, plots_dir / "bsds500_trw_convergence_curves.png")

    iou_scores = [float(row["iou"]) for row in rows]
    acc_scores = [float(row["accuracy"]) for row in rows]
    metric_summary = summarize_segmentation_metrics(iou_scores, acc_scores)

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
            "rho": "uniform_0.5",
        },
    }

    with (results_dir / "bsds500_trw_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="data/BSDS500")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--resize", type=int, nargs=2, default=[60, 60])
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results/segmentation/bsds500")
    parser.add_argument("--no-gt-unary", dest="use_gt_unary", action="store_false")
    parser.set_defaults(use_gt_unary=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    summary = run_experiment(args)
    print("\nBSDS500 TRW experiment summary")
    print(f"Images: {summary['num_images']}")
    print(f"Mean IoU: {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Mean runtime (s): {summary['mean_runtime_sec']:.3f}")
    print(f"Mean iterations: {summary['mean_iterations']:.2f}")
    print(f"Convergence rate: {summary['convergence_rate']:.3f}")


if __name__ == "__main__":
    main()
