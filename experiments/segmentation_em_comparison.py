"""Segmentation experiments with unsupervised EM unary potentials.

This script does not use ground-truth masks to build unary potentials.
It supports both BP and TRW inference while keeping outputs separate from
existing experiment folders.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bsds500_loader import load_bsds500_dataset
from datasets.grabcut_loader import load_grabcut_dataset
from src.belief_propagation import BeliefPropagation
from src.trw_belief_propagation import TreeReweightedBeliefPropagation
from utils.metrics import (
    compute_iou,
    pixel_accuracy,
    save_metrics_csv,
    summarize_segmentation_metrics,
)
from utils.segmentation_em_utils import build_binary_segmentation_mrf_em
from utils.segmentation_utils import beliefs_to_mask
from utils.trw_utils import compute_uniform_edge_weights


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


def _display_image(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")


def _save_triplet_plot(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    _display_image(axes[0], image, "Image")
    axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    axes[2].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_dual_hist(
    raw_values: List[float],
    aligned_values: List[float],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = min(10, max(5, len(raw_values)))
    ax.hist(raw_values, bins=bins, alpha=0.65, label="Raw labels", color="#4C72B0", edgecolor="black")
    ax.hist(
        aligned_values,
        bins=bins,
        alpha=0.65,
        label="Label-aligned for eval",
        color="#55A868",
        edgecolor="black",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_curves(curves: List[List[float]], output_path: Path, title: str) -> None:
    if not curves:
        return

    max_len = max(len(c) for c in curves)
    curve_matrix = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, curve in enumerate(curves):
        curve_matrix[i, : len(curve)] = np.asarray(curve, dtype=np.float64)

    mean_curve = np.nanmean(curve_matrix, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for curve in curves:
        ax.plot(np.arange(1, len(curve) + 1), curve, color="#BFBFBF", alpha=0.45, linewidth=1.2)

    ax.plot(
        np.arange(1, mean_curve.size + 1),
        mean_curve,
        color="#C44E52",
        linewidth=2.4,
        label="Mean message delta",
    )
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max message difference")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_selected_examples(samples: List[Dict[str, object]], output_path: Path, title: str) -> None:
    if not samples:
        return

    ious = np.array([float(s["iou_eval"]) for s in samples], dtype=np.float64)
    order = np.argsort(ious)
    chosen = [order[0], order[len(order) // 2], order[-1]]
    labels = ["Worst IoU", "Median IoU", "Best IoU"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row_idx, sample_idx in enumerate(chosen):
        sample = samples[int(sample_idx)]
        image = np.asarray(sample["image"])
        gt = np.asarray(sample["gt"])
        pred = np.asarray(sample["pred_eval"])

        _display_image(axes[row_idx, 0], image, f"{labels[row_idx]} | Image")
        axes[row_idx, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Ground Truth")
        axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title(
            f"Prediction\nIoU={float(sample['iou_eval']):.3f}, Acc={float(sample['acc_eval']):.3f}"
        )
        axes[row_idx, 2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _load_dataset(
    dataset: str,
    data_path: str,
    max_images: int,
    resize: Tuple[int, int] | None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if dataset == "bsds500":
        return load_bsds500_dataset(data_path, max_images=max_images, resize=resize)

    if dataset == "grabcut":
        images, masks = load_grabcut_dataset(data_path, max_images=max_images)
        if resize is None:
            return images, [(np.asarray(m) > 0).astype(np.uint8) for m in masks]
        resized_images: List[np.ndarray] = []
        resized_masks: List[np.ndarray] = []
        for image, mask in zip(images, masks):
            image_r, mask_r = _resize_pair(image, (np.asarray(mask) > 0).astype(np.uint8), resize)
            resized_images.append(image_r)
            resized_masks.append(mask_r)
        return resized_images, resized_masks

    raise ValueError(f"Unsupported dataset: {dataset}")


def _run_inference(
    inference: str,
    mrf,
    max_iters: int,
    tol: float,
    damping: float,
):
    if inference == "bp":
        model = BeliefPropagation(mrf, max_iters=max_iters, tol=tol, damping=damping)
    elif inference == "trw":
        rho = compute_uniform_edge_weights(mrf)
        model = TreeReweightedBeliefPropagation(
            mrf,
            rho=rho,
            max_iters=max_iters,
            tol=tol,
            damping=damping,
        )
    else:
        raise ValueError(f"Unsupported inference method: {inference}")

    converged = model.run()
    beliefs = model.compute_beliefs()
    return model, beliefs, converged


def _default_results_dir(dataset: str, inference: str) -> Path:
    return PROJECT_ROOT / "results" / "segmentation" / f"{dataset}_em_{inference}"


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    resize = tuple(args.resize) if args.resize is not None else None
    images, masks = _load_dataset(
        dataset=args.dataset,
        data_path=args.data_path,
        max_images=args.max_images,
        resize=resize,
    )

    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir(args.dataset, args.inference)
    predictions_raw_dir = results_dir / "predictions_raw"
    predictions_eval_dir = results_dir / "predictions_eval"
    plots_dir = results_dir / "plots"
    predictions_raw_dir.mkdir(parents=True, exist_ok=True)
    predictions_eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    curves: List[List[float]] = []
    samples: List[Dict[str, object]] = []

    for idx, (image, gt_mask_raw) in enumerate(zip(images, masks)):
        gt_mask = (np.asarray(gt_mask_raw) > 0).astype(np.uint8)

        t0 = time.perf_counter()
        mrf, em_info = build_binary_segmentation_mrf_em(
            image=image,
            lambda_=args.lambda_,
            em_max_iters=args.em_max_iters,
            em_tol=args.em_tol,
            em_regularization=args.em_regularization,
            em_seed=args.seed + idx,
        )
        inference_model, beliefs, converged = _run_inference(
            inference=args.inference,
            mrf=mrf,
            max_iters=args.max_iters,
            tol=args.tol,
            damping=args.damping,
        )
        pred_raw = beliefs_to_mask(beliefs, gt_mask.shape)
        runtime_sec = time.perf_counter() - t0

        # Raw metrics preserve arbitrary binary label identity from unsupervised EM.
        iou_raw = compute_iou(pred_raw, gt_mask)
        acc_raw = pixel_accuracy(pred_raw, gt_mask)

        # For fair unsupervised evaluation, also report best label permutation.
        pred_flipped = 1 - pred_raw
        iou_flipped = compute_iou(pred_flipped, gt_mask)
        acc_flipped = pixel_accuracy(pred_flipped, gt_mask)
        use_flipped = iou_flipped > iou_raw
        pred_eval = pred_flipped if use_flipped else pred_raw
        iou_eval = iou_flipped if use_flipped else iou_raw
        acc_eval = acc_flipped if use_flipped else acc_raw

        io.imsave(
            predictions_raw_dir / f"sample_{idx:03d}_pred_raw.png",
            (pred_raw * 255).astype(np.uint8),
            check_contrast=False,
        )
        io.imsave(
            predictions_eval_dir / f"sample_{idx:03d}_pred_eval.png",
            (pred_eval * 255).astype(np.uint8),
            check_contrast=False,
        )

        _save_triplet_plot(
            image=image,
            prediction=pred_eval,
            ground_truth=gt_mask,
            output_path=plots_dir / f"sample_{idx:03d}_comparison.png",
            title=(
                f"{args.dataset.upper()} {args.inference.upper()} EM | Sample {idx:03d} | "
                f"IoU(raw/eval)={iou_raw:.3f}/{iou_eval:.3f} | "
                f"Conv={converged} Iter={inference_model.num_iters}"
            ),
        )

        row = {
            "index": idx,
            "iou_raw": iou_raw,
            "accuracy_raw": acc_raw,
            "iou_eval": iou_eval,
            "accuracy_eval": acc_eval,
            "label_flipped_for_eval": float(use_flipped),
            "runtime_sec": runtime_sec,
            "iterations": float(inference_model.num_iters),
            "converged": float(converged),
            "fg_ratio_raw": float(np.mean(pred_raw)),
            "fg_ratio_eval": float(np.mean(pred_eval)),
            "em_iterations": float(int(em_info["iterations"])),
            "em_log_likelihood": float(em_info["log_likelihood"]),
        }
        rows.append(row)
        curves.append(list(inference_model.message_deltas))
        samples.append(
            {
                "image": image,
                "gt": gt_mask,
                "pred_eval": pred_eval,
                "iou_eval": iou_eval,
                "acc_eval": acc_eval,
            }
        )

        print(
            f"[{idx + 1}/{len(images)}] {args.inference.upper()} "
            f"IoU(raw/eval)={iou_raw:.4f}/{iou_eval:.4f} "
            f"Acc(raw/eval)={acc_raw:.4f}/{acc_eval:.4f} "
            f"Runtime={runtime_sec:.2f}s Iter={inference_model.num_iters} Conv={converged}"
        )

    metrics_filename = f"{args.dataset}_{args.inference}_em_metrics.csv"
    save_metrics_csv(rows, results_dir / metrics_filename)

    iou_raw_scores = [float(r["iou_raw"]) for r in rows]
    acc_raw_scores = [float(r["accuracy_raw"]) for r in rows]
    iou_eval_scores = [float(r["iou_eval"]) for r in rows]
    acc_eval_scores = [float(r["accuracy_eval"]) for r in rows]

    raw_summary = summarize_segmentation_metrics(iou_raw_scores, acc_raw_scores)
    eval_summary = summarize_segmentation_metrics(iou_eval_scores, acc_eval_scores)

    _save_dual_hist(
        raw_values=iou_raw_scores,
        aligned_values=iou_eval_scores,
        title=f"{args.dataset.upper()} {args.inference.upper()} EM IoU Distribution",
        xlabel="IoU",
        output_path=plots_dir / f"{args.dataset}_{args.inference}_em_iou_hist.png",
    )
    _save_dual_hist(
        raw_values=acc_raw_scores,
        aligned_values=acc_eval_scores,
        title=f"{args.dataset.upper()} {args.inference.upper()} EM Accuracy Distribution",
        xlabel="Pixel accuracy",
        output_path=plots_dir / f"{args.dataset}_{args.inference}_em_accuracy_hist.png",
    )
    _save_convergence_curves(
        curves=curves,
        output_path=plots_dir / f"{args.dataset}_{args.inference}_em_convergence_curves.png",
        title=f"{args.dataset.upper()} {args.inference.upper()} EM Convergence",
    )
    _save_selected_examples(
        samples=samples,
        output_path=plots_dir / f"{args.dataset}_{args.inference}_em_best_median_worst.png",
        title=f"{args.dataset.upper()} {args.inference.upper()} EM Qualitative Comparison",
    )

    summary = {
        "dataset": args.dataset,
        "inference": args.inference,
        "num_images": len(rows),
        "mean_iou_raw": raw_summary["mean_iou"],
        "std_iou_raw": raw_summary["std_iou"],
        "mean_accuracy_raw": raw_summary["mean_accuracy"],
        "std_accuracy_raw": raw_summary["std_accuracy"],
        "mean_iou_eval": eval_summary["mean_iou"],
        "std_iou_eval": eval_summary["std_iou"],
        "mean_accuracy_eval": eval_summary["mean_accuracy"],
        "std_accuracy_eval": eval_summary["std_accuracy"],
        "flip_rate_for_eval": float(np.mean([r["label_flipped_for_eval"] for r in rows])),
        "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in rows])),
        "mean_iterations": float(np.mean([r["iterations"] for r in rows])),
        "convergence_rate": float(np.mean([r["converged"] for r in rows])),
        "mean_em_iterations": float(np.mean([r["em_iterations"] for r in rows])),
        "config": {
            "data_path": str(args.data_path),
            "max_images": args.max_images,
            "resize": list(resize) if resize is not None else None,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "damping": args.damping,
            "lambda": args.lambda_,
            "em_max_iters": args.em_max_iters,
            "em_tol": args.em_tol,
            "em_regularization": args.em_regularization,
            "seed": args.seed,
        },
    }

    summary_filename = f"{args.dataset}_{args.inference}_em_summary.json"
    with (results_dir / summary_filename).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, choices=["bsds500", "grabcut"], required=True)
    parser.add_argument("--inference", type=str, choices=["bp", "trw"], default="bp")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--resize", type=int, nargs=2, default=[60, 60])
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=2.0)
    parser.add_argument("--em-max-iters", type=int, default=60)
    parser.add_argument("--em-tol", type=float, default=1e-4)
    parser.add_argument("--em-regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="")
    args = parser.parse_args()

    if not args.data_path:
        args.data_path = "data/BSDS500" if args.dataset == "bsds500" else "data/grabcut"
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    summary = run_experiment(args)

    print(f"\n{args.dataset.upper()} {args.inference.upper()} EM summary")
    print(f"Images: {summary['num_images']}")
    print(
        f"Mean IoU raw/eval: {summary['mean_iou_raw']:.4f} ± {summary['std_iou_raw']:.4f} / "
        f"{summary['mean_iou_eval']:.4f} ± {summary['std_iou_eval']:.4f}"
    )
    print(
        f"Mean Acc raw/eval: {summary['mean_accuracy_raw']:.4f} ± {summary['std_accuracy_raw']:.4f} / "
        f"{summary['mean_accuracy_eval']:.4f} ± {summary['std_accuracy_eval']:.4f}"
    )
    print(f"Flip rate for eval: {summary['flip_rate_for_eval']:.3f}")
    print(f"Mean runtime (s): {summary['mean_runtime_sec']:.3f}")
    print(f"Mean iterations: {summary['mean_iterations']:.2f}")
    print(f"Convergence rate: {summary['convergence_rate']:.3f}")
    print(f"Mean EM iterations: {summary['mean_em_iterations']:.2f}")


if __name__ == "__main__":
    main()

