"""Evaluation metrics and summary utilities for BP experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def compute_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    """Compute binary Intersection-over-Union (IoU)."""
    pred_bin = np.asarray(pred).astype(bool)
    gt_bin = np.asarray(gt).astype(bool)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0
    return float(intersection / (union + eps))


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute segmentation pixel accuracy."""
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"Shape mismatch: pred {pred_arr.shape} vs gt {gt_arr.shape}")

    return float(np.mean(pred_arr == gt_arr))


def summarize_segmentation_metrics(
    iou_scores: Iterable[float],
    accuracy_scores: Iterable[float],
) -> dict:
    """Return dataset-level IoU/accuracy summary statistics."""
    iou = np.asarray(list(iou_scores), dtype=np.float64)
    acc = np.asarray(list(accuracy_scores), dtype=np.float64)
    if iou.size == 0 or acc.size == 0:
        raise ValueError("iou_scores and accuracy_scores must be non-empty")

    return {
        "mean_iou": float(np.mean(iou)),
        "std_iou": float(np.std(iou)),
        "mean_accuracy": float(np.mean(acc)),
        "std_accuracy": float(np.std(acc)),
        "min_iou": float(np.min(iou)),
        "max_iou": float(np.max(iou)),
        "min_accuracy": float(np.min(acc)),
        "max_accuracy": float(np.max(acc)),
    }


def save_metrics_csv(rows: Iterable[Mapping[str, object]], output_path: str | Path) -> None:
    """Save iterable of row dictionaries to a CSV file."""
    row_list = list(rows)
    if not row_list:
        raise ValueError("Cannot save empty metrics rows")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row_list[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)
