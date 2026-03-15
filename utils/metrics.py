"""Evaluation metrics for segmentation and BP experiments."""

from __future__ import annotations

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
