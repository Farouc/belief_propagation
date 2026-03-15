"""Utility functions to build common unary and pairwise potentials."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a normalized copy of ``vec`` with sum 1.

    Falls back to a uniform distribution if numerical issues occur.
    """
    arr = np.asarray(vec, dtype=np.float64)
    total = float(arr.sum())
    if total <= eps or not np.isfinite(total):
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return arr / total


def normalize_last_axis(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize an array along its last axis."""
    out = np.asarray(arr, dtype=np.float64)
    totals = out.sum(axis=-1, keepdims=True)
    bad = (totals <= eps) | ~np.isfinite(totals)
    totals = np.where(bad, 1.0, totals)
    out = out / totals
    if np.any(bad):
        k = out.shape[-1]
        out[bad.repeat(k, axis=-1)] = 1.0 / k
    return out


def potts_potential(
    num_states: int, same_weight: float = 2.0, diff_weight: float = 1.0
) -> np.ndarray:
    """Create a Potts pairwise potential matrix.

    Diagonal entries (equal labels) are ``same_weight`` and off-diagonal entries
    are ``diff_weight``.
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if same_weight < 0 or diff_weight < 0:
        raise ValueError("weights must be non-negative")

    mat = np.full((num_states, num_states), diff_weight, dtype=np.float64)
    np.fill_diagonal(mat, same_weight)
    return mat


def binary_unary_from_intensity(
    image: np.ndarray,
    means: Tuple[float, float] = (0.35, 0.65),
    sigma: float = 0.18,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build binary unary potentials from grayscale intensities.

    Parameters
    ----------
    image:
        2D grayscale image with values in [0, 1].
    means:
        Gaussian means for labels (background, foreground).
    sigma:
        Shared Gaussian standard deviation.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, 2) with normalized unary potentials.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")

    mu0, mu1 = means
    coeff = -0.5 / (sigma * sigma)
    p0 = np.exp(coeff * (img - mu0) ** 2)
    p1 = np.exp(coeff * (img - mu1) ** 2)

    unary = np.stack([p0, p1], axis=-1) + eps
    unary_sum = unary.sum(axis=-1, keepdims=True)
    return unary / unary_sum
