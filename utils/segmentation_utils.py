"""Helper functions to build image segmentation MRFs for belief propagation."""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
from skimage import color

from src.graph import PairwiseMRF


def _to_float_image(image: np.ndarray) -> np.ndarray:
    """Convert image to float64 in [0, 1], preserving channels when present."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def _feature_matrix(image: np.ndarray) -> np.ndarray:
    """Return per-pixel color feature matrix of shape (num_pixels, channels)."""
    img = _to_float_image(image)
    if img.ndim == 2:
        return img.reshape(-1, 1)
    return img.reshape(-1, img.shape[-1])


def _estimate_gaussian(features: np.ndarray, regularization: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate mean/covariance with diagonal regularization."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be non-empty with shape (N, D)")

    mean = np.mean(features, axis=0)
    centered = features - mean
    cov = (centered.T @ centered) / max(features.shape[0] - 1, 1)

    # Keep covariance positive-definite for stable likelihood computations.
    cov = cov + regularization * np.eye(cov.shape[0], dtype=np.float64)
    return mean, cov


def _gaussian_likelihood(features: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Evaluate multivariate Gaussian likelihood for each feature vector."""
    dim = features.shape[1]
    centered = features - mean

    try:
        solve = np.linalg.solve(cov, centered.T).T
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("non-positive determinant")
    except np.linalg.LinAlgError:
        cov_fallback = cov + 1e-3 * np.eye(cov.shape[0], dtype=np.float64)
        solve = np.linalg.solve(cov_fallback, centered.T).T
        _, logdet = np.linalg.slogdet(cov_fallback)

    mahal = np.sum(centered * solve, axis=1)
    log_prob = -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahal)

    # Stabilize before exponentiating.
    log_prob -= np.max(log_prob)
    return np.exp(log_prob)


def build_grid_graph(image: np.ndarray) -> nx.Graph:
    """Build a 4-neighborhood grid graph for an image.

    Nodes are integer pixel indices in row-major order: ``node = r * W + c``.
    """
    height, width = image.shape[:2]
    grid = nx.grid_2d_graph(height, width)
    mapping = {(r, c): r * width + c for r in range(height) for c in range(width)}
    return nx.relabel_nodes(grid, mapping)


def compute_unary_potentials(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute binary unary potentials using Gaussian color models.

    Parameters
    ----------
    image:
        Grayscale or RGB image.
    mask:
        Optional binary mask used to estimate foreground/background Gaussians.
        If ``None``, a pseudo-mask is estimated from image intensity median.

    Returns
    -------
    np.ndarray
        Unary potentials with shape ``(num_pixels, 2)`` where columns are
        ``[background, foreground]``.
    """
    features = _feature_matrix(image)
    num_pixels = features.shape[0]

    if mask is None:
        img = _to_float_image(image)
        gray = color.rgb2gray(img) if img.ndim == 3 else img
        pseudo_fg = (gray >= np.median(gray)).astype(np.uint8)
        flat_mask = pseudo_fg.reshape(-1).astype(bool)
    else:
        flat_mask = np.asarray(mask).reshape(-1).astype(bool)
        if flat_mask.size != num_pixels:
            raise ValueError("mask size does not match image size")

    fg_features = features[flat_mask]
    bg_features = features[~flat_mask]

    # Keep model estimation robust when one class is underrepresented.
    if fg_features.shape[0] < 2 or bg_features.shape[0] < 2:
        gray = features.mean(axis=1)
        threshold = np.quantile(gray, 0.6)
        fallback_mask = gray >= threshold
        fg_features = features[fallback_mask]
        bg_features = features[~fallback_mask]

    if fg_features.shape[0] < 2 or bg_features.shape[0] < 2:
        # Last-resort fallback to broad fixed Gaussians.
        d = features.shape[1]
        fg_mean = np.full(d, 0.7, dtype=np.float64)
        bg_mean = np.full(d, 0.3, dtype=np.float64)
        fg_cov = np.eye(d, dtype=np.float64) * 0.08
        bg_cov = np.eye(d, dtype=np.float64) * 0.08
    else:
        fg_mean, fg_cov = _estimate_gaussian(fg_features)
        bg_mean, bg_cov = _estimate_gaussian(bg_features)

    prior_fg = max(float(fg_features.shape[0]) / float(num_pixels), 1e-3)
    prior_bg = max(float(bg_features.shape[0]) / float(num_pixels), 1e-3)

    fg_like = _gaussian_likelihood(features, fg_mean, fg_cov) * prior_fg
    bg_like = _gaussian_likelihood(features, bg_mean, bg_cov) * prior_bg

    unary = np.stack([bg_like, fg_like], axis=1) + eps
    unary /= unary.sum(axis=1, keepdims=True)
    return unary.astype(np.float64)


def compute_pairwise_potential(
    pixel_i: np.ndarray,
    pixel_j: np.ndarray,
    beta: float,
    lambda_: float,
) -> np.ndarray:
    """Compute contrast-sensitive Potts pairwise potential for one edge."""
    diff = float(np.sum((pixel_i - pixel_j) ** 2))
    weight = np.exp(-beta * diff)
    off_diag = np.exp(-lambda_ * weight)
    return np.array([[1.0, off_diag], [off_diag, 1.0]], dtype=np.float64)


def _compute_beta(image: np.ndarray) -> float:
    """Compute contrast parameter beta = 1 / (2 * mean(||Ii-Ij||^2))."""
    img = _to_float_image(image)
    if img.ndim == 2:
        img = img[..., None]

    diff_h = img[:, 1:, :] - img[:, :-1, :]
    diff_v = img[1:, :, :] - img[:-1, :, :]

    sq_h = np.sum(diff_h**2, axis=-1).ravel()
    sq_v = np.sum(diff_v**2, axis=-1).ravel()
    diffs = np.concatenate([sq_h, sq_v])

    mean_diff = float(np.mean(diffs)) if diffs.size > 0 else 1.0
    mean_diff = max(mean_diff, 1e-8)
    return 1.0 / (2.0 * mean_diff)


def build_binary_segmentation_mrf(
    image: np.ndarray,
    mask_for_unary: np.ndarray | None = None,
    lambda_: float = 2.0,
) -> PairwiseMRF:
    """Build a binary segmentation MRF with contrast-sensitive smoothing.

    Parameters
    ----------
    image:
        Input image used for unary and pairwise modeling.
    mask_for_unary:
        Optional binary mask used only to estimate Gaussian unary parameters.
        This acts like a supervised color model estimation step.
    lambda_:
        Smoothness strength in the contrast-sensitive pairwise model.
    """
    height, width = image.shape[:2]
    num_nodes = height * width

    mrf = PairwiseMRF(num_nodes=num_nodes, states_per_node=2)
    graph = build_grid_graph(image)
    unary = compute_unary_potentials(image=image, mask=mask_for_unary)

    img = _to_float_image(image)
    if img.ndim == 2:
        img = img[..., None]
    flat_pixels = img.reshape(-1, img.shape[-1])

    beta = _compute_beta(image)

    for i, j in graph.edges():
        mrf.add_edge(i, j)

    for node in range(num_nodes):
        mrf.set_unary_potential(node, unary[node])

    for i, j in graph.edges():
        pairwise = compute_pairwise_potential(
            pixel_i=flat_pixels[i],
            pixel_j=flat_pixels[j],
            beta=beta,
            lambda_=lambda_,
        )
        mrf.set_pairwise_potential(i, j, pairwise)

    return mrf


def beliefs_to_mask(
    beliefs: Dict[int, np.ndarray], image_shape: Tuple[int, int]
) -> np.ndarray:
    """Convert node beliefs to MAP binary mask."""
    height, width = image_shape
    labels = np.zeros(height * width, dtype=np.uint8)
    for node, belief in beliefs.items():
        labels[node] = np.uint8(np.argmax(belief))
    return labels.reshape(height, width)
