"""EM-based unary modeling for binary image segmentation MRFs.

This module adds an unsupervised unary potential builder that fits a
2-component Gaussian mixture (foreground/background proxy) on pixel colors.
It is intentionally independent from the existing GT-based unary path.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.graph import PairwiseMRF
from utils.segmentation_utils import build_grid_graph, compute_pairwise_potential


def _to_float_image(image: np.ndarray) -> np.ndarray:
    """Convert image to float64 in [0, 1], keeping channels when available."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def _feature_matrix(image: np.ndarray) -> np.ndarray:
    """Return flattened per-pixel features with shape (num_pixels, channels)."""
    img = _to_float_image(image)
    if img.ndim == 2:
        return img.reshape(-1, 1)
    return img.reshape(-1, img.shape[-1])


def _log_gaussian_pdf(features: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Evaluate log N(x | mean, cov) for each row in features."""
    dim = features.shape[1]
    centered = features - mean

    try:
        solve = np.linalg.solve(cov, centered.T).T
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("non-positive determinant")
    except np.linalg.LinAlgError:
        cov = cov + 1e-3 * np.eye(cov.shape[0], dtype=np.float64)
        solve = np.linalg.solve(cov, centered.T).T
        _, logdet = np.linalg.slogdet(cov)

    mahal = np.sum(centered * solve, axis=1)
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahal)


def _weighted_mean_and_cov(
    features: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute weighted mean/covariance with covariance regularization."""
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-12:
        # Fallback to broad isotropic model if a component collapses.
        mean = np.mean(features, axis=0)
        cov = np.eye(features.shape[1], dtype=np.float64) * 0.08
        return mean, cov

    normalized = weights / total_weight
    mean = np.sum(features * normalized[:, None], axis=0)
    centered = features - mean
    cov = (centered * normalized[:, None]).T @ centered
    cov += regularization * np.eye(cov.shape[0], dtype=np.float64)
    return mean, cov


def fit_two_gaussian_em(
    features: np.ndarray,
    max_iters: int = 60,
    tol: float = 1e-4,
    regularization: float = 1e-4,
    seed: int = 0,
) -> Dict[str, np.ndarray | float | int]:
    """Fit a 2-component Gaussian mixture with EM.

    EM equations:
      gamma_nk = pi_k N(x_n | mu_k, Sigma_k) / sum_j pi_j N(x_n | mu_j, Sigma_j)
      N_k = sum_n gamma_nk
      pi_k = N_k / N
      mu_k = (1/N_k) sum_n gamma_nk x_n
      Sigma_k = (1/N_k) sum_n gamma_nk (x_n-mu_k)(x_n-mu_k)^T + eps*I
    """
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("features must have shape (N, D) with N >= 2")

    n_samples, dim = x.shape
    rng = np.random.default_rng(seed)

    # Deterministic intensity-based warm start (works for gray and RGB).
    gray = np.mean(x, axis=1)
    threshold = float(np.median(gray))
    hard_assign = (gray >= threshold).astype(np.int32)

    # Ensure both clusters are non-empty.
    if np.all(hard_assign == 0) or np.all(hard_assign == 1):
        hard_assign = np.zeros(n_samples, dtype=np.int32)
        hard_assign[rng.choice(n_samples, size=n_samples // 2, replace=False)] = 1

    resp = np.zeros((n_samples, 2), dtype=np.float64)
    resp[np.arange(n_samples), hard_assign] = 1.0

    means = np.zeros((2, dim), dtype=np.float64)
    covs = np.zeros((2, dim, dim), dtype=np.float64)
    mix = np.zeros(2, dtype=np.float64)

    for k in range(2):
        weights = resp[:, k]
        means[k], covs[k] = _weighted_mean_and_cov(
            x,
            weights=weights,
            regularization=regularization,
        )
        mix[k] = max(float(np.sum(weights)) / n_samples, 1e-6)

    mix /= np.sum(mix)
    log_likelihoods = []

    for iteration in range(1, max_iters + 1):
        log_prob = np.zeros((n_samples, 2), dtype=np.float64)
        for k in range(2):
            log_prob[:, k] = np.log(mix[k] + 1e-12) + _log_gaussian_pdf(x, means[k], covs[k])

        # Stable log-sum-exp normalization.
        max_log = np.max(log_prob, axis=1, keepdims=True)
        shifted = log_prob - max_log
        log_norm = max_log + np.log(np.sum(np.exp(shifted), axis=1, keepdims=True) + 1e-12)
        resp = np.exp(log_prob - log_norm)

        current_ll = float(np.sum(log_norm))
        log_likelihoods.append(current_ll)

        # M-step
        nk = np.sum(resp, axis=0) + 1e-12
        mix = nk / np.sum(nk)
        for k in range(2):
            means[k], covs[k] = _weighted_mean_and_cov(
                x,
                weights=resp[:, k],
                regularization=regularization,
            )

        if len(log_likelihoods) >= 2:
            improvement = log_likelihoods[-1] - log_likelihoods[-2]
            if improvement < tol:
                break

    return {
        "means": means,
        "covariances": covs,
        "mixing_weights": mix,
        "responsibilities": resp,
        "iterations": iteration,
        "log_likelihood": log_likelihoods[-1],
        "log_likelihood_history": np.asarray(log_likelihoods, dtype=np.float64),
    }


def compute_unary_potentials_em(
    image: np.ndarray,
    max_iters: int = 60,
    tol: float = 1e-4,
    regularization: float = 1e-4,
    seed: int = 0,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, np.ndarray | float | int]]:
    """Compute binary unary potentials with unsupervised Gaussian EM.

    Component-to-label assignment:
      - foreground component = component with larger mean intensity.
      - background component = the other component.
    """
    features = _feature_matrix(image)
    em_result = fit_two_gaussian_em(
        features,
        max_iters=max_iters,
        tol=tol,
        regularization=regularization,
        seed=seed,
    )
    resp = np.asarray(em_result["responsibilities"], dtype=np.float64)
    means = np.asarray(em_result["means"], dtype=np.float64)

    # Choose foreground as brighter Gaussian component.
    fg_component = int(np.argmax(np.mean(means, axis=1)))
    bg_component = 1 - fg_component

    unary = np.stack([resp[:, bg_component], resp[:, fg_component]], axis=1) + eps
    unary /= unary.sum(axis=1, keepdims=True)

    info: Dict[str, np.ndarray | float | int] = dict(em_result)
    info["bg_component"] = bg_component
    info["fg_component"] = fg_component
    return unary.astype(np.float64), info


def _compute_beta(image: np.ndarray) -> float:
    """Compute beta = 1 / (2 * mean(||Ii - Ij||^2)) over 4-neighborhood pairs."""
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


def build_binary_segmentation_mrf_em(
    image: np.ndarray,
    lambda_: float = 2.0,
    em_max_iters: int = 60,
    em_tol: float = 1e-4,
    em_regularization: float = 1e-4,
    em_seed: int = 0,
) -> Tuple[PairwiseMRF, Dict[str, np.ndarray | float | int]]:
    """Build binary segmentation MRF where unary potentials come from EM."""
    height, width = image.shape[:2]
    num_nodes = height * width

    unary, em_info = compute_unary_potentials_em(
        image=image,
        max_iters=em_max_iters,
        tol=em_tol,
        regularization=em_regularization,
        seed=em_seed,
    )

    mrf = PairwiseMRF(num_nodes=num_nodes, states_per_node=2)
    grid = build_grid_graph(image)

    img = _to_float_image(image)
    if img.ndim == 2:
        img = img[..., None]
    flat_pixels = img.reshape(-1, img.shape[-1])
    beta = _compute_beta(image)

    for i, j in grid.edges():
        mrf.add_edge(i, j)

    for node in range(num_nodes):
        mrf.set_unary_potential(node, unary[node])

    for i, j in grid.edges():
        pairwise = compute_pairwise_potential(
            pixel_i=flat_pixels[i],
            pixel_j=flat_pixels[j],
            beta=beta,
            lambda_=lambda_,
        )
        mrf.set_pairwise_potential(i, j, pairwise)

    return mrf, em_info

