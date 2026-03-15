"""Helper functions to build image segmentation MRFs for belief propagation."""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
from skimage import color

from src.graph import PairwiseMRF


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale float64 in [0, 1]."""
    img = np.asarray(image)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


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
    foreground_mean: float = 0.65,
    background_mean: float = 0.35,
    sigma: float = 0.18,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute binary unary potentials from image intensity.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_pixels, 2)`` where column 0 is background and
        column 1 is foreground potential.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    gray = _to_grayscale_float(image)
    coeff = -0.5 / (sigma * sigma)

    bg = np.exp(coeff * (gray - background_mean) ** 2)
    fg = np.exp(coeff * (gray - foreground_mean) ** 2)

    unary = np.stack([bg, fg], axis=-1) + eps
    unary /= unary.sum(axis=-1, keepdims=True)
    return unary.reshape(-1, 2)


def compute_pairwise_potentials(
    image: np.ndarray,
    smoothness: float = 2.0,
    min_same_label_weight: float = 1.1,
) -> np.ndarray:
    """Compute a 2x2 Potts-style pairwise potential matrix.

    A stronger smoothness is used for lower average contrast.
    """
    if smoothness < 0:
        raise ValueError("smoothness must be non-negative")

    gray = _to_grayscale_float(image)
    diff_h = np.abs(np.diff(gray, axis=1))
    diff_v = np.abs(np.diff(gray, axis=0))
    local_contrast = float(np.mean(np.concatenate([diff_h.ravel(), diff_v.ravel()])))

    same_weight = min_same_label_weight + smoothness * np.exp(-5.0 * local_contrast)
    same_weight = float(max(same_weight, min_same_label_weight))

    pairwise = np.array(
        [[same_weight, 1.0], [1.0, same_weight]],
        dtype=np.float64,
    )
    return pairwise


def build_binary_segmentation_mrf(
    image: np.ndarray,
    smoothness: float = 2.0,
) -> PairwiseMRF:
    """Build a binary PairwiseMRF from an image for segmentation."""
    height, width = image.shape[:2]
    num_nodes = height * width

    mrf = PairwiseMRF(num_nodes=num_nodes, states_per_node=2)
    graph = build_grid_graph(image)
    unary = compute_unary_potentials(image)
    pairwise = compute_pairwise_potentials(image, smoothness=smoothness)

    for i, j in graph.edges():
        mrf.add_edge(i, j)

    for node in range(num_nodes):
        mrf.set_unary_potential(node, unary[node])

    for i, j in graph.edges():
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
