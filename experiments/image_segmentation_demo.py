"""Image segmentation demo with loopy belief propagation on a grid MRF."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage import data, transform

# Allow running as: python experiments/image_segmentation_demo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.belief_propagation import BeliefPropagation
from src.graph import PairwiseMRF
from src.potentials import binary_unary_from_intensity, potts_potential
from utils.visualization import plot_segmentation


def pixel_to_node(r: int, c: int, width: int) -> int:
    """Convert pixel coordinates to flat node index."""
    return r * width + c


def build_grid_mrf(image: np.ndarray) -> PairwiseMRF:
    """Create a 4-neighbor grid MRF for binary segmentation."""
    height, width = image.shape
    mrf = PairwiseMRF(num_nodes=height * width, states_per_node=2)

    unary = binary_unary_from_intensity(image, means=(0.35, 0.65), sigma=0.18)

    # Add unary terms and 4-neighborhood edges.
    for r in range(height):
        for c in range(width):
            i = pixel_to_node(r, c, width)
            mrf.set_unary_potential(i, unary[r, c])

            if r + 1 < height:
                j = pixel_to_node(r + 1, c, width)
                mrf.add_edge(i, j)
            if c + 1 < width:
                j = pixel_to_node(r, c + 1, width)
                mrf.add_edge(i, j)

    # Pairwise smoothing: same labels are favored.
    pairwise = potts_potential(num_states=2, same_weight=2.2, diff_weight=1.0)
    for i, j in mrf.graph.edges():
        mrf.set_pairwise_potential(i, j, pairwise)

    return mrf


def beliefs_to_labels(
    beliefs: dict[int, np.ndarray], image_shape: tuple[int, int]
) -> np.ndarray:
    """Convert node beliefs to MAP labels arranged as an image."""
    height, width = image_shape
    labels = np.zeros(height * width, dtype=np.int32)
    for idx, belief in beliefs.items():
        labels[idx] = int(np.argmax(belief))
    return labels.reshape((height, width))


def check_beliefs_sum_to_one(beliefs: dict[int, np.ndarray]) -> None:
    """Sanity check for normalized beliefs."""
    for i, belief in beliefs.items():
        if not np.isclose(np.sum(belief), 1.0, atol=1e-8):
            raise AssertionError(f"Belief at node {i} is not normalized")


def main() -> None:
    # Deterministic preprocessing.
    image = data.camera().astype(np.float64) / 255.0
    image_small = transform.resize(image, (48, 48), anti_aliasing=True)

    mrf = build_grid_mrf(image_small)
    bp = BeliefPropagation(mrf, max_iters=30, tol=1e-4)
    converged = bp.run()
    beliefs = bp.compute_beliefs()

    check_beliefs_sum_to_one(beliefs)

    labels = beliefs_to_labels(beliefs, image_small.shape)

    foreground_ratio = float(np.mean(labels))
    print(
        f"Loopy BP finished. Converged: {converged} in {bp.num_iters} iterations. "
        f"Foreground ratio: {foreground_ratio:.3f}"
    )

    plot_segmentation(image_small, labels)


if __name__ == "__main__":
    main()
