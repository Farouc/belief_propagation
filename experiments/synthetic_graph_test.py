"""Synthetic chain-graph validation for sum-product belief propagation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running as: python experiments/synthetic_graph_test.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.belief_propagation import BeliefPropagation
from src.graph import PairwiseMRF
from src.potentials import potts_potential


def build_chain_mrf() -> PairwiseMRF:
    """Create a 4-node binary chain X1-X2-X3-X4."""
    mrf = PairwiseMRF(num_nodes=4, states_per_node=2)

    edges = [(0, 1), (1, 2), (2, 3)]
    for i, j in edges:
        mrf.add_edge(i, j)

    # Slightly informative unary potentials at the ends.
    mrf.set_unary_potential(0, np.array([0.90, 0.10]))
    mrf.set_unary_potential(1, np.array([0.50, 0.50]))
    mrf.set_unary_potential(2, np.array([0.50, 0.50]))
    mrf.set_unary_potential(3, np.array([0.10, 0.90]))

    # Potts potential encourages equal neighboring states.
    pairwise = potts_potential(num_states=2, same_weight=3.0, diff_weight=1.0)
    for i, j in edges:
        mrf.set_pairwise_potential(i, j, pairwise)

    return mrf


def check_normalization(bp: BeliefPropagation, beliefs: dict[int, np.ndarray]) -> None:
    """Small sanity checks for message and belief normalization."""
    for (i, j), msg in bp.messages.items():
        if not np.isclose(msg.sum(), 1.0, atol=1e-9):
            raise AssertionError(f"Message ({i}->{j}) does not sum to 1: {msg.sum()}")

    for i, belief in beliefs.items():
        if not np.isclose(belief.sum(), 1.0, atol=1e-9):
            raise AssertionError(f"Belief at node {i} does not sum to 1: {belief.sum()}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    mrf = build_chain_mrf()
    bp = BeliefPropagation(mrf, max_iters=100, tol=1e-10)

    bp.initialize_messages()
    converged = bp.run()
    beliefs = bp.compute_beliefs()
    check_normalization(bp, beliefs)

    print("Synthetic chain marginals:")
    for i in range(mrf.num_nodes):
        b = beliefs[i]
        print(f"Node X{i + 1}: P(0)={b[0]:.4f}, P(1)={b[1]:.4f}")

    print(f"Converged: {converged} in {bp.num_iters} iterations")

    # Plot marginals for visual inspection.
    node_ids = np.arange(1, mrf.num_nodes + 1)
    p0 = np.array([beliefs[i][0] for i in range(mrf.num_nodes)])
    p1 = np.array([beliefs[i][1] for i in range(mrf.num_nodes)])

    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(node_ids - width / 2, p0, width=width, label="P(x=0)", color="#4E79A7")
    ax.bar(node_ids + width / 2, p1, width=width, label="P(x=1)", color="#F28E2B")
    ax.set_xticks(node_ids)
    ax.set_xlabel("Node")
    ax.set_ylabel("Probability")
    ax.set_title("Beliefs on Chain Graph")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = PROJECT_ROOT / "synthetic_marginals.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Non-interactive backend '{backend}': saved figure to {output_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
