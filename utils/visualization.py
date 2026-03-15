"""Visualization helpers for MRF structure and segmentation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.graph import PairwiseMRF


def _show_or_save(fig: plt.Figure, default_filename: str, save_path: str | None = None) -> None:
    """Show figure if backend is interactive, otherwise save it to disk.

    Parameters
    ----------
    fig:
        Matplotlib figure.
    default_filename:
        Fallback filename used in non-interactive mode.
    save_path:
        Optional explicit output path.
    """
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output = Path(save_path) if save_path is not None else Path(default_filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Non-interactive backend '{backend}': saved figure to {output.resolve()}")
        plt.close(fig)
        return

    plt.show()


def plot_graph(graph: PairwiseMRF, save_path: str | None = None) -> None:
    """Plot the MRF graph topology."""
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = nx.spring_layout(graph.graph, seed=0)
    nx.draw_networkx(
        graph.graph,
        pos=pos,
        with_labels=True,
        node_size=700,
        node_color="#A8D0E6",
        edge_color="#374785",
        ax=ax,
    )
    ax.set_title("Pairwise MRF Graph")
    ax.axis("off")
    plt.tight_layout()
    _show_or_save(fig, default_filename="graph.png", save_path=save_path)


def plot_segmentation(
    original: np.ndarray, labels: np.ndarray, save_path: str | None = None
) -> None:
    """Display original grayscale image and discrete segmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(labels, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("BP Segmentation")
    axes[1].axis("off")

    plt.tight_layout()
    _show_or_save(fig, default_filename="segmentation_result.png", save_path=save_path)
