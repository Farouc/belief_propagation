"""Utility helpers for Tree-Reweighted Belief Propagation."""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx


def compute_uniform_edge_weights(graph: object) -> Dict[Tuple[int, int], float]:
    """Return uniform TRW edge appearance probabilities.

    Parameters
    ----------
    graph:
        Either a NetworkX graph or an object exposing a ``graph`` attribute
        containing a NetworkX graph (e.g. ``PairwiseMRF``).

    Returns
    -------
    dict
        Directed-edge mapping with ``rho[(i, j)] = rho[(j, i)] = 0.5`` for each
        undirected edge ``(i, j)``.
    """
    nx_graph = graph.graph if hasattr(graph, "graph") else graph
    if not isinstance(nx_graph, nx.Graph):
        raise TypeError("graph must be a networkx.Graph or expose .graph")

    rho: Dict[Tuple[int, int], float] = {}
    for i, j in sorted(nx_graph.edges()):
        rho[(i, j)] = 0.5
        rho[(j, i)] = 0.5

    return rho
