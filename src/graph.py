"""Graph data structures for pairwise discrete Markov Random Fields."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import networkx as nx
import numpy as np


class PairwiseMRF:
    """Pairwise Markov Random Field with discrete states.

    Parameters
    ----------
    num_nodes:
        Number of variables in the model.
    states_per_node:
        Either one integer ``K`` (same number of states for all nodes)
        or a sequence of length ``num_nodes``.

    Notes
    -----
    Unary potentials are stored as ``psi_i(x_i)`` and pairwise potentials as
    ``psi_ij(x_i, x_j)``.
    """

    def __init__(
        self, num_nodes: int, states_per_node: Union[int, Sequence[int]]
    ) -> None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")

        self.num_nodes = int(num_nodes)
        self.states_per_node = self._parse_states_per_node(states_per_node)

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))

        # unary_potentials[i] has shape (K_i,)
        self.unary_potentials: Dict[int, np.ndarray] = {
            i: np.ones(self.states_per_node[i], dtype=np.float64)
            for i in range(self.num_nodes)
        }

        # pairwise_potentials[(i, j)] has shape (K_i, K_j)
        # and pairwise_potentials[(j, i)] stores the transpose.
        self.pairwise_potentials: Dict[Tuple[int, int], np.ndarray] = {}

    def _parse_states_per_node(
        self, states_per_node: Union[int, Sequence[int]]
    ) -> List[int]:
        if isinstance(states_per_node, int):
            if states_per_node <= 0:
                raise ValueError("states_per_node must be positive")
            return [states_per_node] * self.num_nodes

        states = list(states_per_node)
        if len(states) != self.num_nodes:
            raise ValueError("states_per_node sequence must have length num_nodes")
        if any(k <= 0 for k in states):
            raise ValueError("all state counts must be positive")
        return [int(k) for k in states]

    def _validate_node(self, i: int) -> None:
        if i < 0 or i >= self.num_nodes:
            raise IndexError(f"node index out of range: {i}")

    def add_edge(self, i: int, j: int) -> None:
        """Add an undirected edge between nodes ``i`` and ``j``."""
        self._validate_node(i)
        self._validate_node(j)
        if i == j:
            raise ValueError("self-loops are not allowed in PairwiseMRF")
        self.graph.add_edge(i, j)

    def set_unary_potential(self, i: int, potential_vector: np.ndarray) -> None:
        """Set unary potential ``psi_i`` for node ``i``.

        ``potential_vector`` must have shape ``(K_i,)`` and non-negative entries.
        """
        self._validate_node(i)

        potential = np.asarray(potential_vector, dtype=np.float64).reshape(-1)
        expected_shape = (self.states_per_node[i],)
        if potential.shape != expected_shape:
            raise ValueError(
                f"unary potential for node {i} must have shape {expected_shape}, "
                f"got {potential.shape}"
            )
        if np.any(potential < 0):
            raise ValueError("unary potentials must be non-negative")

        self.unary_potentials[i] = potential.copy()

    def set_pairwise_potential(
        self, i: int, j: int, potential_matrix: np.ndarray
    ) -> None:
        """Set pairwise potential ``psi_ij`` for edge ``(i, j)``.

        ``potential_matrix`` must have shape ``(K_i, K_j)`` and non-negative entries.
        The reverse direction ``(j, i)`` is stored as the transpose.
        """
        self._validate_node(i)
        self._validate_node(j)
        if not self.graph.has_edge(i, j):
            raise ValueError(f"edge ({i}, {j}) does not exist")

        potential = np.asarray(potential_matrix, dtype=np.float64)
        expected_shape = (self.states_per_node[i], self.states_per_node[j])
        if potential.shape != expected_shape:
            raise ValueError(
                f"pairwise potential for edge ({i}, {j}) must have shape "
                f"{expected_shape}, got {potential.shape}"
            )
        if np.any(potential < 0):
            raise ValueError("pairwise potentials must be non-negative")

        self.pairwise_potentials[(i, j)] = potential.copy()
        self.pairwise_potentials[(j, i)] = potential.T.copy()

    def get_pairwise_potential(self, i: int, j: int) -> np.ndarray:
        """Return pairwise potential matrix ``psi_ij`` with shape ``(K_i, K_j)``."""
        key = (i, j)
        if key not in self.pairwise_potentials:
            raise KeyError(f"pairwise potential for edge ({i}, {j}) is not set")
        return self.pairwise_potentials[key]

    def neighbors(self, i: int) -> List[int]:
        """Return sorted neighbors of node ``i`` for deterministic iteration order."""
        self._validate_node(i)
        return sorted(self.graph.neighbors(i))
