"""Sum-product belief propagation for pairwise discrete MRFs."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.graph import PairwiseMRF


class BeliefPropagation:
    """Sum-product Belief Propagation (BP) for a PairwiseMRF.

    Notes
    -----
    Messages are stored as ``messages[(i, j)]`` and represent distributions over
    states of node ``j``.
    """

    def __init__(self, graph: PairwiseMRF, max_iters: int = 50, tol: float = 1e-6) -> None:
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")

        self.graph = graph
        self.max_iters = int(max_iters)
        self.tol = float(tol)

        self.messages: Dict[Tuple[int, int], np.ndarray] = {}
        self.num_iters: int = 0
        self.converged: bool = False

    @staticmethod
    def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        total = float(vec.sum())
        if total <= eps or not np.isfinite(total):
            return np.full(vec.shape, 1.0 / vec.size, dtype=np.float64)
        return vec / total

    def _directed_edges(self) -> List[Tuple[int, int]]:
        directed: List[Tuple[int, int]] = []
        for i, j in sorted(self.graph.graph.edges()):
            directed.append((i, j))
            directed.append((j, i))
        return directed

    def initialize_messages(self) -> None:
        """Initialize all directed-edge messages to uniform distributions."""
        self.messages = {}
        for i, j in self._directed_edges():
            k_j = self.graph.states_per_node[j]
            self.messages[(i, j)] = np.full(k_j, 1.0 / k_j, dtype=np.float64)

    def update_message(
        self,
        i: int,
        j: int,
        message_source: Dict[Tuple[int, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        r"""Compute updated message ``m_{i->j}``.

        The update is:
            m_{i->j}(x_j) \propto sum_{x_i} psi_i(x_i) psi_{ij}(x_i, x_j)
                                  prod_{k in N(i) \ {j}} m_{k->i}(x_i)
        """
        if message_source is None:
            message_source = self.messages

        local = self.graph.unary_potentials[i].copy()

        # Multiply incoming messages to i except from j.
        for k in self.graph.neighbors(i):
            if k == j:
                continue
            local *= message_source[(k, i)]

        # Sum over x_i via matrix multiplication:
        # local shape: (K_i,), psi_ij shape: (K_i, K_j), result: (K_j,)
        pairwise = self.graph.get_pairwise_potential(i, j)
        new_message = local @ pairwise

        return self._normalize(new_message)

    def run(self) -> bool:
        """Run synchronous loopy BP updates until convergence or max iterations.

        Returns
        -------
        bool
            ``True`` if converged, ``False`` otherwise.
        """
        if not self.messages:
            self.initialize_messages()

        directed_edges = self._directed_edges()
        self.converged = False

        for iteration in range(1, self.max_iters + 1):
            old_messages = self.messages
            new_messages: Dict[Tuple[int, int], np.ndarray] = {}

            max_delta = 0.0
            for i, j in directed_edges:
                msg = self.update_message(i, j, message_source=old_messages)
                new_messages[(i, j)] = msg

                delta = float(np.max(np.abs(msg - old_messages[(i, j)])))
                if delta > max_delta:
                    max_delta = delta

            self.messages = new_messages
            self.num_iters = iteration

            if max_delta < self.tol:
                self.converged = True
                break

        return self.converged

    def compute_beliefs(self) -> Dict[int, np.ndarray]:
        """Compute normalized node beliefs (approximate marginals)."""
        if not self.messages:
            self.initialize_messages()

        beliefs: Dict[int, np.ndarray] = {}
        for i in range(self.graph.num_nodes):
            belief = self.graph.unary_potentials[i].copy()
            for k in self.graph.neighbors(i):
                belief *= self.messages[(k, i)]
            beliefs[i] = self._normalize(belief)

        return beliefs
