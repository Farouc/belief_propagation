"""Tree-Reweighted Belief Propagation (TRW-BP) for pairwise discrete MRFs."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.graph import PairwiseMRF


class TreeReweightedBeliefPropagation:
    """Sum-product Tree-Reweighted Belief Propagation.

    Parameters
    ----------
    graph:
        Pairwise MRF on which to run inference.
    rho:
        Optional dictionary of edge appearance probabilities. Keys can be
        directed ``(i, j)`` and/or reverse ``(j, i)``. Missing entries default
        to ``0.5``.
    max_iters:
        Maximum number of synchronous updates.
    tol:
        Convergence threshold on max message change.
    damping:
        Message damping coefficient in ``(0, 1]``.

    Notes
    -----
    Messages are stored as ``messages[(i, j)]`` and represent distributions over
    states of node ``j``.
    """

    def __init__(
        self,
        graph: PairwiseMRF,
        rho: Dict[Tuple[int, int], float] | None = None,
        max_iters: int = 50,
        tol: float = 1e-3,
        damping: float = 0.5,
    ) -> None:
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if not (0.0 < damping <= 1.0):
            raise ValueError("damping must be in (0, 1]")

        self.graph = graph
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.damping = float(damping)

        self.messages: Dict[Tuple[int, int], np.ndarray] = {}
        self.num_iters: int = 0
        self.converged: bool = False
        self.message_deltas: List[float] = []

        self.rho = self._initialize_rho(rho)

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

    def _initialize_rho(
        self,
        rho: Dict[Tuple[int, int], float] | None,
    ) -> Dict[Tuple[int, int], float]:
        """Initialize directed edge appearance probabilities."""
        rho_map: Dict[Tuple[int, int], float] = {}
        user_rho = rho or {}

        for i, j in self._directed_edges():
            if (i, j) in user_rho:
                value = float(user_rho[(i, j)])
            elif (j, i) in user_rho:
                value = float(user_rho[(j, i)])
            else:
                value = 0.5

            if value < 0.0:
                raise ValueError(f"rho[{(i, j)}] must be non-negative")

            rho_map[(i, j)] = value

        return rho_map

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
        r"""Compute the TRW message update ``m_{i->j}``.

        m_{i->j}(x_j) = sum_{x_i} psi_i(x_i) psi_{ij}(x_i, x_j)^{rho_ij}
                        prod_{k in N(i)\{j}} m_{k->i}(x_i)^{rho_ki}
        """
        if message_source is None:
            message_source = self.messages

        local = self.graph.unary_potentials[i].copy()

        # Incoming messages are weighted by edge appearance probabilities.
        for k in self.graph.neighbors(i):
            if k == j:
                continue
            incoming = np.clip(message_source[(k, i)], 0.0, None)
            local *= np.power(incoming, self.rho[(k, i)])

        pairwise = np.clip(self.graph.get_pairwise_potential(i, j), 0.0, None)
        pairwise_weighted = np.power(pairwise, self.rho[(i, j)])
        raw_message = local @ pairwise_weighted

        return self._normalize(raw_message)

    def run(self) -> bool:
        """Run synchronous TRW updates until convergence or max iterations."""
        if not self.messages:
            self.initialize_messages()

        directed_edges = self._directed_edges()
        self.converged = False
        self.message_deltas = []

        for iteration in range(1, self.max_iters + 1):
            old_messages = self.messages
            new_messages: Dict[Tuple[int, int], np.ndarray] = {}

            max_delta = 0.0
            for i, j in directed_edges:
                updated = self.update_message(i, j, message_source=old_messages)
                old = old_messages[(i, j)]

                damped = (1.0 - self.damping) * old + self.damping * updated
                damped = self._normalize(damped)

                new_messages[(i, j)] = damped

                delta = float(np.max(np.abs(damped - old)))
                if delta > max_delta:
                    max_delta = delta

            self.messages = new_messages
            self.num_iters = iteration
            self.message_deltas.append(max_delta)

            if max_delta < self.tol:
                self.converged = True
                break

        return self.converged

    def compute_beliefs(self) -> Dict[int, np.ndarray]:
        """Compute normalized node beliefs with TRW reweighting."""
        if not self.messages:
            self.initialize_messages()

        beliefs: Dict[int, np.ndarray] = {}
        for i in range(self.graph.num_nodes):
            belief = self.graph.unary_potentials[i].copy()
            for k in self.graph.neighbors(i):
                incoming = np.clip(self.messages[(k, i)], 0.0, None)
                belief *= np.power(incoming, self.rho[(k, i)])
            beliefs[i] = self._normalize(belief)

        return beliefs
