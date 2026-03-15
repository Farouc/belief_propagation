"""Utilities for building and evaluating Sudoku graphical models."""

from __future__ import annotations

from itertools import combinations
from typing import Dict

import networkx as nx
import numpy as np

from src.graph import PairwiseMRF


def cell_to_node(row: int, col: int) -> int:
    """Map Sudoku cell coordinates to node index."""
    return row * 9 + col


def build_sudoku_graph() -> nx.Graph:
    """Build Sudoku constraint graph with 81 nodes.

    An edge connects two cells that must hold different values
    (same row, column, or 3x3 block).
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(81))

    # Row constraints.
    for row in range(9):
        row_nodes = [cell_to_node(row, col) for col in range(9)]
        for i, j in combinations(row_nodes, 2):
            graph.add_edge(i, j)

    # Column constraints.
    for col in range(9):
        col_nodes = [cell_to_node(row, col) for row in range(9)]
        for i, j in combinations(col_nodes, 2):
            graph.add_edge(i, j)

    # 3x3 block constraints.
    for block_row in range(3):
        for block_col in range(3):
            nodes = []
            for dr in range(3):
                for dc in range(3):
                    row = block_row * 3 + dr
                    col = block_col * 3 + dc
                    nodes.append(cell_to_node(row, col))
            for i, j in combinations(nodes, 2):
                graph.add_edge(i, j)

    return graph


def build_sudoku_mrf(puzzle: np.ndarray, unary_scale: float = 1e24) -> PairwiseMRF:
    """Build a Sudoku PairwiseMRF with 9 states per node.

    Unary potentials:
      - given clue: one-hot
      - empty cell: uniform

    Pairwise potentials enforce inequality:
      psi(x_i, x_j) = 0 if x_i == x_j else 1

    Notes
    -----
    ``unary_scale`` preserves relative probabilities while keeping message values
    away from tiny magnitudes in high-degree Sudoku nodes.
    """
    puzzle_arr = np.asarray(puzzle, dtype=np.int32)
    if puzzle_arr.shape != (9, 9):
        raise ValueError(f"Expected puzzle shape (9, 9), got {puzzle_arr.shape}")
    if unary_scale <= 0:
        raise ValueError("unary_scale must be positive")

    mrf = PairwiseMRF(num_nodes=81, states_per_node=9)
    graph = build_sudoku_graph()

    for i, j in graph.edges():
        mrf.add_edge(i, j)

    uniform_unary = np.full(9, unary_scale, dtype=np.float64)
    for row in range(9):
        for col in range(9):
            node = cell_to_node(row, col)
            value = int(puzzle_arr[row, col])

            if 1 <= value <= 9:
                unary = np.zeros(9, dtype=np.float64)
                unary[value - 1] = unary_scale
            else:
                unary = uniform_unary.copy()

            mrf.set_unary_potential(node, unary)

    pairwise = np.ones((9, 9), dtype=np.float64)
    np.fill_diagonal(pairwise, 0.0)
    for i, j in graph.edges():
        mrf.set_pairwise_potential(i, j, pairwise)

    return mrf


def beliefs_to_grid(beliefs: Dict[int, np.ndarray]) -> np.ndarray:
    """Convert BP beliefs to a 9x9 Sudoku grid using MAP per cell."""
    grid = np.zeros(81, dtype=np.int32)
    for node in range(81):
        grid[node] = int(np.argmax(beliefs[node])) + 1
    return grid.reshape(9, 9)


def enforce_clues(predicted_grid: np.ndarray, puzzle: np.ndarray) -> np.ndarray:
    """Keep clue cells fixed in final prediction."""
    pred = np.asarray(predicted_grid, dtype=np.int32).copy()
    clues = np.asarray(puzzle, dtype=np.int32)
    pred[clues > 0] = clues[clues > 0]
    return pred


def is_valid_sudoku(grid: np.ndarray) -> bool:
    """Check whether a grid is a valid completed Sudoku solution."""
    arr = np.asarray(grid, dtype=np.int32)
    if arr.shape != (9, 9):
        return False
    if np.any((arr < 1) | (arr > 9)):
        return False

    required = set(range(1, 10))

    for i in range(9):
        if set(arr[i, :].tolist()) != required:
            return False
        if set(arr[:, i].tolist()) != required:
            return False

    for block_row in range(3):
        for block_col in range(3):
            block = arr[
                block_row * 3 : (block_row + 1) * 3,
                block_col * 3 : (block_col + 1) * 3,
            ]
            if set(block.ravel().tolist()) != required:
                return False

    return True


def format_grid(grid: np.ndarray) -> str:
    """Pretty string formatter for Sudoku grids."""
    arr = np.asarray(grid)
    lines = []
    for row in range(9):
        values = " ".join(str(int(v)) for v in arr[row])
        lines.append(values)
    return "\n".join(lines)
