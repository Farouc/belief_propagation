"""Sudoku solving experiment using belief propagation on a constraint MRF."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Allow running as: python experiments/sudoku_bp_solver.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.sudoku_loader import load_sudoku_dataset
from src.belief_propagation import BeliefPropagation
from utils.sudoku_utils import (
    beliefs_to_grid,
    build_sudoku_mrf,
    enforce_clues,
    format_grid,
    is_valid_sudoku,
)


def _save_convergence_plot(rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save iterations per puzzle and convergence status."""
    indices = [int(row["index"]) for row in rows]
    iterations = [int(row["iterations"]) for row in rows]
    converged = [bool(row["converged"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2E8B57" if c else "#C44E52" for c in converged]
    ax.bar(indices, iterations, color=colors)
    ax.set_xlabel("Puzzle Index")
    ax.set_ylabel("BP Iterations")
    ax.set_title("Sudoku BP Convergence Behavior")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_example(
    puzzle: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    solved: bool,
    converged: bool,
    runtime_sec: float,
) -> None:
    """Save one puzzle/prediction text report."""
    text = [
        f"Solved: {solved}",
        f"Converged: {converged}",
        f"Runtime (s): {runtime_sec:.4f}",
        "",
        "Puzzle:",
        format_grid(puzzle),
        "",
        "Prediction:",
        format_grid(prediction),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    """Run BP Sudoku solving experiment and aggregate metrics."""
    puzzles = load_sudoku_dataset(n_samples=args.n_samples)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = results_dir / "examples"
    plots_dir = results_dir / "plots"
    examples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []

    for idx, puzzle in enumerate(puzzles):
        t0 = time.perf_counter()

        mrf = build_sudoku_mrf(puzzle)
        bp = BeliefPropagation(mrf, max_iters=args.max_iters, tol=args.tol)
        converged = bp.run()
        beliefs = bp.compute_beliefs()

        predicted_grid = beliefs_to_grid(beliefs)
        predicted_grid = enforce_clues(predicted_grid, puzzle)

        runtime_sec = time.perf_counter() - t0
        clues_consistent = bool(np.all(predicted_grid[puzzle > 0] == puzzle[puzzle > 0]))
        solved = clues_consistent and is_valid_sudoku(predicted_grid)

        row = {
            "index": idx,
            "solved": float(solved),
            "clues_consistent": float(clues_consistent),
            "converged": float(converged),
            "iterations": float(bp.num_iters),
            "runtime_sec": runtime_sec,
        }
        rows.append(row)

        _save_example(
            puzzle=puzzle,
            prediction=predicted_grid,
            output_path=examples_dir / f"puzzle_{idx:03d}.txt",
            solved=solved,
            converged=converged,
            runtime_sec=runtime_sec,
        )

        print(
            f"[{idx + 1}/{len(puzzles)}] Solved={solved} CluesOK={clues_consistent} "
            f"Converged={converged} Iter={bp.num_iters} Runtime={runtime_sec:.3f}s"
        )

    csv_path = results_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "solved",
                "clues_consistent",
                "converged",
                "iterations",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _save_convergence_plot(rows, plots_dir / "convergence.png")

    summary = {
        "num_puzzles": len(rows),
        "solve_rate": float(np.mean([row["solved"] for row in rows])),
        "clue_consistency_rate": float(np.mean([row["clues_consistent"] for row in rows])),
        "convergence_rate": float(np.mean([row["converged"] for row in rows])),
        "mean_iterations": float(np.mean([row["iterations"] for row in rows])),
        "mean_runtime_sec": float(np.mean([row["runtime_sec"] for row in rows])),
        "config": {
            "n_samples": args.n_samples,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "seed": args.seed,
        },
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-iters", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results/sudoku")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    np.random.seed(args.seed)

    summary = run_experiment(args)
    print("\nSudoku BP summary")
    print(f"Puzzles: {summary['num_puzzles']}")
    print(f"Solve rate: {summary['solve_rate']:.4f}")
    print(f"Clue consistency rate: {summary['clue_consistency_rate']:.4f}")
    print(f"Convergence rate: {summary['convergence_rate']:.4f}")
    print(f"Mean iterations: {summary['mean_iterations']:.2f}")
    print(f"Mean runtime (s): {summary['mean_runtime_sec']:.4f}")


if __name__ == "__main__":
    main()
