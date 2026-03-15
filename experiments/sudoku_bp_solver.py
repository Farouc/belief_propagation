"""Sudoku solving experiment using belief propagation on a constraint MRF."""

from __future__ import annotations

import argparse
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
from utils.metrics import save_metrics_csv
from utils.sudoku_utils import (
    beliefs_to_grid,
    build_sudoku_mrf,
    enforce_clues,
    format_grid,
    is_valid_sudoku,
)


def _save_iterations_distribution(rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save histogram of BP iterations across puzzles."""
    iterations = [float(row["iterations"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(iterations, bins=min(15, max(6, len(iterations) // 5)), color="#4C72B0", edgecolor="black")
    ax.set_title("Sudoku BP Iteration Distribution")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_runtime_distribution(rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save runtime histogram across puzzles."""
    runtimes = [float(row["runtime_sec"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(runtimes, bins=min(15, max(6, len(runtimes) // 5)), color="#55A868", edgecolor="black")
    ax.set_title("Sudoku Runtime Distribution")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_solve_rate_plot(rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save bar chart of solved vs unsolved puzzle counts."""
    solved_count = int(np.sum([row["solved"] for row in rows]))
    unsolved_count = len(rows) - solved_count

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Solved", "Unsolved"], [solved_count, unsolved_count], color=["#2E8B57", "#C44E52"])
    ax.set_title("Sudoku Solve Outcomes")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_curve_example(rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save puzzle-wise iterations colored by convergence."""
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
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
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
    solved_dir = examples_dir / "solved"
    unsolved_dir = examples_dir / "unsolved"
    plots_dir = results_dir / "plots"
    solved_dir.mkdir(parents=True, exist_ok=True)
    unsolved_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []

    for idx, puzzle in enumerate(puzzles):
        t0 = time.perf_counter()

        mrf = build_sudoku_mrf(puzzle)
        bp = BeliefPropagation(
            mrf,
            max_iters=args.max_iters,
            tol=args.tol,
            damping=args.damping,
        )
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

        example_dir = solved_dir if solved else unsolved_dir
        _save_example(
            puzzle=puzzle,
            prediction=predicted_grid,
            output_path=example_dir / f"puzzle_{idx:03d}.txt",
            solved=solved,
            converged=converged,
            runtime_sec=runtime_sec,
        )

        print(
            f"[{idx + 1}/{len(puzzles)}] Solved={solved} CluesOK={clues_consistent} "
            f"Converged={converged} Iter={bp.num_iters} Runtime={runtime_sec:.3f}s"
        )

    save_metrics_csv(rows, results_dir / "metrics.csv")

    _save_convergence_curve_example(rows, plots_dir / "convergence.png")
    _save_iterations_distribution(rows, plots_dir / "iterations_hist.png")
    _save_runtime_distribution(rows, plots_dir / "runtime_hist.png")
    _save_solve_rate_plot(rows, plots_dir / "solve_rate.png")

    solved_values = np.array([row["solved"] for row in rows], dtype=np.float64)
    converged_values = np.array([row["converged"] for row in rows], dtype=np.float64)

    summary = {
        "num_puzzles": len(rows),
        "solve_rate": float(np.mean(solved_values)),
        "solve_rate_std": float(np.std(solved_values)),
        "clue_consistency_rate": float(np.mean([row["clues_consistent"] for row in rows])),
        "convergence_rate": float(np.mean(converged_values)),
        "mean_iterations": float(np.mean([row["iterations"] for row in rows])),
        "std_iterations": float(np.std([row["iterations"] for row in rows])),
        "mean_runtime_sec": float(np.mean([row["runtime_sec"] for row in rows])),
        "std_runtime_sec": float(np.std([row["runtime_sec"] for row in rows])),
        "config": {
            "n_samples": args.n_samples,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "damping": args.damping,
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
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
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
    print(f"Solve rate: {summary['solve_rate']:.4f} ± {summary['solve_rate_std']:.4f}")
    print(f"Clue consistency rate: {summary['clue_consistency_rate']:.4f}")
    print(f"Convergence rate: {summary['convergence_rate']:.4f}")
    print(f"Mean iterations: {summary['mean_iterations']:.2f} ± {summary['std_iterations']:.2f}")
    print(f"Mean runtime (s): {summary['mean_runtime_sec']:.4f} ± {summary['std_runtime_sec']:.4f}")


if __name__ == "__main__":
    main()
