"""Sudoku solving experiment using Tree-Reweighted Belief Propagation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.sudoku_loader import load_sudoku_dataset
from src.trw_belief_propagation import TreeReweightedBeliefPropagation
from utils.metrics import save_metrics_csv
from utils.sudoku_utils import beliefs_to_grid, build_sudoku_mrf, enforce_clues, is_valid_sudoku
from utils.trw_utils import compute_uniform_edge_weights


def _save_iterations_distribution(rows: List[Dict[str, float]], output_path: Path) -> None:
    iterations = [float(row["iterations"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(iterations, bins=min(15, max(6, len(iterations) // 5)), color="#6A3D9A", edgecolor="black")
    ax.set_title("Sudoku TRW Iteration Distribution")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_solve_rate_plot(rows: List[Dict[str, float]], output_path: Path) -> None:
    solved_count = int(np.sum([row["solved"] for row in rows]))
    unsolved_count = len(rows) - solved_count

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Solved", "Unsolved"], [solved_count, unsolved_count], color=["#2E8B57", "#C44E52"])
    ax.set_title("Sudoku TRW Solve Outcomes")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_plot(rows: List[Dict[str, float]], output_path: Path) -> None:
    indices = [int(row["index"]) for row in rows]
    iterations = [int(row["iterations"]) for row in rows]
    converged = [bool(row["converged"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2E8B57" if c else "#C44E52" for c in converged]
    ax.bar(indices, iterations, color=colors)
    ax.set_xlabel("Puzzle Index")
    ax.set_ylabel("TRW Iterations")
    ax.set_title("Sudoku TRW Convergence Behavior")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    puzzles = load_sudoku_dataset(n_samples=args.n_samples)

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []

    for idx, puzzle in enumerate(puzzles):
        t0 = time.perf_counter()

        mrf = build_sudoku_mrf(puzzle)
        rho = compute_uniform_edge_weights(mrf)
        trw = TreeReweightedBeliefPropagation(
            mrf,
            rho=rho,
            max_iters=args.max_iters,
            tol=args.tol,
            damping=args.damping,
        )
        converged = trw.run()
        beliefs = trw.compute_beliefs()

        predicted_grid = beliefs_to_grid(beliefs)
        predicted_grid = enforce_clues(predicted_grid, puzzle)

        runtime_sec = time.perf_counter() - t0
        clues_consistent = bool(np.all(predicted_grid[puzzle > 0] == puzzle[puzzle > 0]))
        solved = clues_consistent and is_valid_sudoku(predicted_grid)

        rows.append(
            {
                "index": idx,
                "solved": float(solved),
                "clues_consistent": float(clues_consistent),
                "converged": float(converged),
                "iterations": float(trw.num_iters),
                "runtime_sec": runtime_sec,
            }
        )

        print(
            f"[TRW {idx + 1}/{len(puzzles)}] Solved={solved} CluesOK={clues_consistent} "
            f"Converged={converged} Iter={trw.num_iters} Runtime={runtime_sec:.3f}s"
        )

    save_metrics_csv(rows, results_dir / "sudoku_trw_metrics.csv")
    _save_convergence_plot(rows, plots_dir / "sudoku_trw_convergence.png")
    _save_iterations_distribution(rows, plots_dir / "sudoku_trw_iterations_hist.png")
    _save_solve_rate_plot(rows, plots_dir / "sudoku_trw_solve_rate.png")

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
            "rho": "uniform_0.5",
        },
    }

    with (results_dir / "sudoku_trw_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-iters", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results/sudoku")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    summary = run_experiment(args)
    print("\nSudoku TRW summary")
    print(f"Puzzles: {summary['num_puzzles']}")
    print(f"Solve rate: {summary['solve_rate']:.4f} ± {summary['solve_rate_std']:.4f}")
    print(f"Clue consistency rate: {summary['clue_consistency_rate']:.4f}")
    print(f"Convergence rate: {summary['convergence_rate']:.4f}")
    print(f"Mean iterations: {summary['mean_iterations']:.2f} ± {summary['std_iterations']:.2f}")
    print(f"Mean runtime (s): {summary['mean_runtime_sec']:.4f} ± {summary['std_runtime_sec']:.4f}")


if __name__ == "__main__":
    main()
