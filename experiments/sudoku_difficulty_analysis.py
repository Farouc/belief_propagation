"""Difficulty-by-clue analysis for Sudoku BP/TRW experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.sudoku_loader import load_sudoku_dataset
from utils.metrics import save_metrics_csv


CLASSIC_ORDER = ["hard", "medium", "easy"]


def _read_metrics_csv(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")

    parsed: List[Dict[str, float]] = []
    for row in rows:
        parsed.append({key: float(value) for key, value in row.items()})
    return parsed


def _sort_by_index(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    return sorted(rows, key=lambda r: int(r["index"]))


def _clue_counts(n_samples: int) -> np.ndarray:
    puzzles = load_sudoku_dataset(n_samples=n_samples)
    clues = [int(np.sum(np.asarray(puzzle) > 0)) for puzzle in puzzles]
    return np.asarray(clues, dtype=np.int32)


def _clue_counts_from_examples(examples_root: Path, n_samples: int) -> np.ndarray | None:
    """Read clue counts from saved Sudoku example text files.

    This path guarantees alignment with the exact puzzle set used in the
    previously saved metrics when those example files come from the same run.
    """
    if not examples_root.exists():
        return None

    pattern = re.compile(r"puzzle_(\d+)\.txt$")
    index_to_clues: Dict[int, int] = {}

    for txt_path in sorted(examples_root.rglob("puzzle_*.txt")):
        match = pattern.search(txt_path.name)
        if match is None:
            continue
        idx = int(match.group(1))
        if idx >= n_samples:
            continue

        lines = txt_path.read_text(encoding="utf-8").splitlines()
        if "Puzzle:" not in lines:
            continue
        start = lines.index("Puzzle:") + 1
        grid_lines = lines[start : start + 9]
        if len(grid_lines) != 9:
            continue

        values: List[int] = []
        for line in grid_lines:
            tokens = line.strip().split()
            if len(tokens) != 9:
                values = []
                break
            values.extend(int(t) for t in tokens)
        if len(values) != 81:
            continue

        index_to_clues[idx] = int(np.sum(np.asarray(values, dtype=np.int32) > 0))

    if len(index_to_clues) < n_samples:
        return None

    clues = np.array([index_to_clues[i] for i in range(n_samples)], dtype=np.int32)
    return clues


def _assign_classic(clue_count: int) -> str:
    # Common practical binning for clue-count difficulty proxy.
    if clue_count >= 36:
        return "easy"
    if clue_count >= 30:
        return "medium"
    return "hard"


def _assign_quantile_bins(clues: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    q1 = float(np.quantile(clues, 1.0 / 3.0))
    q2 = float(np.quantile(clues, 2.0 / 3.0))
    labels = np.empty(clues.shape[0], dtype=object)
    labels[clues <= q1] = "hard_q"
    labels[(clues > q1) & (clues <= q2)] = "medium_q"
    labels[clues > q2] = "easy_q"
    return labels, {"q1": q1, "q2": q2}


def _aggregate_by_bucket(
    method: str,
    rows: Sequence[Dict[str, float]],
    clues: np.ndarray,
    buckets: Sequence[str],
    labels: Sequence[str],
) -> List[Dict[str, float | str]]:
    output: List[Dict[str, float | str]] = []
    labels_arr = np.asarray(labels, dtype=object)

    for bucket in buckets:
        indices = np.where(labels_arr == bucket)[0]
        if indices.size == 0:
            output.append(
                {
                    "method": method,
                    "difficulty_bucket": bucket,
                    "count": 0.0,
                    "clue_min": np.nan,
                    "clue_max": np.nan,
                    "clue_mean": np.nan,
                    "solve_rate": np.nan,
                    "convergence_rate": np.nan,
                    "mean_iterations": np.nan,
                    "std_iterations": np.nan,
                    "mean_runtime_sec": np.nan,
                    "std_runtime_sec": np.nan,
                }
            )
            continue

        selected_rows = [rows[int(i)] for i in indices]
        solve = np.array([r["solved"] for r in selected_rows], dtype=np.float64)
        converged = np.array([r["converged"] for r in selected_rows], dtype=np.float64)
        iterations = np.array([r["iterations"] for r in selected_rows], dtype=np.float64)
        runtime = np.array([r["runtime_sec"] for r in selected_rows], dtype=np.float64)
        clue_vals = clues[indices].astype(np.float64)

        output.append(
            {
                "method": method,
                "difficulty_bucket": bucket,
                "count": float(indices.size),
                "clue_min": float(np.min(clue_vals)),
                "clue_max": float(np.max(clue_vals)),
                "clue_mean": float(np.mean(clue_vals)),
                "solve_rate": float(np.mean(solve)),
                "convergence_rate": float(np.mean(converged)),
                "mean_iterations": float(np.mean(iterations)),
                "std_iterations": float(np.std(iterations)),
                "mean_runtime_sec": float(np.mean(runtime)),
                "std_runtime_sec": float(np.std(runtime)),
            }
        )
    return output


def _plot_clue_histogram(clues: np.ndarray, output_path: Path, q1: float, q2: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(int(clues.min()), int(clues.max()) + 2) - 0.5
    ax.hist(clues, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.axvline(30, color="#C44E52", linestyle="--", linewidth=1.8, label="classic: hard/medium")
    ax.axvline(36, color="#55A868", linestyle="--", linewidth=1.8, label="classic: medium/easy")
    ax.axvline(q1, color="#8172B2", linestyle=":", linewidth=2.0, label=f"quantile q1={q1:.1f}")
    ax.axvline(q2, color="#937860", linestyle=":", linewidth=2.0, label=f"quantile q2={q2:.1f}")
    ax.set_title("Sudoku Clue Count Distribution")
    ax.set_xlabel("Number of given clues")
    ax.set_ylabel("Puzzle count")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _grouped_bar(
    rows: Sequence[Dict[str, float | str]],
    buckets: Sequence[str],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    row_map = {(str(r["method"]), str(r["difficulty_bucket"])): r for r in rows}
    bp_vals = [float(row_map[("bp", b)][value_key]) for b in buckets]
    trw_vals = [float(row_map[("trw", b)][value_key]) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, bp_vals, width, label="BP", color="#4C72B0")
    ax.bar(x + width / 2, trw_vals, width, label="TRW", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(list(buckets))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_clue_count(
    bp_rows: Sequence[Dict[str, float]],
    trw_rows: Sequence[Dict[str, float]],
    clues: np.ndarray,
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    y_lim_01: bool = False,
) -> None:
    clue_values = np.unique(clues)
    clue_values = np.sort(clue_values)

    def _stats(rows: Sequence[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = []
        stds = []
        counts = []
        metric = np.array([r[metric_key] for r in rows], dtype=np.float64)
        for c in clue_values:
            idx = np.where(clues == c)[0]
            vals = metric[idx]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            counts.append(float(len(vals)))
        return np.array(means), np.array(stds), np.array(counts)

    bp_mean, bp_std, bp_counts = _stats(bp_rows)
    trw_mean, trw_std, trw_counts = _stats(trw_rows)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bp_size = 30 + 10 * np.sqrt(bp_counts)
    trw_size = 30 + 10 * np.sqrt(trw_counts)

    ax.plot(clue_values, bp_mean, color="#4C72B0", linewidth=2.3, label="BP")
    ax.scatter(clue_values, bp_mean, s=bp_size, color="#4C72B0", alpha=0.9)
    ax.fill_between(clue_values, bp_mean - bp_std, bp_mean + bp_std, color="#4C72B0", alpha=0.18)

    ax.plot(clue_values, trw_mean, color="#C44E52", linewidth=2.3, label="TRW")
    ax.scatter(clue_values, trw_mean, s=trw_size, color="#C44E52", alpha=0.9)
    ax.fill_between(clue_values, trw_mean - trw_std, trw_mean + trw_std, color="#C44E52", alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Clue count")
    ax.set_ylabel(ylabel)
    if y_lim_01:
        ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_trw_bp_gaps_by_clue(
    bp_rows: Sequence[Dict[str, float]],
    trw_rows: Sequence[Dict[str, float]],
    clues: np.ndarray,
    output_path: Path,
) -> None:
    clue_values = np.sort(np.unique(clues))
    bp_solve = np.array([r["solved"] for r in bp_rows], dtype=np.float64)
    trw_solve = np.array([r["solved"] for r in trw_rows], dtype=np.float64)
    bp_runtime = np.array([r["runtime_sec"] for r in bp_rows], dtype=np.float64)
    trw_runtime = np.array([r["runtime_sec"] for r in trw_rows], dtype=np.float64)

    solve_gap = []
    runtime_ratio = []
    for c in clue_values:
        idx = np.where(clues == c)[0]
        solve_gap.append(float(np.mean(trw_solve[idx]) - np.mean(bp_solve[idx])))
        runtime_ratio.append(float(np.mean(trw_runtime[idx]) / max(np.mean(bp_runtime[idx]), 1e-12)))

    solve_gap_arr = np.array(solve_gap, dtype=np.float64)
    runtime_ratio_arr = np.array(runtime_ratio, dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    colors = ["#55A868" if v >= 0 else "#C44E52" for v in solve_gap_arr]
    axes[0].bar(clue_values, solve_gap_arr, color=colors, alpha=0.9)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("Solve-rate gap (TRW - BP)")
    axes[0].set_title("TRW vs BP gap by clue count")
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].plot(clue_values, runtime_ratio_arr, color="#8172B2", linewidth=2.3, marker="o")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Clue count")
    axes[1].set_ylabel("Runtime ratio (TRW / BP)")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_quantile_pareto(
    quantile_rows: Sequence[Dict[str, float | str]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {"bp": "#4C72B0", "trw": "#C44E52"}
    marker_map = {"hard_q": "s", "medium_q": "o", "easy_q": "^"}

    for row in quantile_rows:
        method = str(row["method"])
        bucket = str(row["difficulty_bucket"])
        runtime = float(row["mean_runtime_sec"])
        solve_rate = float(row["solve_rate"])
        count = float(row["count"])
        marker_size = 35 + 4.0 * count

        ax.scatter(
            runtime,
            solve_rate,
            s=marker_size,
            color=color_map[method],
            marker=marker_map[bucket],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.7,
        )
        ax.annotate(
            f"{method.upper()}-{bucket}",
            (runtime, solve_rate),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8.5,
        )

    ax.set_xlabel("Mean runtime (s)")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Sudoku difficulty buckets: runtime/solve Pareto view")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> Dict[str, object]:
    bp_rows = _sort_by_index(_read_metrics_csv(Path(args.bp_metrics)))
    trw_rows = _sort_by_index(_read_metrics_csv(Path(args.trw_metrics)))
    n_available = min(len(bp_rows), len(trw_rows), args.n_samples)

    clue_source = "loader"
    clues = _clue_counts_from_examples(Path(args.examples_root), n_samples=n_available)
    if clues is None:
        clues = _clue_counts(n_samples=n_available)
    else:
        clue_source = "examples"

    n = min(n_available, clues.size)
    bp_rows = bp_rows[:n]
    trw_rows = trw_rows[:n]
    clues = clues[:n]

    classic_labels = np.array([_assign_classic(int(c)) for c in clues], dtype=object)
    quantile_labels, quantiles = _assign_quantile_bins(clues)

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    classic_rows: List[Dict[str, float | str]] = []
    classic_rows.extend(_aggregate_by_bucket("bp", bp_rows, clues, CLASSIC_ORDER, classic_labels))
    classic_rows.extend(_aggregate_by_bucket("trw", trw_rows, clues, CLASSIC_ORDER, classic_labels))

    quantile_order = ["hard_q", "medium_q", "easy_q"]
    quantile_rows: List[Dict[str, float | str]] = []
    quantile_rows.extend(_aggregate_by_bucket("bp", bp_rows, clues, quantile_order, quantile_labels))
    quantile_rows.extend(_aggregate_by_bucket("trw", trw_rows, clues, quantile_order, quantile_labels))

    save_metrics_csv(classic_rows, results_dir / "difficulty_classic_table.csv")
    save_metrics_csv(quantile_rows, results_dir / "difficulty_quantile_table.csv")

    _plot_clue_histogram(
        clues=clues,
        output_path=plots_dir / "clue_count_histogram.png",
        q1=float(quantiles["q1"]),
        q2=float(quantiles["q2"]),
    )
    _grouped_bar(
        rows=classic_rows,
        buckets=CLASSIC_ORDER,
        value_key="solve_rate",
        title="Solve Rate by Difficulty Bucket (classic clue-count bins)",
        ylabel="Solve rate",
        output_path=plots_dir / "solve_rate_by_classic_bucket.png",
    )
    _grouped_bar(
        rows=classic_rows,
        buckets=CLASSIC_ORDER,
        value_key="mean_iterations",
        title="Mean Iterations by Difficulty Bucket",
        ylabel="Mean iterations",
        output_path=plots_dir / "iterations_by_classic_bucket.png",
    )
    _grouped_bar(
        rows=classic_rows,
        buckets=CLASSIC_ORDER,
        value_key="mean_runtime_sec",
        title="Mean Runtime by Difficulty Bucket",
        ylabel="Mean runtime (s)",
        output_path=plots_dir / "runtime_by_classic_bucket.png",
    )
    _plot_metric_vs_clue_count(
        bp_rows=bp_rows,
        trw_rows=trw_rows,
        clues=clues,
        metric_key="solved",
        title="Solve rate vs clue count",
        ylabel="Solve rate",
        output_path=plots_dir / "solve_rate_vs_clue_count.png",
        y_lim_01=True,
    )
    _plot_metric_vs_clue_count(
        bp_rows=bp_rows,
        trw_rows=trw_rows,
        clues=clues,
        metric_key="runtime_sec",
        title="Runtime vs clue count",
        ylabel="Runtime (s)",
        output_path=plots_dir / "runtime_vs_clue_count.png",
        y_lim_01=False,
    )
    _plot_metric_vs_clue_count(
        bp_rows=bp_rows,
        trw_rows=trw_rows,
        clues=clues,
        metric_key="iterations",
        title="Iterations vs clue count",
        ylabel="Iterations",
        output_path=plots_dir / "iterations_vs_clue_count.png",
        y_lim_01=False,
    )
    _plot_trw_bp_gaps_by_clue(
        bp_rows=bp_rows,
        trw_rows=trw_rows,
        clues=clues,
        output_path=plots_dir / "trw_bp_gap_by_clue_count.png",
    )
    _plot_quantile_pareto(
        quantile_rows=quantile_rows,
        output_path=plots_dir / "quantile_pareto_runtime_vs_solve.png",
    )

    classic_counts = {b: int(np.sum(classic_labels == b)) for b in CLASSIC_ORDER}
    quantile_counts = {b: int(np.sum(quantile_labels == b)) for b in quantile_order}
    unique_clues = np.unique(clues)
    diversity_ok = len(unique_clues) >= args.min_unique_clues and min(classic_counts.values()) >= args.min_bucket_size

    summary = {
        "num_puzzles": int(n),
        "num_unique_clue_counts": int(len(unique_clues)),
        "clue_count_min": int(np.min(clues)),
        "clue_count_max": int(np.max(clues)),
        "clue_count_mean": float(np.mean(clues)),
        "classic_bucket_counts": classic_counts,
        "quantile_bucket_counts": quantile_counts,
        "quantile_thresholds": quantiles,
        "diversity_sufficient_for_classic_bins": bool(diversity_ok),
        "clue_source": clue_source,
        "config": {
            "bp_metrics": str(args.bp_metrics),
            "trw_metrics": str(args.trw_metrics),
            "n_samples": args.n_samples,
            "min_unique_clues": args.min_unique_clues,
            "min_bucket_size": args.min_bucket_size,
            "examples_root": str(args.examples_root),
        },
    }
    with (results_dir / "difficulty_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp-metrics", type=str, default="results/sudoku/metrics.csv")
    parser.add_argument("--trw-metrics", type=str, default="results/sudoku/sudoku_trw_metrics.csv")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--min-unique-clues", type=int, default=6)
    parser.add_argument("--min-bucket-size", type=int, default=8)
    parser.add_argument("--examples-root", type=str, default="results/sudoku/examples")
    parser.add_argument("--results-dir", type=str, default="results/sudoku/difficulty_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(args)
    print("Sudoku difficulty-by-clue analysis summary")
    print(f"Puzzles analyzed: {summary['num_puzzles']}")
    print(
        f"Clue range: {summary['clue_count_min']}..{summary['clue_count_max']} "
        f"(unique counts: {summary['num_unique_clue_counts']})"
    )
    print(f"Classic bucket counts: {summary['classic_bucket_counts']}")
    print(f"Quantile bucket counts: {summary['quantile_bucket_counts']}")
    print(f"Clue source: {summary['clue_source']}")
    print(f"Diversity sufficient for classic bins: {summary['diversity_sufficient_for_classic_bins']}")


if __name__ == "__main__":
    main()
