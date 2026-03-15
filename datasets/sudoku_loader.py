"""Sudoku dataset loader using HuggingFace datasets API with offline fallback."""

from __future__ import annotations

import json
import subprocess
import warnings
from typing import List

import numpy as np

DATASET_CANDIDATES = (
    "Ritvik19/Sudoku-Dataset",
    "sunnytqin/sudoku",
    "deanrhowe87/sudoku_data_small",
)

# Small fallback set so experiments remain runnable offline.
FALLBACK_PUZZLES = [
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
    "300000000005009000200504000020000700160000058704310600000890100000067080000005437",
    "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
    "020810740700003100090002805009040087400208003160030200302700060005600008076051090",
    "100920000524010000000000070050008102000000000402700090060000000000030945000071006",
    "043080250600000000000001094900004070000608000010200003820500000000000005034090710",
    "480006902002008001900370060840010200003704100001060049020085007700900600609200018",
    "000900002050123400030000160908000000070000090000000205091000050007439020400007000",
    "001900003900700160030005007050000009004302600200000070600100030042007006500006800",
]


def _parse_sudoku(raw: object) -> np.ndarray:
    """Parse raw Sudoku representation into a 9x9 int grid (0 = empty)."""
    if isinstance(raw, np.ndarray):
        grid = np.asarray(raw, dtype=np.int32)
        if grid.shape != (9, 9):
            raise ValueError(f"Expected shape (9, 9), got {grid.shape}")
        return np.where((grid >= 1) & (grid <= 9), grid, 0)

    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw)
        if arr.shape == (9, 9):
            grid = arr.astype(np.int32)
            return np.where((grid >= 1) & (grid <= 9), grid, 0)
        if arr.size == 81:
            grid = arr.reshape(9, 9).astype(np.int32)
            return np.where((grid >= 1) & (grid <= 9), grid, 0)

    if isinstance(raw, str):
        cleaned = ["0" if ch == "." else ch for ch in raw if ch in "0123456789."]
        if len(cleaned) == 81:
            grid = np.array([int(c) for c in cleaned], dtype=np.int32).reshape(9, 9)
            return np.where((grid >= 1) & (grid <= 9), grid, 0)

        tokens = raw.replace(",", " ").split()
        numeric_tokens = [t for t in tokens if t.isdigit()]
        if len(numeric_tokens) == 81:
            grid = np.array([int(t) for t in numeric_tokens], dtype=np.int32).reshape(9, 9)
            return np.where((grid >= 1) & (grid <= 9), grid, 0)

    raise ValueError(f"Unsupported Sudoku sample format: {type(raw)!r}")


def _load_with_huggingface(n_samples: int) -> List[np.ndarray]:
    """Load Sudoku puzzles with HuggingFace datasets in an isolated subprocess.

    We run loading in a subprocess with working directory outside the repo to
    avoid import name conflicts with this project's local ``datasets`` package.
    """
    child_code = r'''
import json
import numpy as np
import sys
from datasets import load_dataset


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def extract_raw(record):
    field_candidates = ("puzzle", "quiz", "quizzes", "question", "input", "grid")
    for key in field_candidates:
        if key in record:
            return record[key]
    for value in record.values():
        if isinstance(value, (str, list, tuple, dict)):
            return value
    return None


dataset_name = sys.argv[1]
n_samples = int(sys.argv[2])
k = max(5 * n_samples, 20)

try:
    ds = load_dataset(dataset_name, split=f"train[:{k}]")
except Exception:
    ds_dict = load_dataset(dataset_name)
    split = "train" if "train" in ds_dict else next(iter(ds_dict.keys()))
    ds = ds_dict[split]

raw_puzzles = []
for record in ds:
    raw = extract_raw(record)
    if raw is not None:
        raw_puzzles.append(to_jsonable(raw))
    if len(raw_puzzles) >= n_samples:
        break

print(json.dumps(raw_puzzles))
'''

    errors: List[str] = []
    for dataset_name in DATASET_CANDIDATES:
        proc = subprocess.run(
            [
                "python",
                "-c",
                child_code,
                dataset_name,
                str(n_samples),
            ],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
            errors.append(f"{dataset_name}: {err}")
            continue

        try:
            raw_items = json.loads(proc.stdout)
        except Exception as exc:
            errors.append(f"{dataset_name}: invalid JSON output ({exc})")
            continue

        puzzles: List[np.ndarray] = []
        for item in raw_items:
            try:
                puzzles.append(_parse_sudoku(item))
            except Exception:
                continue

        if puzzles:
            return puzzles[:n_samples]

        errors.append(f"{dataset_name}: no parseable Sudoku puzzles found")

    joined = " | ".join(errors[:5])
    raise RuntimeError(f"All HF Sudoku dataset candidates failed. Details: {joined}")


def _load_fallback(n_samples: int) -> List[np.ndarray]:
    """Return deterministic fallback puzzle set if HF loading is unavailable."""
    puzzles: List[np.ndarray] = []
    for i in range(n_samples):
        puzzles.append(_parse_sudoku(FALLBACK_PUZZLES[i % len(FALLBACK_PUZZLES)]))
    return puzzles


def load_sudoku_dataset(n_samples: int = 100) -> List[np.ndarray]:
    """Load Sudoku puzzles as 9x9 grids with zeros for empty cells.

    The loader first tries HuggingFace datasets API, then falls back to a
    deterministic local puzzle set if API access is unavailable.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    try:
        return _load_with_huggingface(n_samples)
    except Exception as exc:
        warnings.warn(
            "Falling back to built-in Sudoku puzzles because HuggingFace loading "
            f"failed: {exc}",
            RuntimeWarning,
        )
        return _load_fallback(n_samples)
