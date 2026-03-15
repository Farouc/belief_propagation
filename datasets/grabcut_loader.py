"""Utilities for loading GrabCut-style foreground/background datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from skimage import io

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}
MASK_HINTS = ("mask", "trimap", "gt", "annot", "alpha")
MASK_SUFFIXES = ("_mask", "_gt", "_trimap", "_alpha", "_seg")


def _load_image(path: Path) -> np.ndarray:
    """Load image as float64 array in [0, 1]."""
    image = io.imread(path).astype(np.float64)
    if image.max() > 1.0:
        image /= 255.0
    return image


def _normalize_stem(stem: str) -> str:
    """Normalize mask/image stems so matching survives naming variations."""
    normalized = stem.lower()
    for suffix in MASK_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    # Map variants like image1652 / mask1652 / img_1652 to "1652".
    match = re.match(r"^(?:image|img|mask)[_\-]?(\d+)$", normalized)
    if match:
        return match.group(1)

    return normalized


def _build_mask_index(root: Path) -> Dict[str, List[Path]]:
    """Index mask files by normalized stem."""
    index: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        descriptor = "/".join(part.lower() for part in path.parts)
        if not any(hint in descriptor for hint in MASK_HINTS):
            continue

        key = _normalize_stem(path.stem)
        index.setdefault(key, []).append(path)
    return index


def _is_image_file(path: Path) -> bool:
    """Check whether a path is likely an input image (not a mask)."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    descriptor = "/".join(part.lower() for part in path.parts)
    return not any(hint in descriptor for hint in MASK_HINTS)


def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert GrabCut mask conventions to binary (foreground=1, background=0)."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]

    unique_values = set(np.unique(arr).tolist())
    if unique_values.issubset({0, 1, 2, 3}):
        # OpenCV GrabCut convention:
        # 0 = background, 1 = foreground, 2 = probable background, 3 = probable foreground.
        return np.isin(arr, [1, 3]).astype(np.uint8)

    if unique_values.issubset({0, 255}):
        return (arr == 255).astype(np.uint8)

    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
        return (arr > 0.5).astype(np.uint8)

    return (arr > 127).astype(np.uint8)


def _score_mask(mask: np.ndarray) -> float:
    """Score mask quality so we prefer candidates with both classes present."""
    unique, counts = np.unique(mask, return_counts=True)
    if unique.size < 2:
        return 0.0
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return entropy


def _choose_best_mask_candidate(candidates: List[Path]) -> Path | None:
    """Select the most informative candidate mask among possible matches."""
    if not candidates:
        return None

    best_path: Path | None = None
    best_score = -1.0
    for path in candidates:
        try:
            mask = _mask_to_binary(io.imread(path))
            score = _score_mask(mask)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_path = path

    return best_path if best_path is not None else candidates[0]


def _find_mask_for_image(image_path: Path, mask_index: Dict[str, List[Path]], root: Path) -> Path | None:
    """Resolve best-effort matching mask path for an image."""
    stem_key = _normalize_stem(image_path.stem)

    if stem_key in mask_index:
        candidate = _choose_best_mask_candidate(mask_index[stem_key])
        if candidate is not None:
            return candidate

    digits = re.findall(r"\d+", image_path.stem)
    if digits:
        numeric_key = digits[-1]
        if numeric_key in mask_index:
            candidate = _choose_best_mask_candidate(mask_index[numeric_key])
            if candidate is not None:
                return candidate

    candidate_dirs = [
        root / "masks",
        root / "mask",
        root / "ground_truth",
        root / "gt",
        root / "annotations",
        root / "raw",
    ]

    digit = digits[-1] if digits else ""
    candidate_names = [
        image_path.stem,
        f"{image_path.stem}_mask",
        f"{image_path.stem}_gt",
        f"{image_path.stem}_trimap",
        f"{image_path.stem}_alpha",
    ]
    if digit:
        candidate_names.extend([f"mask{digit}", f"image{digit}", f"image{digit}_mask"])

    candidates: List[Path] = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for name in candidate_names:
            for ext in IMAGE_EXTENSIONS:
                candidate = directory / f"{name}{ext}"
                if candidate.exists():
                    candidates.append(candidate)

    return _choose_best_mask_candidate(candidates)


def load_grabcut_dataset(
    path: str | Path,
    max_images: int = 20,
    require_both_classes: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load GrabCut-style dataset from a local directory.

    Parameters
    ----------
    path:
        Root dataset directory.
    max_images:
        Number of image/mask pairs to load.
    require_both_classes:
        If ``True``, skip masks that are all-foreground or all-background.

    Returns
    -------
    images:
        List of loaded images as float arrays in [0, 1].
    binary_masks:
        List of binary masks with values in {0, 1}.
    """
    if max_images <= 0:
        raise ValueError("max_images must be positive")

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"GrabCut dataset path not found: {root}")

    mask_index = _build_mask_index(root)
    image_files = sorted(p for p in root.rglob("*") if p.is_file() and _is_image_file(p))

    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    seen_keys: set[str] = set()

    for image_path in image_files:
        if len(images) >= max_images:
            break

        image_key = _normalize_stem(image_path.stem)
        if image_key in seen_keys:
            continue

        mask_path = _find_mask_for_image(image_path, mask_index, root)
        if mask_path is None:
            continue

        image = _load_image(image_path)
        mask = _mask_to_binary(io.imread(mask_path)).astype(np.uint8)

        if require_both_classes and np.unique(mask).size < 2:
            continue

        images.append(image)
        masks.append(mask)
        seen_keys.add(image_key)

    if not images:
        raise RuntimeError(
            "No GrabCut image/mask pairs were loaded. Check dataset layout and filenames."
        )

    return images, masks
