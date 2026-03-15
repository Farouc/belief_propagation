"""Utilities for loading GrabCut-style foreground/background datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from skimage import io

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_HINTS = ("mask", "trimap", "gt", "annot", "alpha")
MASK_SUFFIXES = ("_mask", "_gt", "_trimap", "_alpha", "_seg")


def _load_image(path: Path) -> np.ndarray:
    """Load image as float64 array in [0, 1]."""
    image = io.imread(path).astype(np.float64)
    if image.max() > 1.0:
        image /= 255.0
    return image


def _normalize_stem(stem: str) -> str:
    """Normalize mask filename stems by removing common suffix hints."""
    normalized = stem.lower()
    for suffix in MASK_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _build_mask_index(root: Path) -> Dict[str, Path]:
    """Index mask files by normalized stem."""
    index: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        descriptor = "/".join(part.lower() for part in path.parts)
        if not any(hint in descriptor for hint in MASK_HINTS):
            continue

        index.setdefault(_normalize_stem(path.stem), path)
    return index


def _is_image_file(path: Path) -> bool:
    """Check whether a path is likely an input image (not a mask)."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    descriptor = "/".join(part.lower() for part in path.parts)
    return not any(hint in descriptor for hint in MASK_HINTS)


def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert several common GrabCut mask conventions to binary mask."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]

    unique_values = set(np.unique(arr).tolist())
    if unique_values.issubset({0, 1, 2, 3}):
        # OpenCV GrabCut convention: 0/2 background, 1/3 foreground.
        return np.isin(arr, [1, 3]).astype(np.uint8)

    if arr.max() <= 1.0:
        return (arr > 0.5).astype(np.uint8)

    return (arr > 127).astype(np.uint8)


def _find_mask_for_image(image_path: Path, mask_index: Dict[str, Path], root: Path) -> Path | None:
    """Resolve best-effort matching mask path for an image."""
    stem = image_path.stem.lower()
    if stem in mask_index:
        return mask_index[stem]

    candidate_dirs = [root / "masks", root / "mask", root / "ground_truth", root / "gt"]
    candidate_names = [
        image_path.stem,
        f"{image_path.stem}_mask",
        f"{image_path.stem}_gt",
        f"{image_path.stem}_trimap",
        f"{image_path.stem}_alpha",
    ]

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for name in candidate_names:
            for ext in IMAGE_EXTENSIONS:
                candidate = directory / f"{name}{ext}"
                if candidate.exists():
                    return candidate
    return None


def load_grabcut_dataset(
    path: str | Path, max_images: int = 20
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load GrabCut-style dataset from a local directory.

    Parameters
    ----------
    path:
        Root dataset directory.
    max_images:
        Number of image/mask pairs to load.

    Returns
    -------
    images:
        List of loaded images as float arrays in [0, 1].
    masks:
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

    for image_path in image_files:
        if len(images) >= max_images:
            break

        mask_path = _find_mask_for_image(image_path, mask_index, root)
        if mask_path is None:
            continue

        image = _load_image(image_path)
        mask = _mask_to_binary(io.imread(mask_path))

        images.append(image)
        masks.append(mask)

    if not images:
        raise RuntimeError(
            "No GrabCut image/mask pairs were loaded. Check dataset layout and filenames."
        )

    return images, masks
