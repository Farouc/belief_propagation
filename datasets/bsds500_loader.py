"""Utilities for loading BSDS500 images and binary masks for BP segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
from skimage import color, io, transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = (".mat", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


def _to_float_gray(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale float64 in [0, 1]."""
    img = np.asarray(image)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def _collect_numeric_arrays(obj: object, out: List[np.ndarray]) -> None:
    """Recursively collect numeric arrays from nested scipy mat structures."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for item in obj.flat:
                _collect_numeric_arrays(item, out)
        elif np.issubdtype(obj.dtype, np.number):
            out.append(np.asarray(obj))
        return

    if isinstance(obj, np.void):
        for name in obj.dtype.names or ():
            _collect_numeric_arrays(obj[name], out)


def _load_bsds_mat_mask(path: Path) -> np.ndarray:
    """Load the first segmentation map from a BSDS500 .mat annotation file."""
    data = loadmat(path)
    if "groundTruth" not in data:
        raise ValueError(f"Expected key 'groundTruth' in {path}")

    gt = data["groundTruth"]

    # Standard BSDS format: groundTruth is object array where each item is a
    # struct-like ndarray with fields {'Segmentation', 'Boundaries'}.
    for item in gt.flat:
        if isinstance(item, np.ndarray) and item.dtype.names and "Segmentation" in item.dtype.names:
            seg = item["Segmentation"][0, 0]
            seg = np.asarray(seg).squeeze()
            if seg.ndim == 2:
                return seg

    # Fallback: recursively scan numeric arrays and pick the richest 2D map.
    arrays: List[np.ndarray] = []
    _collect_numeric_arrays(gt, arrays)
    arrays_2d = [arr.squeeze() for arr in arrays if arr.squeeze().ndim == 2]
    if not arrays_2d:
        raise ValueError(f"Could not extract 2D segmentation array from {path}")

    def score(arr: np.ndarray) -> Tuple[int, int]:
        unique_count = int(np.unique(arr).size)
        return unique_count, int(arr.size)

    return max(arrays_2d, key=score)


def _to_binary_mask(segmentation: np.ndarray) -> np.ndarray:
    """Convert a segmentation map to a binary mask (foreground=1, background=0)."""
    seg = np.asarray(segmentation)
    if seg.ndim == 3:
        seg = seg[..., 0]

    values = np.unique(seg)
    if values.size <= 2:
        return (seg > values.min()).astype(np.uint8)

    # Treat the dominant label as background, all others as foreground.
    labels, counts = np.unique(seg, return_counts=True)
    background_label = labels[np.argmax(counts)]
    return (seg != background_label).astype(np.uint8)


def _resolve_image_and_gt_roots(dataset_root: Path) -> Tuple[Path, Path]:
    """Find likely BSDS500 image and ground-truth directories."""
    image_candidates = [
        dataset_root / "data" / "images",
        dataset_root / "images",
        dataset_root,
    ]
    gt_candidates = [
        dataset_root / "data" / "groundTruth",
        dataset_root / "groundTruth",
        dataset_root / "masks",
        dataset_root / "annotations",
    ]

    image_root = next((p for p in image_candidates if p.exists()), None)
    gt_root = next((p for p in gt_candidates if p.exists()), None)

    if image_root is None or gt_root is None:
        if dataset_root.exists() and dataset_root.is_dir():
            entries = sorted(p.name for p in dataset_root.iterdir())
            entries_text = ", ".join(entries) if entries else "(empty directory)"
        else:
            entries_text = "(directory does not exist)"

        image_candidates_text = "\n".join(f"  - {p}" for p in image_candidates)
        gt_candidates_text = "\n".join(f"  - {p}" for p in gt_candidates)

        raise FileNotFoundError(
            "Could not locate BSDS500 images/groundTruth directories.\n"
            f"Provided root: {dataset_root}\n"
            "Expected image directory at one of:\n"
            f"{image_candidates_text}\n"
            "Expected annotation directory at one of:\n"
            f"{gt_candidates_text}\n"
            f"Existing entries under provided root: {entries_text}\n"
            "Tip: pass the folder that contains `data/images` and `data/groundTruth`, "
            "or `images` and `groundTruth`."
        )

    return image_root, gt_root


def _list_image_files(image_root: Path) -> List[Path]:
    """List image files in deterministic order across train/val/test."""
    files: List[Path] = []
    for split in SPLITS:
        split_dir = image_root / split
        if split_dir.exists():
            split_files = [p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            files.extend(sorted(split_files))

    if files:
        return files

    return sorted(
        p
        for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_mask_file(image_path: Path, image_root: Path, gt_root: Path) -> Path | None:
    """Find a matching ground-truth file for an image path by stem and split."""
    relative = image_path.relative_to(image_root)
    stem = image_path.stem

    search_dirs: List[Path] = []
    if len(relative.parts) >= 2 and (gt_root / relative.parts[0]).exists():
        search_dirs.append(gt_root / relative.parts[0])
    search_dirs.append(gt_root)

    for directory in search_dirs:
        for ext in MASK_EXTENSIONS:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        recursive_matches = sorted(
            p for p in directory.rglob(f"{stem}.*") if p.suffix.lower() in MASK_EXTENSIONS
        )
        if recursive_matches:
            return recursive_matches[0]

    return None


def load_bsds500_dataset(
    path: str | Path, max_images: int = 20, resize: Tuple[int, int] | None = (80, 80)
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load a BSDS500 subset and convert annotations to binary masks.

    Parameters
    ----------
    path:
        Root path to BSDS500 dataset.
    max_images:
        Maximum number of image/mask pairs to load.
    resize:
        Optional output size (height, width). If ``None``, keep original sizes.

    Returns
    -------
    images:
        List of grayscale float images in [0, 1].
    ground_truth_masks:
        List of binary masks with values in {0, 1}.
    """
    if max_images <= 0:
        raise ValueError("max_images must be positive")

    dataset_root = Path(path)
    image_root, gt_root = _resolve_image_and_gt_roots(dataset_root)

    image_files = _list_image_files(image_root)
    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for image_path in image_files:
        if len(images) >= max_images:
            break

        mask_path = _find_mask_file(image_path, image_root, gt_root)
        if mask_path is None:
            continue

        image = _to_float_gray(io.imread(image_path))
        if mask_path.suffix.lower() == ".mat":
            mask_raw = _load_bsds_mat_mask(mask_path)
        else:
            mask_raw = io.imread(mask_path)
        mask = _to_binary_mask(mask_raw)

        if resize is not None:
            image = transform.resize(
                image,
                resize,
                order=1,
                anti_aliasing=True,
                preserve_range=True,
            )
            mask = transform.resize(
                mask.astype(np.float64),
                resize,
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.uint8)

        images.append(image.astype(np.float64))
        masks.append(mask.astype(np.uint8))

    if not images:
        raise RuntimeError(
            "No BSDS500 image/mask pairs were loaded. Check dataset path and format."
        )

    return images, masks
