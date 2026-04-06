from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_HINTS = {
    "annotation",
    "annotations",
    "groundtruth",
    "ground_truth",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
}

SATELLITE_HINTS = {
    "sat",
    "satellite",
    "image",
    "images",
    "img",
    "rgb",
}


@dataclass(frozen=True)
class ImageMaskPair:
    sample_id: str
    image_path: Path
    mask_path: Path


def _tokenize_path(path: Path) -> List[str]:
    text = " ".join(path.parts[:-1] + (path.stem,))
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _normalize_sample_id(path: Path) -> str:
    tokens = [
        token
        for token in _tokenize_path(path)
        if token not in MASK_HINTS and token not in SATELLITE_HINTS
    ]
    return "_".join(tokens)


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_mask_path(path: Path) -> bool:
    return any(token in MASK_HINTS for token in _tokenize_path(path))


def _prefer_satellite_image(paths: Sequence[Path]) -> Path:
    return sorted(
        paths,
        key=lambda path: (
            0 if any(token in SATELLITE_HINTS for token in _tokenize_path(path)) else 1,
            len(path.parts),
            path.name,
        ),
    )[0]


def _build_grouped_index(files: Sequence[Path]) -> Dict[str, List[Path]]:
    grouped = {}
    for path in files:
        sample_id = _normalize_sample_id(path)
        grouped.setdefault(sample_id, []).append(path)
    return grouped


def discover_image_mask_pairs(dataset_root: Path) -> List[ImageMaskPair]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    files = sorted(path for path in dataset_root.rglob("*") if _is_supported_image(path))
    mask_files = [path for path in files if _is_mask_path(path)]
    image_files = [path for path in files if not _is_mask_path(path)]

    image_map = _build_grouped_index(image_files)
    mask_map = _build_grouped_index(mask_files)

    pairs = []
    for sample_id, candidate_masks in sorted(mask_map.items()):
        candidate_images = image_map.get(sample_id, [])
        if not candidate_images:
            continue

        image_path = _prefer_satellite_image(candidate_images)
        mask_path = sorted(candidate_masks, key=lambda path: (len(path.parts), path.name))[0]
        pairs.append(
            ImageMaskPair(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    if not pairs:
        raise ValueError(
            "No image/mask pairs were found. Confirm the dataset is extracted and "
            "that mask files contain names like 'mask', 'label', or 'gt'."
        )

    return sorted(pairs, key=lambda pair: pair.sample_id)


def _mask_positive_fraction(mask_array: np.ndarray) -> float:
    mask_array = np.asarray(mask_array)
    return float((mask_array > 0).mean())


def _mask_unique_values(mask_array: np.ndarray, max_values: int = 10) -> List[int]:
    unique_values = np.unique(mask_array)
    clipped = unique_values[:max_values]
    return [int(value) for value in clipped.tolist()]


def _mask_is_binary(mask_array: np.ndarray) -> bool:
    unique_values = set(np.unique(mask_array).tolist())
    return unique_values.issubset({0, 1, 255})


def build_sample_table(
    pairs: Sequence[ImageMaskPair],
    max_samples: int | None = None,
) -> pd.DataFrame:
    selected_pairs = list(pairs[:max_samples]) if max_samples is not None else list(pairs)
    rows = []

    for pair in selected_pairs:
        with Image.open(pair.image_path) as image:
            image_array = np.array(image)
            image_width, image_height = image.size
            image_mode = image.mode

        with Image.open(pair.mask_path) as mask:
            mask_array = np.array(mask)
            mask_width, mask_height = mask.size
            mask_mode = mask.mode

        rows.append(
            {
                "sample_id": pair.sample_id,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "image_width": image_width,
                "image_height": image_height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "image_mode": image_mode,
                "mask_mode": mask_mode,
                "size_matches": (image_width, image_height) == (mask_width, mask_height),
                "road_coverage_pct": _mask_positive_fraction(mask_array) * 100.0,
                "mask_unique_values": _mask_unique_values(mask_array),
                "mask_is_binary": _mask_is_binary(mask_array),
                "image_channels": 1 if image_array.ndim == 2 else int(image_array.shape[2]),
            }
        )

    return pd.DataFrame(rows)


def compute_dataset_overview(sample_table: pd.DataFrame) -> pd.Series:
    if sample_table.empty:
        raise ValueError("Sample table is empty.")

    coverage = sample_table["road_coverage_pct"]
    return pd.Series(
        {
            "num_samples": int(len(sample_table)),
            "num_images_with_matching_mask_size": int(sample_table["size_matches"].sum()),
            "num_binary_masks": int(sample_table["mask_is_binary"].sum()),
            "num_empty_masks": int((coverage == 0).sum()),
            "mean_road_coverage_pct": float(coverage.mean()),
            "median_road_coverage_pct": float(coverage.median()),
            "p95_road_coverage_pct": float(coverage.quantile(0.95)),
            "min_image_width": int(sample_table["image_width"].min()),
            "max_image_width": int(sample_table["image_width"].max()),
            "min_image_height": int(sample_table["image_height"].min()),
            "max_image_height": int(sample_table["image_height"].max()),
        }
    )
