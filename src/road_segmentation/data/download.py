from pathlib import Path
import shutil
from typing import Optional

from road_segmentation.paths import DEEPGLOBE_DATASET_DIR, RAW_DATA_DIR


DATASET_HANDLE = "balraj98/deepglobe-road-extraction-dataset"


def _download_from_kagglehub() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the dataset. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def download_dataset(destination: Optional[Path] = None) -> Path:
    target_dir = Path(destination) if destination is not None else DEEPGLOBE_DATASET_DIR
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    source_dir = _download_from_kagglehub()

    if target_dir.exists():
        shutil.rmtree(target_dir)
    # I know this is a duplicate so le me decide this later
    shutil.copytree(source_dir, target_dir)
    return target_dir
