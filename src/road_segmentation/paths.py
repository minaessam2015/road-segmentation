from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DEEPGLOBE_DATASET_NAME = "deepglobe-road-extraction-dataset"
DEEPGLOBE_DATASET_DIR = RAW_DATA_DIR / DEEPGLOBE_DATASET_NAME
