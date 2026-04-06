"""Environment variable loading from .env file.

Call ``load_env()`` once at the start of any entrypoint (script, notebook,
API server) to load credentials from ``.env`` into ``os.environ``.

Supports three locations (checked in order):
1. PROJECT_ROOT/.env  (local development)
2. ~/.env              (user home fallback)
3. Already set in os.environ (Colab secrets, Docker, CI)
"""

from __future__ import annotations

import os
from pathlib import Path

from road_segmentation.paths import PROJECT_ROOT


def load_env() -> None:
    """Load .env file into os.environ if it exists."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # dotenv not installed — rely on existing env vars

    # Try project root first, then home directory
    for env_path in [PROJECT_ROOT / ".env", Path.home() / ".env"]:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return


def get_kaggle_credentials() -> dict:
    """Return Kaggle credentials from environment."""
    return {
        "username": os.environ.get("KAGGLE_USERNAME", ""),
        "key": os.environ.get("KAGGLE_KEY", ""),
    }


def get_wandb_config() -> dict:
    """Return W&B configuration from environment."""
    return {
        "api_key": os.environ.get("WANDB_API_KEY", ""),
        "project": os.environ.get("WANDB_PROJECT", "road-segmentation"),
        "entity": os.environ.get("WANDB_ENTITY", ""),
    }
