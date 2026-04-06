"""Experiment configuration with YAML loading and validation."""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    arch: str = "Unet"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = 3
    classes: int = 1
    decoder_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    dataset_dir: Optional[str] = None
    image_size: int = 512
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    val_split_ratio: float = 0.15
    val_split_seed: int = 42
    subset_size: Optional[int] = None


@dataclass
class AugmentationStep:
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    train: List[AugmentationStep] = field(default_factory=list)
    val: List[AugmentationStep] = field(default_factory=list)


@dataclass
class NormalizationConfig:
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class LossConfig:
    type: str = "bce_dice"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "from_logits": True,
        "smooth": 1.0,
    })


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    params: Dict[str, Any] = field(default_factory=lambda: {"eta_min": 1e-6})


@dataclass
class TrainingConfig:
    epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_iou"
    early_stopping_mode: str = "max"
    grad_accumulation_steps: int = 1
    mixed_precision: bool = True
    freeze_encoder_epochs: int = 5
    encoder_lr_factor: float = 0.1
    ema: bool = False
    ema_decay: float = 0.999


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_last: bool = True
    resume_from: Optional[str] = None


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    experiment_name: Optional[str] = None
    log_every_n_steps: int = 10
    save_visualizations_every_n_epochs: int = 5
    num_visualization_samples: int = 6
    tensorboard: bool = False
    wandb: bool = False
    save_training_curves: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_SECTION_CLASSES = {
    "model": ModelConfig,
    "data": DataConfig,
    "augmentations": AugmentationConfig,
    "normalization": NormalizationConfig,
    "loss": LossConfig,
    "optimizer": OptimizerConfig,
    "scheduler": SchedulerConfig,
    "training": TrainingConfig,
    "checkpoint": CheckpointConfig,
    "logging": LoggingConfig,
}


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    """Recursively convert a dict into the target dataclass."""
    if not isinstance(data, dict):
        return data

    # Special handling for AugmentationConfig — parse list of steps
    if cls is AugmentationConfig:
        train_steps = [
            AugmentationStep(**step) if isinstance(step, dict) else step
            for step in data.get("train", [])
        ]
        val_steps = [
            AugmentationStep(**step) if isinstance(step, dict) else step
            for step in data.get("val", [])
        ]
        return AugmentationConfig(train=train_steps, val=val_steps)

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: Path) -> ExperimentConfig:
    """Load a YAML config file and return a validated ExperimentConfig.

    Missing sections/fields use defaults from the dataclass definitions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = ExperimentConfig()

    # Top-level scalar
    if "seed" in raw:
        config.seed = int(raw["seed"])

    # Nested sections
    for section_name, section_cls in _SECTION_CLASSES.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            setattr(config, section_name, _dict_to_dataclass(section_cls, raw[section_name]))

    # Auto-generate experiment name if not set
    if config.logging.experiment_name is None:
        config.logging.experiment_name = _generate_experiment_name(config)

    validate_config(config)
    return config


def _generate_experiment_name(config: ExperimentConfig) -> str:
    """Generate a name like 'Unet_resnet34_20260406_143022'."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.model.arch}_{config.model.encoder_name}_{timestamp}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def config_to_dict(config: ExperimentConfig) -> dict:
    """Serialize config to a plain dict (for checkpoint saving and YAML export)."""
    return asdict(config)


def save_config(config: ExperimentConfig, path: Path) -> None:
    """Save config as a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_LOSS_TYPES = {"bce_dice", "focal_dice", "bce_jaccard", "tversky", "dice", "focal_tversky", "boundary_bce_dice"}
_VALID_SCHEDULER_TYPES = {"cosine", "cosine_warm_restarts", "step", "plateau", "one_cycle"}
_VALID_OPTIMIZER_TYPES = {"adamw", "sgd", "adam"}


def validate_config(config: ExperimentConfig) -> None:
    """Validate config consistency. Raises ValueError on critical issues."""
    errors = []

    if config.loss.type not in _VALID_LOSS_TYPES:
        errors.append(f"Unknown loss type '{config.loss.type}'. Valid: {_VALID_LOSS_TYPES}")

    if config.scheduler.type not in _VALID_SCHEDULER_TYPES:
        errors.append(f"Unknown scheduler '{config.scheduler.type}'. Valid: {_VALID_SCHEDULER_TYPES}")

    if config.optimizer.type not in _VALID_OPTIMIZER_TYPES:
        errors.append(f"Unknown optimizer '{config.optimizer.type}'. Valid: {_VALID_OPTIMIZER_TYPES}")

    if config.training.freeze_encoder_epochs >= config.training.epochs:
        errors.append(
            f"freeze_encoder_epochs ({config.training.freeze_encoder_epochs}) "
            f"must be < epochs ({config.training.epochs})"
        )

    if config.data.val_split_ratio <= 0 or config.data.val_split_ratio >= 1:
        errors.append(f"val_split_ratio must be in (0, 1), got {config.data.val_split_ratio}")

    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# CLI override support
# ---------------------------------------------------------------------------


def apply_overrides(config: ExperimentConfig, overrides: List[str]) -> None:
    """Apply dot-notation overrides to a config.

    Example: apply_overrides(config, ["training.epochs=10", "data.batch_size=4"])
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: '{override}'")

        key, value_str = override.split("=", 1)
        parts = key.strip().split(".")

        # Navigate to the parent object
        obj = config
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise ValueError(f"Unknown config path: '{key}' (no attribute '{part}')")
            obj = getattr(obj, part)

        attr_name = parts[-1]
        if not hasattr(obj, attr_name):
            raise ValueError(f"Unknown config path: '{key}' (no attribute '{attr_name}')")

        # Coerce value to the existing field type
        current_value = getattr(obj, attr_name)
        coerced = _coerce_value(value_str, current_value)
        setattr(obj, attr_name, coerced)


def _coerce_value(value_str: str, current_value: Any) -> Any:
    """Coerce a string value to match the type of the current value."""
    if current_value is None:
        # Try int, float, then keep as string
        for cast in (int, float):
            try:
                return cast(value_str)
            except (ValueError, TypeError):
                continue
        if value_str.lower() == "null" or value_str.lower() == "none":
            return None
        return value_str

    if isinstance(current_value, bool):
        return value_str.lower() in ("true", "1", "yes")
    if isinstance(current_value, int):
        return int(value_str)
    if isinstance(current_value, float):
        return float(value_str)
    return value_str
