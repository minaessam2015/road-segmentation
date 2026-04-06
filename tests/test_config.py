"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml

from road_segmentation.config import (
    ExperimentConfig,
    apply_overrides,
    load_config,
    validate_config,
)


@pytest.fixture
def config_file(tmp_path):
    """Write a minimal valid config YAML."""
    config = {
        "seed": 42,
        "model": {"arch": "Unet", "encoder_name": "resnet18"},
        "data": {"image_size": 256, "batch_size": 4},
        "loss": {"type": "bce_dice"},
        "training": {"epochs": 10, "freeze_encoder_epochs": 2},
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


class TestLoadConfig:
    def test_loads_valid_config(self, config_file):
        config = load_config(config_file)
        assert config.model.arch == "Unet"
        assert config.model.encoder_name == "resnet18"
        assert config.data.image_size == 256
        assert config.seed == 42

    def test_defaults_for_missing_fields(self, config_file):
        config = load_config(config_file)
        # These should be defaults since they're not in the YAML
        assert config.optimizer.type == "adamw"
        assert config.scheduler.type == "cosine"
        assert config.normalization.mean == [0.485, 0.456, 0.406]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_generates_experiment_name(self, config_file):
        config = load_config(config_file)
        assert config.logging.experiment_name is not None
        assert "Unet" in config.logging.experiment_name
        assert "resnet18" in config.logging.experiment_name


class TestValidation:
    def test_valid_config_passes(self):
        config = ExperimentConfig()
        validate_config(config)  # should not raise

    def test_invalid_loss_type(self):
        config = ExperimentConfig()
        config.loss.type = "fake_loss"
        with pytest.raises(ValueError, match="Unknown loss type"):
            validate_config(config)

    def test_freeze_epochs_exceeds_total(self):
        config = ExperimentConfig()
        config.training.epochs = 10
        config.training.freeze_encoder_epochs = 15
        with pytest.raises(ValueError, match="freeze_encoder_epochs"):
            validate_config(config)

    def test_invalid_val_ratio(self):
        config = ExperimentConfig()
        config.data.val_split_ratio = 0.0
        with pytest.raises(ValueError, match="val_split_ratio"):
            validate_config(config)


class TestOverrides:
    def test_simple_override(self, config_file):
        config = load_config(config_file)
        apply_overrides(config, ["training.epochs=20"])
        assert config.training.epochs == 20

    def test_float_override(self, config_file):
        config = load_config(config_file)
        apply_overrides(config, ["optimizer.lr=0.001"])
        assert config.optimizer.lr == 0.001

    def test_bool_override(self, config_file):
        config = load_config(config_file)
        apply_overrides(config, ["training.mixed_precision=false"])
        assert config.training.mixed_precision is False

    def test_invalid_key_raises(self, config_file):
        config = load_config(config_file)
        with pytest.raises(ValueError, match="Unknown config path"):
            apply_overrides(config, ["nonexistent.key=value"])
