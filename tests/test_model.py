"""Tests for model creation and forward pass."""

from __future__ import annotations

import pytest
import torch

from road_segmentation.models.factory import (
    count_parameters,
    create_model,
    freeze_encoder,
    get_decoder_parameters,
    get_encoder_parameters,
    unfreeze_encoder,
)


class TestCreateModel:
    def test_unet_resnet34(self):
        model = create_model("Unet", "resnet34", encoder_weights=None, classes=1)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_deeplabv3plus(self):
        model = create_model("DeepLabV3Plus", "resnet34", encoder_weights=None, classes=1)
        model.eval()  # BatchNorm needs eval mode with batch_size=1
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_linknet(self):
        model = create_model("Linknet", "resnet34", encoder_weights=None, classes=1)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_output_is_logits(self):
        """Output should be raw logits (not sigmoid), as required by BCE loss."""
        model = create_model("Unet", "resnet18", encoder_weights=None, classes=1)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        # Logits can be any real number, not constrained to [0, 1]
        assert out.min() < 0 or out.max() > 1  # at least one value outside [0,1]

    def test_invalid_arch_raises(self):
        with pytest.raises(Exception):
            create_model("NonExistentArch", "resnet34")


class TestEncoderManagement:
    def test_freeze_encoder(self, sample_model):
        freeze_encoder(sample_model)
        for p in sample_model.encoder.parameters():
            assert not p.requires_grad

    def test_unfreeze_encoder(self, sample_model):
        freeze_encoder(sample_model)
        unfreeze_encoder(sample_model)
        for p in sample_model.encoder.parameters():
            assert p.requires_grad

    def test_get_encoder_decoder_params(self, sample_model):
        enc_params = get_encoder_parameters(sample_model)
        dec_params = get_decoder_parameters(sample_model)
        assert len(enc_params) > 0
        assert len(dec_params) > 0
        # No overlap
        enc_ids = {id(p) for p in enc_params}
        dec_ids = {id(p) for p in dec_params}
        assert enc_ids.isdisjoint(dec_ids)

    def test_count_parameters(self, sample_model):
        counts = count_parameters(sample_model)
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]

        freeze_encoder(sample_model)
        counts_frozen = count_parameters(sample_model)
        assert counts_frozen["trainable"] < counts_frozen["total"]
