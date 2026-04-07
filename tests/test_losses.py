"""Tests for loss functions."""

from __future__ import annotations

import pytest
import torch

from road_segmentation.training.losses import (
    BoundaryWeightedBCEDiceLoss,
    create_loss,
)


class TestCreateLoss:
    @pytest.mark.parametrize("loss_type", [
        "bce_dice", "focal_dice", "bce_jaccard", "dice", "tversky",
        "focal_tversky", "boundary_bce_dice",
    ])
    def test_all_loss_types_construct(self, loss_type):
        loss = create_loss(loss_type, {"from_logits": True})
        assert loss is not None

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss("nonexistent", {})

    @pytest.mark.parametrize("loss_type", [
        "bce_dice", "focal_dice", "boundary_bce_dice",
    ])
    def test_loss_forward_pass(self, loss_type):
        loss_fn = create_loss(loss_type, {"from_logits": True})
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_loss_decreases_with_better_prediction(self):
        loss_fn = create_loss("bce_dice", {"from_logits": True})
        target = torch.ones(1, 1, 64, 64)

        # Bad prediction (all negative logits)
        bad_pred = torch.full((1, 1, 64, 64), -5.0)
        loss_bad = loss_fn(bad_pred, target)

        # Good prediction (all positive logits)
        good_pred = torch.full((1, 1, 64, 64), 5.0)
        loss_good = loss_fn(good_pred, target)

        assert loss_good < loss_bad


class TestBoundaryLoss:
    def test_boundary_weights_shape(self):
        loss_fn = BoundaryWeightedBCEDiceLoss()
        target = torch.zeros(2, 1, 64, 64)
        target[:, :, 20:25, 10:50] = 1.0  # road stripe
        weights = loss_fn._compute_boundary_weights(target)
        assert weights.shape == target.shape

    def test_boundary_weights_higher_at_edges(self):
        loss_fn = BoundaryWeightedBCEDiceLoss(edge_weight=5.0, boundary_width=3)
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 20:40, 20:40] = 1.0  # square block
        weights = loss_fn._compute_boundary_weights(target)

        # Interior pixels should have weight 1.0
        assert weights[0, 0, 30, 30].item() == pytest.approx(1.0)
        # Edge pixels should have weight > 1.0
        assert weights[0, 0, 20, 30].item() > 1.0

    def test_boundary_loss_forward(self):
        loss_fn = BoundaryWeightedBCEDiceLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)
