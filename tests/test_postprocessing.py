"""Tests for post-processing steps and pipeline."""

from __future__ import annotations

import numpy as np

from road_segmentation.postprocessing.pipeline import (
    PipelineConfig,
    ablation_configs,
    run_pipeline,
)
from road_segmentation.postprocessing.steps import (
    apply_threshold,
    find_optimal_threshold,
    morphological_close,
    morphological_open,
    remove_small_components,
    skeletonize_mask,
)


class TestThreshold:
    def test_apply_threshold(self):
        prob = np.array([[0.3, 0.7], [0.5, 0.9]], dtype=np.float32)
        mask = apply_threshold(prob, 0.5)
        expected = np.array([[0, 255], [255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_find_optimal_threshold(self, sample_prob_map, sample_mask_binary):
        best_t, results = find_optimal_threshold(
            [sample_prob_map], [sample_mask_binary],
            thresholds=np.arange(0.3, 0.8, 0.1),
        )
        assert 0.3 <= best_t <= 0.8
        assert len(results) == 5
        assert all(0 <= v <= 1 for v in results.values())


class TestComponentFilter:
    def test_removes_small_blobs(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:12, 10:12] = 255  # 4 pixels — too small
        mask[50:70, 50:70] = 255  # 400 pixels — keep
        result = remove_small_components(mask, min_area=50)
        assert result[11, 11] == 0  # small blob removed
        assert result[60, 60] == 255  # large blob kept

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = remove_small_components(mask, min_area=10)
        assert result.sum() == 0


class TestMorphology:
    def test_closing_fills_gaps(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Create a thicker road with a small gap
        mask[48:53, 20:40] = 255
        mask[48:53, 42:60] = 255  # 2-pixel gap in a 5px-wide road
        result = morphological_close(mask, kernel_size=5)
        assert result[50, 41] == 255  # gap should be filled

    def test_opening_removes_noise(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:70, 50:70] = 255
        mask[10, 10] = 255  # single pixel noise
        result = morphological_open(mask, kernel_size=3)
        assert result[10, 10] == 0  # noise removed
        assert result[60, 60] == 255  # road kept


class TestSkeletonize:
    def test_skeleton_is_thin(self, sample_mask_binary):
        skeleton = skeletonize_mask(sample_mask_binary)
        assert skeleton.dtype == np.uint8
        # Skeleton should have fewer pixels than original
        assert skeleton.sum() < sample_mask_binary.sum()
        # But still have some pixels
        assert skeleton.sum() > 0


class TestPipeline:
    def test_minimal_pipeline(self, sample_prob_map):
        config = PipelineConfig(
            threshold=0.5,
            remove_small_components=False,
            morphological_closing=False,
            morphological_opening=False,
            gap_bridging=False,
            skeletonize=False,
        )
        result = run_pipeline(sample_prob_map, config)
        assert result.mask.shape == sample_prob_map.shape
        assert result.mask.dtype == np.uint8

    def test_full_pipeline(self, sample_prob_map):
        config = PipelineConfig(
            threshold=0.5,
            remove_small_components=True,
            min_component_area=10,
            morphological_closing=True,
            closing_kernel_size=3,
            gap_bridging=False,
            skeletonize=False,
        )
        result = run_pipeline(sample_prob_map, config)
        assert result.mask.shape == sample_prob_map.shape
        assert len(result.steps_applied) >= 3

    def test_pipeline_collects_intermediates(self, sample_prob_map):
        config = PipelineConfig(
            threshold=0.5,
            remove_small_components=True,
            morphological_closing=True,
        )
        result = run_pipeline(sample_prob_map, config, collect_intermediates=True)
        assert len(result.intermediates) > 0
        for name, mask in result.intermediates.items():
            assert mask.shape == sample_prob_map.shape

    def test_ablation_configs_all_valid(self):
        configs = ablation_configs()
        assert len(configs) >= 5
        prob = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
        for name, config in configs.items():
            config.threshold = 0.5  # set a valid threshold
            result = run_pipeline(prob, config)
            assert result.mask.shape == prob.shape
