"""Regression & feature tests for final Quang notebook integration.

The default pipeline stays unchanged. Watershed is deliberately opt-in.
"""
import numpy as np
import cv2
import pytest
from core.pipeline import PipelineConfig, process_image
from processing.morphology_module import (
    manual_morphology, binarize_image, create_kernel, separate_touching_objects
)


def touching_circles():
    im = np.zeros((220, 300, 3), dtype=np.uint8)
    cv2.circle(im, (115, 110), 45, (255, 255, 255), -1)
    cv2.circle(im, (205, 110), 45, (255, 255, 255), -1)
    return im


@pytest.mark.parametrize('shape', ['rect', 'ellipse', 'cross'])
@pytest.mark.parametrize('size', [3, 5, 7])
@pytest.mark.parametrize('mode', ['erosion', 'dilation'])
def test_final_manual_morphology_matches_opencv(shape, size, mode):
    # Include border pixels, where old Scratch erosion was subtly different.
    rng = np.random.default_rng(2026)
    binary = rng.choice([0, 255], size=(30, 32)).astype(np.uint8)
    k = create_kernel(shape, size)
    expected = cv2.erode(binary, k) if mode == 'erosion' else cv2.dilate(binary, k)
    assert np.array_equal(manual_morphology(binary, k, mode), expected)


def test_quang_binarization_auto_polarity_can_be_requested_explicitly():
    bright_background = np.full((70, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(bright_background, (10, 10), (24, 24), (0, 0, 0), -1)
    inverted = binarize_image(bright_background, auto_otsu=True, auto_polarity=True)
    assert inverted[0, 0] == 0
    assert inverted[15, 15] == 255
    # Backwards compatible: the historical manual threshold still applies.
    raw = binarize_image(bright_background, thresh_val=127)
    assert raw[0, 0] == 255


def test_watershed_disabled_by_default_and_old_ccl_preserved():
    default = PipelineConfig()
    assert default.use_watershed is False
    result = process_image(touching_circles(), PipelineConfig(threshold_method='manual'))
    assert result['used_watershed'] is False
    assert result['raw_count'] == 1
    assert result['count'] == 1
    assert result['watershed_markers'] is None
    assert result['label_method'] == 'connected_components'


def test_optional_watershed_separates_touching_shapes_then_uses_minh_anh_stats():
    src = touching_circles()
    original = src.copy()
    result = process_image(src, PipelineConfig(threshold_method='manual', use_watershed=True,
                                               min_area=100, distance_ratio=0.35))
    assert result['countable']
    assert result['used_watershed'] is True
    assert result['label_method'] == 'watershed'
    assert result['raw_count'] == 2
    assert result['count'] == 2
    assert len(result['stats']['objects']) == 2
    assert result['watershed_markers'].dtype == np.int32
    assert result['labels'].min() == 0
    assert sorted(np.unique(result['labels'])) == [0, 1, 2]
    assert result['count_mask'].shape == src.shape[:2]
    assert result['annotated_bgr'].shape == src.shape
    assert np.array_equal(src, original)


def test_watershed_handles_empty_and_does_not_count_feature_operations():
    empty = np.zeros((100, 120, 3), dtype=np.uint8)
    result = process_image(empty, PipelineConfig(threshold_method='manual', use_watershed=True))
    assert result['raw_count'] == 0
    assert result['count'] == 0
    assert np.count_nonzero(result['count_mask']) == 0
    feature = process_image(touching_circles(), PipelineConfig(operation='gradient', use_watershed=True))
    assert feature['countable'] is False
    assert feature['used_watershed'] is False
    assert feature['count'] is None


def test_watershed_respects_area_filter_and_config_validation():
    r = process_image(touching_circles(), PipelineConfig(threshold_method='manual',use_watershed=True,
                                                        min_area=100_000))
    assert r['raw_count'] == 2
    assert r['count'] == 0
    assert r['stats']['objects'] == []
    with pytest.raises(ValueError):
        PipelineConfig(distance_ratio=1.5).validate()


def test_quang_watershed_returns_marker_convention():
    gray = cv2.cvtColor(touching_circles(), cv2.COLOR_BGR2GRAY)
    mask, markers = separate_touching_objects(gray, distance_ratio=0.35)
    assert mask.dtype == np.uint8 and markers.dtype == np.int32
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.any(markers == 1)
    assert np.any(markers == -1)
    assert len(np.unique(markers[markers > 1])) == 2
