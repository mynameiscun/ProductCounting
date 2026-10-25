"""Kiểm chứng module Statistics thật của Minh Anh trong pipeline 3 thành viên."""
import cv2
import numpy as np
import pytest
from core.pipeline import PipelineConfig, process_image
from processing.statistics_module import analyze_objects


def synthetic_labels():
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[2:5, 3:7] = 7        # 12 px, bbox x=3 y=2 w=4 h=3
    labels[10:12, 20:22] = 13   # 4 px, bbox x=20 y=10 w=2 h=2
    return labels


def test_statistics_correct_area_centroid_bbox_and_unchanged_inputs():
    labels = synthetic_labels()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    orig_labels = labels.copy()
    orig_img = image.copy()
    result = analyze_objects(labels, image, min_area=0)
    assert result['count'] == 2
    assert [o['id'] for o in result['objects']] == [7, 13]
    large, small = result['objects']
    assert large['area'] == 12
    assert large['centroid'] == pytest.approx((4.5, 3.0))
    assert large['bbox'] == (3, 2, 4, 3)
    assert small['area'] == 4
    assert small['centroid'] == pytest.approx((20.5, 10.5))
    assert small['bbox'] == (20, 10, 2, 2)
    assert result['annotated_image'].shape == image.shape
    assert not np.array_equal(result['annotated_image'], image)
    assert np.array_equal(labels, orig_labels)
    assert np.array_equal(image, orig_img)


def test_statistics_area_range_inclusive():
    labels = synthetic_labels()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    for lo, hi, expected in [(0, None, 2), (4, 12, 2), (5, None, 1),
                             (12, 12, 1), (13, None, 0), (0, 3, 0)]:
        result = analyze_objects(labels, image, min_area=lo, max_area=hi)
        assert result['count'] == expected
        assert len(result['objects']) == expected


def test_no_object_empty_image():
    labels = np.zeros((16, 18), dtype=np.int32)
    image = np.zeros((16, 18, 3), dtype=np.uint8)
    result = analyze_objects(labels, image, min_area=100)
    assert result['count'] == 0
    assert result['objects'] == []
    assert np.array_equal(result['annotated_image'], image)


def make_img():
    image = np.zeros((130, 190, 3), dtype=np.uint8)
    cv2.circle(image, (48, 65), 16, (255, 255, 255), -1)
    cv2.circle(image, (145, 65), 16, (255, 255, 255), -1)
    return image


@pytest.mark.parametrize('method', ['opencv', 'scratch'])
def test_end_to_end_counting_and_annotations(method):
    image = make_img()
    before = image.copy()
    result = process_image(image, PipelineConfig(
        threshold_method='manual', threshold_value=127,
        algorithm=method, min_area=100, max_area=None))
    assert result['countable']
    assert result['raw_count'] == 2
    assert result['count'] == 2
    assert len(result['stats']['objects']) == 2
    assert result['annotated_bgr'].shape == image.shape
    assert not np.array_equal(result['annotated_bgr'], image)
    assert np.array_equal(image, before)


def test_end_to_end_filter_can_remove_all_and_preserve_raw_count():
    result = process_image(make_img(), PipelineConfig(
        threshold_method='manual', min_area=10000))
    assert result['raw_count'] == 2
    assert result['count'] == 0
    assert result['stats']['objects'] == []
    assert result['annotated_bgr'] is not None


@pytest.mark.parametrize('op', ['gradient', 'top_hat', 'black_hat'])
def test_feature_modes_do_not_use_statistical_counting(op):
    result = process_image(make_img(), PipelineConfig(operation=op))
    assert result['countable'] is False
    assert result['count'] is None
    assert result['stats'] is None
    assert result['annotated_bgr'] is None
