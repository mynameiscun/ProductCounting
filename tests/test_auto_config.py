"""Thử nghiệm tự chọn tham số theo ảnh, không phụ thuộc CustomTkinter."""
import numpy as np
import cv2
from core.pipeline import PipelineConfig, process_image
from core.auto_config import suggest_parameters


def scene():
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.circle(image, (55, 80), 24, (255, 255, 255), -1)
    cv2.circle(image, (165, 80), 24, (255, 255, 255), -1)
    return image


def test_auto_respects_pipeline_contract_and_keeps_image_unchanged():
    image = scene()
    orig = image.copy()
    cfg = PipelineConfig(min_area=33, max_area=8000, threshold_method="manual", kernel_size=9)
    chosen, diagnostics = suggest_parameters(image, cfg)
    chosen.validate()
    assert np.array_equal(orig, image)
    assert chosen.threshold_method == "otsu"
    assert chosen.algorithm == "opencv"
    assert chosen.min_area == 33 and chosen.max_area == 8000
    assert chosen.operation in ("opening", "closing")
    assert chosen.shape in ("ellipse", "rect", "cross")
    assert chosen.kernel_size in (3, 5, 7, 9)
    assert chosen.iterations in (1, 2)
    result = process_image(image, chosen)
    assert result["countable"]
    assert result["labels"].shape == image.shape[:2]
    assert diagnostics.reason


def test_auto_changes_operation_for_enclosed_holes():
    image = scene()
    cv2.circle(image, (55, 80), 8, (0, 0, 0), -1)
    cv2.circle(image, (165, 80), 8, (0, 0, 0), -1)
    chosen, _ = suggest_parameters(image, PipelineConfig())
    assert chosen.operation == "closing"


def test_auto_uses_opening_for_small_white_specks():
    image = scene()
    rng = np.random.default_rng(10)
    y = rng.integers(0, image.shape[0], 350)
    x = rng.integers(0, image.shape[1], 350)
    image[y, x] = 255
    chosen, _ = suggest_parameters(image, PipelineConfig())
    assert chosen.operation == "opening"


def test_auto_inverts_white_background():
    image = 255 - scene()
    chosen, _ = suggest_parameters(image, PipelineConfig())
    assert chosen.invert
    res = process_image(image, chosen)
    assert res["raw_count"] == 2


def test_suggest_is_repeatable():
    im = scene()
    a, details_a = suggest_parameters(im, PipelineConfig())
    b, details_b = suggest_parameters(im, PipelineConfig())
    assert a == b and details_a == details_b
