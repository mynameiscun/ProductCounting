"""Test backend độc lập GUI, dùng pytest từ thư mục gốc."""
import cv2
import numpy as np
import pytest
from core.pipeline import PipelineConfig, process_image, imread_unicode, build_binary
from processing.connected_components_module import run_connected_components, same_partition


def make_test_image():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(image, (40,60), 15, (255,255,255), thickness=-1)
    cv2.circle(image, (115,60), 15, (255,255,255), thickness=-1)
    return image


def test_opening_rect_3x3_count():
    result = process_image(make_test_image(), PipelineConfig(threshold_method="manual"))
    assert result["raw_count"] == 2
    assert result["count"] is None  # Chưa có module statistics của Minh Anh
    assert result["labels"].dtype == np.int32
    assert result["colored_rgb"].shape == (120, 160, 3)
    assert set(np.unique(result["count_mask"])).issubset({0,255})


def test_scratch_vs_opencv_partition():
    img = make_test_image()
    mask = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for conn in (4,8):
        n1,a = run_connected_components(mask,conn,"opencv")
        n2,b = run_connected_components(mask,conn,"scratch")
        assert n1 == n2 == 3
        assert same_partition(a,b)


@pytest.mark.parametrize("op",["gradient","top_hat","black_hat"])
def test_feature_ops_not_miscounted(op):
    res=process_image(make_test_image(),PipelineConfig(operation=op))
    assert res["countable"] is False
    assert res["raw_count"] is None
    assert res["count"] is None
    assert res["labels"] is None


@pytest.mark.parametrize("mode",["binary","gray"])
@pytest.mark.parametrize("thresh",["manual","otsu","adaptive"])
def test_threshold_and_source_modes(mode,thresh):
    res=process_image(make_test_image(),PipelineConfig(morphology_input=mode,threshold_method=thresh))
    assert res["countable"] is True
    assert res["count_mask"].dtype == np.uint8
    assert set(np.unique(res["count_mask"])).issubset({0,255})


def test_config_guards():
    with pytest.raises(ValueError):
        PipelineConfig(kernel_size=4).validate()
    with pytest.raises(ValueError):
        PipelineConfig(min_area=500,max_area=10).validate()


def test_future_statistics_module_integration(monkeypatch):
    """Kiểm chứng trước hợp đồng nhận module Minh Anh, không đặt stub vào project thật."""
    import sys
    from types import ModuleType
    fake = ModuleType("processing.statistics_module")

    def analyze_objects(labels, image_bgr, min_area=100, max_area=None):
        assert labels.ndim == 2 and labels.dtype == np.int32
        assert min_area == 50 and max_area == 2000
        return {"count": 2, "objects": [], "annotated_image": image_bgr.copy()}

    fake.analyze_objects = analyze_objects
    monkeypatch.setitem(sys.modules,"processing.statistics_module",fake)
    config = PipelineConfig(min_area=50, max_area=2000, threshold_method="manual")
    result = process_image(make_test_image(), config)
    assert result["count"] == 2
    assert result["annotated_bgr"].shape == (120,160,3)


def test_unicode_file_io(tmp_path):
    from core.pipeline import imwrite_unicode
    out = tmp_path / "ảnh đếm.png"
    original = make_test_image()
    imwrite_unicode(out, original)
    restored = imread_unicode(out)
    assert np.array_equal(restored,original)


def test_auto_threshold_label_matches_actual_mask_after_grayscale_morphology():
    """Giá trị T hiển thị phải tương ứng với mask SAU Morphology ảnh xám."""
    gray = np.tile(np.arange(80, dtype=np.uint8) * 3, (80, 1))
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cfg = PipelineConfig(morphology_input="gray", threshold_method="otsu", operation="opening")
    result = process_image(image, cfg)
    effective_t = float(result["count_threshold_info"].split("T=")[-1])
    _, expected = cv2.threshold(result["morphology"], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    assert np.array_equal(result["count_mask"], expected)
    actual_t, _ = cv2.threshold(result["morphology"], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    assert effective_t == round(actual_t)


def test_manual_threshold_metadata_remains_manual():
    cfg = PipelineConfig(threshold_method="manual", threshold_value=141)
    result = process_image(make_test_image(), cfg)
    assert result["threshold_info"] == "Manual T=141"
    assert result["count_threshold_info"] == "Manual T=141"


def test_adaptive_threshold_metadata_does_not_invent_global_t():
    cfg = PipelineConfig(threshold_method="adaptive", morphology_input="gray")
    result = process_image(make_test_image(), cfg)
    assert result["count_threshold_info"] == "Adaptive Gaussian 11 / C=2"
