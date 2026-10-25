"""Luồng xử lý trung tâm: Quang -> Linh -> (tùy chọn) Minh Anh.

Không gọi GUI/tkinter trong module này: chạy độc lập bằng script, test hoặc GUI.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib
import time
from typing import Any
import cv2
import numpy as np
from processing.morphology_module import create_kernel, apply_morphology, binarize_image, separate_touching_objects
from processing.connected_components_module import run_connected_components, colorize_labels

REGION_OPERATIONS = ("erosion", "dilation", "opening", "closing")
FEATURE_OPERATIONS = ("gradient", "top_hat", "black_hat")
ALL_OPERATIONS = REGION_OPERATIONS + FEATURE_OPERATIONS

@dataclass(frozen=True)
class PipelineConfig:
    operation: str = "opening"
    shape: str = "ellipse"
    kernel_size: int = 3
    iterations: int = 1
    morphology_input: str = "binary"  # binary | gray
    threshold_method: str = "otsu"  # manual | otsu | adaptive
    threshold_value: int = 127
    invert: bool = False
    connectivity: int = 8
    algorithm: str = "opencv"  # opencv | scratch
    min_area: int = 100
    max_area: int | None = None
    use_watershed: bool = False   # Mặc định TẮT, bảo toàn pipeline cũ
    distance_ratio: float = 0.35  # Ngưỡng lõi marker từ Quang

    def validate(self) -> None:
        if self.operation not in ALL_OPERATIONS:
            raise ValueError(f"Phép Morphology không hợp lệ: {self.operation}")
        if self.shape not in ("rect", "ellipse", "cross"):
            raise ValueError("Kernel shape phải là rect, ellipse hoặc cross")
        if self.kernel_size not in (3, 5, 7, 9):
            raise ValueError("Kernel size phải là 3/5/7/9")
        if not 1 <= self.iterations <= 5:
            raise ValueError("Iterations phải nằm trong khoảng 1..5")
        if self.morphology_input not in ("binary", "gray"):
            raise ValueError("Input Morphology phải là binary hoặc gray")
        if self.threshold_method not in ("manual", "otsu", "adaptive"):
            raise ValueError("Threshold mode không hợp lệ")
        if not 0 <= self.threshold_value <= 255:
            raise ValueError("Threshold phải nằm trong 0..255")
        if self.connectivity not in (4, 8):
            raise ValueError("Connectivity phải là 4 hoặc 8")
        if self.algorithm not in ("opencv", "scratch"):
            raise ValueError("Algorithm phải là opencv hoặc scratch")
        if not 0.05 <= self.distance_ratio <= 0.95:
            raise ValueError("Distance Ratio phải trong khoảng 0.05..0.95")
        if self.min_area < 0:
            raise ValueError("Min Area không được âm")
        if self.max_area is not None and self.max_area < self.min_area:
            raise ValueError("Max Area không được nhỏ hơn Min Area")


def build_binary(image_bgr: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """BGR -> gray và mask 0/255. Manual thường gọi chính hàm của Quang."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if cfg.threshold_method == "manual":
        if not cfg.invert:
            binary = binarize_image(image_bgr, cfg.threshold_value)  # Quang
        else:
            _, binary = cv2.threshold(gray, cfg.threshold_value, 255, cv2.THRESH_BINARY_INV)
        details = f"Manual T={cfg.threshold_value}"
    elif cfg.threshold_method == "otsu":
        flag = cv2.THRESH_BINARY_INV if cfg.invert else cv2.THRESH_BINARY
        threshold, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
        details = f"Otsu T={threshold:.0f}"
    else:
        mode = cv2.THRESH_BINARY_INV if cfg.invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       mode, blockSize=11, C=2)
        details = "Adaptive Gaussian 11 / C=2"
    return gray, binary, details


def _threshold_grayscale(gray: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, str]:
    """Gray Morphology -> mask cho CCL; trả kèm T THỰC dùng trên ảnh sau Morphology."""
    if cfg.threshold_method == "manual":
        code = cv2.THRESH_BINARY_INV if cfg.invert else cv2.THRESH_BINARY
        _, out = cv2.threshold(gray, cfg.threshold_value, 255, code)
        details = f"Manual T={cfg.threshold_value}"
    elif cfg.threshold_method == "otsu":
        code = cv2.THRESH_BINARY_INV if cfg.invert else cv2.THRESH_BINARY
        threshold, out = cv2.threshold(gray, 0, 255, code | cv2.THRESH_OTSU)
        details = f"Otsu T={threshold:.0f}"
    else:
        code = cv2.THRESH_BINARY_INV if cfg.invert else cv2.THRESH_BINARY
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    code, blockSize=11, C=2)
        details = "Adaptive Gaussian 11 / C=2"
    return out, details


def _optional_statistics(labels: np.ndarray, image_bgr: np.ndarray,
                         cfg: PipelineConfig) -> dict[str, Any] | None:
    """Chỉ gọi module thật của Minh Anh nếu có, không tạo số đếm giả."""
    try:
        mod = importlib.import_module("processing.statistics_module")
    except ModuleNotFoundError as e:
        if e.name == "processing.statistics_module":
            return None
        raise
    result = mod.analyze_objects(
        labels=labels,
        image_bgr=image_bgr,
        min_area=cfg.min_area,
        max_area=cfg.max_area,
    )
    if not isinstance(result, dict) or not all(k in result for k in ("count", "objects", "annotated_image")):
        raise ValueError("statistics_module.analyze_objects phải trả dict chứa count, objects, annotated_image")
    if result["annotated_image"].shape[:2] != image_bgr.shape[:2]:
        raise ValueError("Ảnh annotate từ Minh Anh phải cùng H×W với ảnh gốc")
    return result


def process_image(image_bgr: np.ndarray, cfg: PipelineConfig) -> dict[str, Any]:
    """Ảnh BGR -> ảnh trung gian -> labels nếu phép tạo vùng -> thống kê nếu có."""
    cfg.validate()
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("Đầu vào phải là ảnh BGR uint8 3 kênh")
    h, w = image_bgr.shape[:2]
    if cfg.algorithm == "scratch" and h*w > 1_000_000:
        raise ValueError("Scratch BFS giới hạn ảnh <= 1.000.000 pixel trong GUI; dùng OpenCV với ảnh lớn")

    times: dict[str, float] = {}
    t = time.perf_counter()
    gray, binary, threshold_info = build_binary(image_bgr, cfg)
    times["threshold_ms"] = 1000 * (time.perf_counter()-t)

    t = time.perf_counter()
    kernel = create_kernel(cfg.shape, cfg.kernel_size)  # Quang
    morphology_input = binary if cfg.morphology_input == "binary" else gray
    morph = apply_morphology(morphology_input, cfg.operation, kernel, cfg.iterations)  # Quang
    times["morphology_ms"] = 1000 * (time.perf_counter()-t)

    # Gradient / top-hat / black-hat cho ảnh đặc trưng, không phải segmentation mask
    countable = cfg.operation in REGION_OPERATIONS
    note = ""
    count_threshold_info = threshold_info  # Ngưỡng của mask đang HIỂN THỊ / đem vào CCL
    if countable:
        if cfg.morphology_input == "binary":
            mask_for_ccl = np.where(morph > 0, 255, 0).astype(np.uint8)
        else:
            mask_for_ccl, count_threshold_info = _threshold_grayscale(morph, cfg)
            note = "Morphology trên ảnh xám → cắt ngưỡng lại trước Connected Components."
    else:
        mask_for_ccl = None
        note = "Gradient / Top Hat / Black Hat là ảnh đặc trưng, không dùng trực tiếp để đếm vùng sản phẩm."

    stats = None
    n_labels = None
    labels = None
    colored = None
    watershed_markers = None
    used_watershed = bool(countable and cfg.use_watershed)
    times["watershed_ms"] = 0.0
    t = time.perf_counter()
    if countable:
        if used_watershed:
            # Watershed trả marker -1 (biên), 1 (nền), >1 (vật thể).
            # KHÔNG dùng CCL trên mặt nạ này: kết nối chéo có thể gộp trở lại
            # các vật thể Watershed vừa tách. Giữ nhãn Watershed và chuẩn hóa
            # về 0 = nền, 1..N = vùng để Minh Anh tính thống kê chính xác.
            separated, watershed_markers = separate_touching_objects(
                mask_for_ccl, distance_ratio=cfg.distance_ratio
            )
            times["watershed_ms"] = 1000 * (time.perf_counter() - t)
            mask_for_ccl = separated
            marker_ids = np.unique(watershed_markers[watershed_markers > 1])
            labels = np.zeros(watershed_markers.shape, dtype=np.int32)
            for new_id, marker_id in enumerate(marker_ids, start=1):
                labels[watershed_markers == marker_id] = new_id
            n_labels = len(marker_ids) + 1  # nền là nhãn 0
            colored = colorize_labels(labels)
            times["ccl_ms"] = 0.0
            note = "Các vùng đang được tách và gán nhãn bằng Watershed."
        else:
            n_labels, labels = run_connected_components(
                mask_for_ccl, cfg.connectivity, cfg.algorithm
            )  # Linh: CCL OpenCV/Scratch giữ nguyên
            colored = colorize_labels(labels)
            times["ccl_ms"] = 1000 * (time.perf_counter() - t)
        t = time.perf_counter()
        stats = _optional_statistics(labels, image_bgr, cfg)  # Minh Anh giữ nguyên
        times["stats_ms"] = 1000 * (time.perf_counter() - t)
    else:
        times["ccl_ms"] = 0.0
        times["stats_ms"] = 0.0
    times["total_ms"] = sum(times.values())
    return {
        "original_bgr": image_bgr,
        "gray": gray,
        "binary": binary,
        "morphology": morph,
        "count_mask": mask_for_ccl,
        "watershed_markers": watershed_markers,
        "used_watershed": used_watershed,
        "label_method": "watershed" if used_watershed else "connected_components",
        "colored_rgb": colored,
        "labels": labels,
        "num_labels": n_labels,
        "raw_count": None if n_labels is None else n_labels - 1,
        "stats": stats,
        "count": None if stats is None else stats["count"],
        "annotated_bgr": None if stats is None else stats["annotated_image"],
        "countable": countable,
        "threshold_info": threshold_info,
        "count_threshold_info": count_threshold_info,
        "note": note,
        "times": times,
        "config": cfg,
    }


def imread_unicode(path: str | Path) -> np.ndarray:
    """Đọc cả đường dẫn có dấu tiếng Việt trên Windows bằng imdecode."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def imwrite_unicode(path: str | Path, bgr: np.ndarray) -> None:
    ext = Path(path).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        raise ValueError("Định dạng lưu hỗ trợ: .png .jpg .jpeg .bmp .webp")
    ok, encoded = cv2.imencode(ext, bgr)
    if not ok:
        raise ValueError("Không mã hóa được ảnh đầu ra")
    encoded.tofile(str(path))
