"""Gợi ý tham số theo ảnh bằng các quy tắc hình thái học có thể giải thích.

Không phải thuật toán tối ưu số đếm: không có ground truth hoặc module thống kê.
Chỉ tự chọn hướng mask, Opening/Closing, kernel và connectivity.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import cv2
import numpy as np
from core.pipeline import PipelineConfig
from processing.morphology_module import apply_morphology, create_kernel


@dataclass(frozen=True)
class AutoDiagnostics:
    reason: str
    changed: str
    input_noise_pixels: int
    input_hole_pixels: int
    recommendation: str = "Gợi ý dựa trên hình thái học, không bảo đảm đếm chính xác."


def _component_info(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Trả về pixel rác nhỏ, mask thành phần lớn, kích cỡ các vùng."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cutoff = max(8, int(mask.size * 0.0001))
    areas = stats[1:, cv2.CC_STAT_AREA]
    small_ids = np.where(areas < cutoff)[0] + 1
    large_ids = np.where(areas >= cutoff)[0] + 1
    noise_pixels = int(areas[small_ids - 1].sum()) if len(small_ids) else 0
    core = np.isin(labels, large_ids)
    return noise_pixels, core, areas


def _internal_holes(mask: np.ndarray) -> int:
    """Tổng diện tích những vùng tối được bao kín bên trong foreground."""
    inverted = cv2.bitwise_not(mask)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inverted, 8)
    if n <= 1:
        return 0
    height, width = mask.shape
    areas = []
    for x, y, w, h, area in stats[1:]:
        if x > 0 and y > 0 and x + w < width and y + h < height:
            areas.append(int(area))
    return sum(areas)


def _choose_polarity(gray: np.ndarray) -> bool:
    """Nếu vùng trắng chủ yếu là nền ở rìa ảnh thì đảo mask foreground."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    border_white = float(np.mean(border != 0))
    fg_ratio = float(np.mean(binary != 0))
    return border_white > 0.60 and fg_ratio > 0.35


def _connectivity_by_contact(mask: np.ndarray) -> int:
    """Ưu tiên 8; chọn 4 nếu 8 gộp >=2 vùng lớn chỉ tiếp xúc đường chéo.

    Không tự khẳng định các vùng đó là sản phẩm khác nhau.
    """
    _, lab4, stats4, _ = cv2.connectedComponentsWithStats(mask, 4)
    _, lab8, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    size_min = max(24, int(mask.size * 0.00045))
    sig4 = np.where(stats4[1:, cv2.CC_STAT_AREA] >= size_min)[0] + 1
    if len(sig4) < 2:
        return 8
    associations = {}
    for cid in sig4:
        vals = lab8[lab4 == cid]
        if vals.size == 0:
            continue
        destination = int(vals[0])
        associations[destination] = associations.get(destination, 0) + 1
    return 4 if any(count >= 2 for count in associations.values()) else 8


def suggest_parameters(image_bgr: np.ndarray, previous: PipelineConfig) -> tuple[PipelineConfig, AutoDiagnostics]:
    """Trả cấu hình đề xuất cho ảnh vừa mở; không sửa ``previous`` hoặc ảnh gốc.

    Chỉ là heuristic không giám sát: ưu tiên giữ những vùng lớn, xóa rác trắng
    nếu nhiều hạt nhỏ, hoặc lấp lỗ kín nếu lỗ hổng nổi trội. Không tự chỉnh
    thuật toán OpenCV/Scratch, min/max area; đó là lựa chọn vận hành/thống kê.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Cần ảnh BGR để tự cấu hình")

    # Dùng ảnh thu nhỏ để đánh giá, không can thiệp độ phân giải ảnh xử lý thật.
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 680.0 / max(height, width))
    if scale < 1:
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        preview = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        preview = image_bgr

    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    invert = _choose_polarity(gray)
    binary_code = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, binary_code | cv2.THRESH_OTSU)

    noise_pixels, core, _ = _component_info(mask)
    hole_pixels = _internal_holes(mask)
    has_noise = noise_pixels >= 15 and noise_pixels >= hole_pixels * 0.55
    has_holes = hole_pixels >= 15 and hole_pixels > noise_pixels * 1.35
    operation = "closing" if has_holes else "opening"
    description = (
        "Ưu tiên lấp các lỗ kín trong đối tượng" if has_holes else
        "Ưu tiên loại nhiễu trắng nhỏ" if has_noise else
        "Ảnh tương đối sạch, ưu tiên tác động nhẹ"
    )

    # Không có căn cứ để dùng các phép tạo biên Top Hat / Black Hat để đếm.
    # Điểm phạt tránh kernel lớn phá vỡ hình dạng vật thể.
    best = None
    best_score = float("inf")
    core_pixels = max(1, int(np.count_nonzero(core)))
    foreground_pixels = max(1, int(np.count_nonzero(mask)))
    original_residual = hole_pixels if has_holes else noise_pixels

    for size in (3, 5, 7, 9):
        for shape in ("ellipse", "rect", "cross"):
            kernel = create_kernel(shape, size)
            for iterations in (1, 2):
                candidate = apply_morphology(mask, operation, kernel, iterations)
                if has_holes:
                    residual = _internal_holes(candidate)
                else:
                    residual, _, _ = _component_info(candidate)
                destroyed_core = np.count_nonzero(core & (candidate == 0)) / core_pixels
                added_pixels = np.count_nonzero((mask == 0) & (candidate != 0)) / foreground_pixels
                # Lượng cấu trúc thay đổi thường là dấu hiệu cần can thiệp cẩn thận.
                changed = np.count_nonzero(candidate != mask) / foreground_pixels
                residual_ratio = residual / max(1, original_residual)
                score = (1.2 * residual_ratio
                         + 3.0 * destroyed_core
                         + 2.5 * added_pixels
                         + 0.30 * changed
                         + 0.09 * ((size-3)//2)
                         + 0.12 * (iterations-1)
                         + (0.00 if shape == "ellipse" else 0.035))
                if score < best_score:
                    best_score = score
                    best = (shape, size, iterations, candidate)
    assert best is not None
    shape, size, iterations, recommended_mask = best

    # Bảo vệ các sản phẩm đã tách biệt trong ảnh nhị phân đầu vào.
    # Closing có thể vô tình nối 2 đồng xu chạm nhẹ thành 1 vùng (ví dụ:
    # ảnh coin.png: trước Closing 13 vùng, sau Closing 5x5 còn 12 vùng).
    # Chỉ can thiệp khi ảnh có nhiều vùng lớn khá đồng đều, như đồng xu.
    # Các cảnh phức tạp với vật thể khác loại / bị che khuất KHÔNG đủ
    # thông tin để khẳng định vùng nào là sản phẩm: không ép số đếm.
    if operation == "closing":
        h0, w0 = mask.shape
        area_min = max(100, round(h0 * w0 * 0.001))

        def significant_areas(binary_mask):
            _, _, item_stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
            values = item_stats[1:, cv2.CC_STAT_AREA]
            return values[values >= area_min]

        before = significant_areas(mask)
        after = significant_areas(recommended_mask)
        if len(before) >= 5 and len(after) < len(before):
            # Cảnh có nhiều vật tương tự nhau: tránh diễn giải quá mức
            # nếu diện tích vật thể chênh lệch lớn (như bánh răng ở
            # nhiều độ sâu, nhiều vật chồng nhau).
            variability = float(np.std(before) / max(1.0, np.mean(before)))
            if variability < 0.70:
                gentle_kernel = create_kernel("ellipse", 3)
                opening_mask = apply_morphology(mask, "opening", gentle_kernel, 1)
                if len(significant_areas(opening_mask)) >= len(before):
                    operation, shape, size, iterations = "opening", "ellipse", 3, 1
                    recommended_mask = opening_mask
                    description = "Giữ các vùng đã tách biệt; tránh Closing làm dính sản phẩm"

    connectivity = _connectivity_by_contact(recommended_mask)
    cfg = replace(previous,
                  operation=operation,
                  shape=shape,
                  kernel_size=size,
                  iterations=iterations,
                  morphology_input="binary",
                  threshold_method="otsu",
                  invert=invert,
                  connectivity=connectivity,
                  algorithm="opencv")
    cfg.validate()
    reason = description + ("; đảo sáng/tối vì nền sáng chiếm phần lớn rìa ảnh" if invert else "")
    changed = f"{operation.title()}, {shape} {size}×{size}, {iterations} lần, {connectivity} hướng"
    return cfg, AutoDiagnostics(reason, changed, noise_pixels, hole_pixels)
