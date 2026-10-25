# -*- coding: utf-8 -*-
"""Các thuật toán của Quang, trích nguyên từ morphology_module.ipynb.

Không thay đổi hành vi xử lý. GUI bổ sung threshold modes qua core.pipeline.
"""
import numpy as np
import cv2

# ---------------------------------------------------------
# 1. THUẬT TOÁN TỰ BUILD (TƯ DUY SLIDING WINDOW) - Giữ nguyên của bạn
# ---------------------------------------------------------
def manual_morphology(image, kernel, mode='erosion'):
    """
    Tự cài đặt Erosion và Dilation để đối chiếu OpenCV.
    Quy ước: vật thể trắng 255, nền đen 0.
    """
    if mode not in ('erosion', 'dilation'):
        raise ValueError("mode phải là erosion hoặc dilation")

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Erosion: padding trắng; Dilation: padding đen
    pad_value = 255 if mode == 'erosion' else 0

    padded_img = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=pad_value
    )

    output_img = np.zeros_like(image)
    active_kernel = kernel != 0

    for y in range(img_h):
        for x in range(img_w):
            roi = padded_img[y:y + k_h, x:x + k_w]
            pixels = roi[active_kernel]

            if mode == 'erosion':
                if np.all(pixels == 255):
                    output_img[y, x] = 255
            else:
                if np.any(pixels == 255):
                    output_img[y, x] = 255

    return output_img

# ---------------------------------------------------------
# 2. CÁC HÀM TIỆN ÍCH DÀNH CHO BÊN GUI GỌI
# ---------------------------------------------------------

def binarize_image(img_color, thresh_val=127, auto_otsu=False, auto_polarity=False):
    """Chuyển ảnh BGR thành ảnh nhị phân (sản phẩm trắng, nền đen).

    Giữ mặc định *manual* của GUI cũ để không đổi kết quả đang hoạt động.
    Chế độ tự động của Quang có thể gọi với ``auto_otsu=True`` và
    ``auto_polarity=True`` để tự đảo mask dựa vào viền ảnh. GUI hiện có
    thuật toán tự chọn polarity riêng và được giữ nguyên, tránh đảo hai lần.
    """
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    if auto_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    if auto_polarity:
        border = np.concatenate((binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]))
        if np.mean(border) > 127:
            binary = cv2.bitwise_not(binary)
    return binary

def create_kernel(shape_type='rect', ksize=3):
    """
    Hàm 2: Tạo Kernel bằng OpenCV.
    Bên GUI có thể cho người dùng chọn hình dạng (Rect, Ellipse, Cross) và kích thước (3, 5, 7, 9...)
    """
    if shape_type == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    elif shape_type == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    elif shape_type == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))

    # Mặc định
    return cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def apply_morphology(binary_img, op_name, kernel, iterations=1):
    """
    Hàm 3: Áp dụng phép toán hình thái học bằng OpenCV.
    Hỗ trợ đầy đủ các phép toán như trong ảnh bạn cung cấp (opening, closing, top hat...).
    """
    if op_name == 'erosion':
        return cv2.erode(binary_img, kernel, iterations=iterations)
    elif op_name == 'dilation':
        return cv2.dilate(binary_img, kernel, iterations=iterations)
    elif op_name == 'opening':
        return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op_name == 'closing':
        return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op_name == 'gradient':
        return cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif op_name == 'top_hat':
        return cv2.morphologyEx(binary_img, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif op_name == 'black_hat':
        return cv2.morphologyEx(binary_img, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)

    return binary_img

# ---------------------------------------------------------
# WATERSHED: tách vật thể chạm nhau (đoạn mới từ notebook cuối của Quang)
# Mặc định không gọi, chỉ chạy khi người dùng chủ động chọn trong GUI.
# ---------------------------------------------------------
def separate_touching_objects(binary_img, distance_ratio=0.35):
    """
    Tách các vật thể trắng đang tiếp xúc nhau.

    distance_ratio:
        Nhỏ hơn -> marker nhiều hơn, có thể tách quá mức.
        Lớn hơn -> marker ít hơn, có thể vẫn gộp vật thể.

    Trả về:
        separated_mask: mặt nạ các vật thể sau tách.
        labels: nhãn Watershed, -1 là đường biên.
    """
    # Loại bỏ đốm nhỏ và làm kín lỗ nhỏ một cách nhẹ nhàng
    kernel = create_kernel('ellipse', 3)
    cleaned = cv2.morphologyEx(
        binary_img, cv2.MORPH_OPEN, kernel, iterations=1
    )

    # Khoảng cách từ pixel trắng tới pixel nền gần nhất
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return cleaned, np.zeros(cleaned.shape, dtype=np.int32)

    # Vùng lõi của từng vật thể làm marker ban đầu
    _, sure_fg = cv2.threshold(
        dist, distance_ratio * dist.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Ước lượng nền chắc chắn
    sure_bg = cv2.dilate(cleaned, kernel, iterations=2)

    # Vùng chưa biết nằm giữa lõi vật thể và nền chắc chắn
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Tạo nhãn marker
    num_markers, markers = cv2.connectedComponents(sure_fg)

    # Watershed cần marker nền bắt đầu từ 1;
    # vùng chưa biết được đánh dấu bằng 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed nhận ảnh màu và sửa markers tại chỗ
    img_3ch = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_3ch, markers)

    # Mỗi vùng có nhãn > 1 là một vùng vật thể.
    # Bỏ đường biên Watershed (-1) khỏi mặt nạ.
    separated_mask = np.zeros_like(cleaned)

    for label_id in np.unique(markers):
        if label_id > 1:
            separated_mask[markers == label_id] = 255

    return separated_mask, markers
