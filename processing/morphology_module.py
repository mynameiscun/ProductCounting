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
    Hàm tự code thủ công. Dùng để đối chiếu với OpenCV trong báo cáo.
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output_img = np.zeros_like(image)

    for y in range(img_h):
        for x in range(img_w):
            roi = padded_img[y : y + k_h, x : x + k_w]
            pixels_under_kernel = roi[kernel == 1]

            if mode == 'erosion':
                if np.min(pixels_under_kernel) == 255:
                    output_img[y, x] = 255
            elif mode == 'dilation':
                if np.max(pixels_under_kernel) == 255:
                    output_img[y, x] = 255
    return output_img

# ---------------------------------------------------------
# 2. CÁC HÀM TIỆN ÍCH DÀNH CHO BÊN GUI GỌI
# ---------------------------------------------------------

def binarize_image(img_color, thresh_val=127):
    """
    Hàm 1: Tiền xử lý - Chuyển ảnh màu sang nhị phân
    Bên GUI có thể truyền thresh_val từ một thanh trượt (slider).
    """
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)
    return img_binary

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