# -*- coding: utf-8 -*-
"""Thống kê sản phẩm của Minh Anh (chuyển nguyên thuật toán từ notebook bàn giao).

Nhận labels int32 của Linh và ảnh BGR gốc. ID 0 là nền.
Trả dict với count, objects, annotated_image cho core.pipeline và GUI.
Chỉ lọc diện tích; không tự gán nhãn lại hoặc làm thay đổi ảnh gốc.
"""

import cv2
import numpy as np

def analyze_objects(labels, image_bgr, min_area=100, max_area=None):
    if labels.ndim != 2:
        raise ValueError("labels phải là ma trận 2D")
    if image_bgr is None:
        raise ValueError("image_bgr không được để trống")

    h, w = labels.shape
    # Tạo bản sao ảnh gốc để vẽ kết quả, tránh ghi đè lên ảnh gốc
    annotated_image = image_bgr.copy()

    objects_list = []
    valid_count = 0

    # Lấy danh sách các nhãn duy nhất (bỏ qua ID 0 là nền)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    for label_id in unique_labels:
        # Tìm tất cả các tọa độ (y, x) thuộc nhãn hiện tại
        y_indices, x_indices = np.where(labels == label_id)

        # 1. Tính diện tích (area) bằng tổng số pixel thuộc nhãn
        area = len(y_indices)
        if area == 0:
            continue

        # Kiểm tra điều kiện bộ lọc diện tích (min_area và max_area)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        # 2. Tính tâm (centroid): trung bình tọa độ x, y
        cy = float(np.mean(y_indices))
        cx = float(np.mean(x_indices))

        # 3. Tính khung bao (bbox): min_x, max_x, min_y, max_y
        min_x = int(np.min(x_indices))
        max_x = int(np.max(x_indices))
        min_y = int(np.min(y_indices))
        max_y = int(np.max(y_indices))

        box_width = max_x - min_x + 1
        box_height = max_y - min_y + 1
        bbox = (min_x, min_y, box_width, box_height)

        # Tăng biến đếm sản phẩm hợp lệ
        valid_count += 1

        # Lưu thông tin đối tượng vào danh sách
        objects_list.append({
            "id": int(label_id),
            "area": int(area),
            "centroid": (cx, cy),
            "bbox": bbox
        })

        # 4. Vẽ trực tiếp lên ảnh kết quả
        # Vẽ khung chữ nhật (Bounding Box) màu xanh lá (Green: 0, 255, 0)
        cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Ghi số thứ tự (1, 2, 3...) lên phía trên khung chữ nhật
        text_label = str(valid_count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Đặt vị trí text ngay góc trên bên trái của bbox
        text_y = min_y - 5 if min_y - 5 > 10 else min_y + 20
        cv2.putText(annotated_image, text_label, (min_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    return {
        "count": valid_count,
        "objects": objects_list,
        "annotated_image": annotated_image
    }
