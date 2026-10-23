# -*- coding: utf-8 -*-
"""Module Connected Components của Linh: nhận ảnh nhị phân, trả về Label Matrix.

Dữ liệu: binary uint8 2D (nền 0, đối tượng 255); labels int32 2D (ID 0 là nền).
Minh Anh lấy labels để thống kê vùng và lọc diện tích; KHÔNG lấy ảnh RGB tô màu.
"""
import cv2
import numpy as np
from collections import deque

def connected_components_opencv(binary, connectivity=8):
    """Gán nhãn bằng OpenCV. Trả về (num_labels, labels) gồm cả nền ID 0."""
    if connectivity not in (4, 8):
        raise ValueError("Connectivity phải là 4 hoặc 8")
    if not isinstance(binary, np.ndarray) or binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("binary phải là ảnh nhị phân 2D, dtype uint8")
    return cv2.connectedComponents(binary, connectivity=connectivity)

def connected_components_scratch(binary, connectivity=8):
    """Tự gán nhãn vùng liên thông bằng Flood-fill BFS, hỗ trợ 4/8 hướng."""

    # 1. Chỉ cho phép 4 hoặc 8 hướng kết nối
    if connectivity not in (4, 8):
        raise ValueError("Connectivity phải là 4 hoặc 8")

    # 2. Ảnh nhị phân 2D uint8: nền 0, đối tượng khác 0 (Quang trả 0/255)
    if not isinstance(binary, np.ndarray) or binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("binary phải là ảnh nhị phân 2D, dtype uint8")
    h, w = binary.shape

    # 3. Ma trận lưu ID cho từng pixel; số 0 ban đầu nghĩa là chưa gán/nền
    labels = np.zeros((h, w), dtype=np.int32)

    # 4. Hướng hàng xóm 4: trên/dưới/trái/phải
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        # Thêm bốn hướng chéo khi chọn Connectivity 8
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    current_label = 0  # Đếm số vùng đã phát hiện; không tính nền

    # 5. Duyệt lần lượt từng pixel của ảnh để tìm vùng mới
    for y in range(h):
        for x in range(w):
            # Bỏ qua nền hoặc pixel đã thuộc một vùng được tìm trước đó
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue

            # 6. Pixel trắng chưa gán nhãn => bắt đầu vùng mới với ID mới
            current_label += 1
            labels[y, x] = current_label
            queue = deque([(y, x)])  # Hàng đợi các pixel đang chờ kiểm tra hàng xóm

            # 7. BFS: lấy pixel vào trước ra trước và mở rộng vùng hiện tại
            while queue:
                cy, cx = queue.popleft()  # Lấy pixel đầu Queue
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx  # Vị trí pixel hàng xóm

                    # Không cho phép truy cập ra ngoài ranh giới ảnh
                    if 0 <= ny < h and 0 <= nx < w:
                        # Chỉ nhận pixel thuộc đối tượng, chưa được đánh ID
                        if binary[ny, nx] != 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label  # Gán cùng ID vùng
                            queue.append((ny, nx))  # Đưa vào Queue để xét tiếp

    # +1 để tính nhãn nền 0, tương thích giao diện trả về của OpenCV
    return current_label + 1, labels

def colorize_labels(labels, seed=42):
    """Tạo ảnh RGB màu theo nhãn, nền đen. Chỉ dùng để hiển thị, không thống kê."""
    if labels.ndim != 2:
        raise ValueError("labels phải là ma trận nhãn 2D")
    if labels.size == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    if np.min(labels) < 0:
        raise ValueError("labels không được chứa nhãn âm")

    num_colors = int(labels.max()) + 1  # Xác định số màu cần tạo: ID 0 .. ID lớn nhất
    rng = np.random.default_rng(seed)  # Sinh màu ngẫu nhiên nhưng lặp lại được cùng seed
    colors = rng.integers(
        50, 256,
        size=(num_colors, 3),
        dtype=np.uint8
    )  # Mỗi màu gồm 3 giá trị R, G, B

    colors[0] = [0, 0, 0]  # Nền ID 0 luôn màu đen
    colored_image = colors[labels]  # Pixel cùng ID nhận cùng màu từ bảng
    return colored_image  # Ảnh RGB để hiển thị, KHÔNG thay thế labels

def run_connected_components(binary, connectivity=8, method="opencv"):
    """Cổng vào thống nhất cho GUI/Statistics: trả (num_labels, labels)."""
    if method == "opencv":
        return connected_components_opencv(binary, connectivity)
    if method == "scratch":
        return connected_components_scratch(binary, connectivity)
    raise ValueError("method phải là 'opencv' hoặc 'scratch'")

def same_partition(labels_a, labels_b):
    """True nếu hai Label Matrix chia pixel thành cùng vùng (ID có thể khác nhau)."""
    if labels_a.shape != labels_b.shape:
        return False  # Hai ảnh phải cùng kích thước
    if not np.array_equal(labels_a == 0, labels_b == 0):
        return False  # Phần nền ID 0 phải trùng nhau

    # Ghép các cặp ID ở cùng vị trí pixel (OpenCV-ID, Scratch-ID)
    pairs = np.column_stack((labels_a.ravel(), labels_b.ravel()))
    pairs = np.unique(pairs, axis=0)  # Bỏ các cặp ID trùng nhau

    # Mỗi ID của A tương ứng đúng một ID của B, và ngược lại
    return (
        len(pairs) == len(np.unique(pairs[:, 0]))
        and len(pairs) == len(np.unique(pairs[:, 1]))
    )
