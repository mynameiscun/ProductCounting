# ProductCounting – Hệ thống đếm sản phẩm qua xử lý ảnh

Ứng dụng Python + CustomTkinter, tích hợp ba phần của nhóm:

1. **Quang – Tiền xử lý & Morphology:** `processing/morphology_module.py`
2. **Linh – Connected Components:** `processing/connected_components_module.py`
3. **Minh Anh – Statistics & Counting:** `processing/statistics_module.py` (chuyển từ `notebook/statistics_module.ipynb`)

Luồng chính hiện tại nằm tại `core/pipeline.py`:

```
Ảnh BGR → Gray → Thresholding → Morphology → Mask nhị phân
                                                ↓
                   Connected Components (OpenCV/Scratch) → Label Matrix
                                                ↓
                   Statistics (lọc Min/Max Area, tâm, bbox, số lượng)
                                                ↓
                     GUI: ảnh có khung + số lượng + xem/xuất CSV
```

**Không chạy lại Connected Components trong module Statistics**: Minh Anh sử dụng trực tiếp ma trận `labels` do Linh tạo. Ảnh màu vùng liên thông chỉ để quan sát, không làm dữ liệu thống kê.

## Cách chạy trên Windows

1. Tải toàn bộ source và **giải nén** vào một thư mục trên máy.
2. Cài Python 3.10–3.14 (khuyến nghị 3.12) nếu chưa có.
3. Nhấp đúp `run_windows.bat` tại thư mục chứa `main.py` và `requirements.txt`.
4. Lần đầu cần Internet để tạo môi trường `.venv` và cài thư viện. Không cần Run as administrator.

Hoặc chạy trong Terminal của VS Code:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Khi khởi động thất bại, kiểm tra `startup_log.txt` (được tạo bởi BAT).

## Cách sử dụng

- **Mở ảnh**: hỗ trợ PNG, JPG, JPEG, BMP, WEBP; có thể mở đường dẫn chứa tiếng Việt.
- **Tự động theo ảnh mới**: đề xuất phép Morphology, kernel, iterations, đảo màu và Connectivity; có thể chỉnh tay hoặc chọn **Khôi phục mặc định**.
- **Threshold**: Manual, Otsu (T tự động), Adaptive. Khi đổi ngưỡng, ảnh tự cập nhật.
- **Morphology**: Erosion, Dilation, Opening, Closing; ba phép Gradient/Top Hat/Black Hat dùng để hiển thị đặc trưng, **không đếm trực tiếp**.
- **Connected Components**: OpenCV mặc định, Scratch BFS để đối chứng; Connectivity 4/8.
- **Statistics**: nhập Min Area, Max Area (để trống nghĩa là không giới hạn). Nhấn Enter hoặc chuyển trọng tâm ô nhập để áp dụng.
- **Kết quả**: số vùng thô, số sản phẩm hợp lệ, ảnh có Bounding Box và số thứ tự; **Xem chi tiết sản phẩm** để xem ID/diện tích/tâm/bbox và **Xuất CSV**.
- **Canvas**: Zoom/Pan và nút **Vừa khung**; **Lưu kết quả** để lưu ảnh đã đánh dấu.

## Phân biệt số đếm

- `raw_count`: Tổng số vùng liên thông **chưa lọc** (có thể gồm nhiễu).
- `count`: Số vùng sau lọc theo diện tích `min_area <= area <= max_area` (`max_area` có thể là `None`).
- `objects`: Danh sách chi tiết mỗi vùng hợp lệ, bao gồm `id`, `area`, `centroid=(cx,cy)`, `bbox=(x,y,w,h)`.
- `annotated_bgr`: Ảnh **BGR** có Bounding Box và số thứ tự trên mỗi vùng hợp lệ.

Số vùng hợp lệ không nhất thiết chính xác bằng số sản phẩm ngoài đời nếu hai vật dính nhau, một vật bị tách đôi hoặc ngưỡng không phù hợp. Cần có **ground truth** để đánh giá độ chính xác thực tế.

## Kiểm tra dự án

```powershell
python -m pip install pytest
python -m pytest -q tests
```

`tests/test_statistics_integration.py` kiểm tra module thật, bao gồm diện tích, tọa độ tâm, bbox, bộ lọc min/max, ảnh không đối tượng, Scratch/OpenCV và hiển thị khung. Các bài kiểm thử cũ về Morphology/Auto/Connected Components cũng được giữ lại.

## Ghi chú tích hợp

- Phần mềm giữ nguyên các thuật toán Morphology và Connected Components từ bản người dùng cung cấp; hàm `analyze_objects()` giữ nguyên logic của notebook Minh Anh.
- Giao diện được cập nhật **chỉ để hiển thị kết quả Statistics và cho phép xem/xuất danh sách đối tượng**. Các chức năng Zoom/Pan/Auto/Reset/OpenCV/Scratch vẫn giữ nguyên.
- Không đóng gói `.git` hoặc `.venv` trong ZIP: chúng là dữ liệu môi trường máy cá nhân, BAT sẽ tự tạo `.venv` khi cần.
