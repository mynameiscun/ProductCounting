"""Giao diện xử lý ảnh và đếm sản phẩm.

Tách biệt phần điều khiển, hiển thị và thuật toán xử lý ảnh.
"""
from __future__ import annotations
import concurrent.futures
import csv
import queue
import sys
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk

from core.pipeline import PipelineConfig, process_image, imread_unicode, imwrite_unicode
from core.auto_config import suggest_parameters
from gui.image_canvas import ZoomCanvas

BG = "#090F17"
SURFACE = "#111C29"
SURFACE_2 = "#172535"
BORDER = "#253649"
FG = "#EAF0F8"
MUTED = "#93A6BC"
ACCENT = "#32D6BA"
WARNING = "#EAB66E"

OPERATIONS = [
    ("erosion", "Erosion — Phép co"),
    ("dilation", "Dilation — Phép giãn"),
    ("opening", "Opening — Loại nhiễu trắng nhỏ"),
    ("closing", "Closing — Lấp lỗ, khe nhỏ"),
    ("gradient", "Gradient — Trích xuất biên"),
    ("top_hat", "Top Hat — Chi tiết sáng nhỏ"),
    ("black_hat", "Black Hat — Chi tiết tối nhỏ"),
]

SHAPES = [("rect", "Vuông"), ("cross", "Chữ thập"), ("ellipse", "Elip")]

class ProductCounterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("ObjectLab | Phân tích và đếm sản phẩm")
        self.geometry("1440x880")
        self.minsize(1120, 720)
        self.configure(fg_color=BG)

        self.image_bgr = None
        self.image_path = None
        self.last_result = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._requests = queue.Queue()
        self._job = None
        self._generation = 0
        self._pending_process = None
        self._closing = False
        self._restoring_defaults = False  # Chặn callback khi đồng bộ cấu hình tự động hoặc reset
        self.auto_mode = tk.BooleanVar(value=True)
        self._manual_override = False  # Chỉnh tay trên ảnh này; ảnh mới sẽ tự gợi ý lại.

        self.operation = tk.StringVar(value="opening")
        self.shape = tk.StringVar(value="ellipse")
        self.morph_input = tk.StringVar(value="binary")
        self.threshold_method = tk.StringVar(value="Otsu")
        self.invert = tk.BooleanVar(value=False)
        self.connectivity = tk.StringVar(value="8")
        self.algorithm = tk.StringVar(value="OpenCV")
        self.kernel_value = tk.IntVar(value=3)
        self.iterations_value = tk.IntVar(value=1)
        self.threshold_value = tk.IntVar(value=127)
        self.use_watershed = tk.BooleanVar(value=False)
        self.distance_ratio = tk.DoubleVar(value=0.35)
        # Ngưỡng T tự động chỉ phục vụ HIỂN THỊ, không ghi đè ngưỡng Manual.
        self._otsu_threshold = None

        self._build_ui()
        self._poll_worker()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _label(self, parent, text, size=12, bold=False, color=FG, **kwargs):
        return ctk.CTkLabel(parent, text=text, text_color=color,
                            font=ctk.CTkFont(family="Segoe UI", size=size, weight="bold" if bold else "normal"),
                            **kwargs)

    def _section_title(self, text):
        self._label(self.sidebar_scroll, text, 12, True, FG).pack(anchor="w", padx=14, pady=(16,6))

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=338, corner_radius=0, fg_color=SURFACE,
                                     border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(1, weight=1)

        side_top = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        side_top.grid(row=0, column=0, sticky="ew", padx=18, pady=(22,13))
        self._label(side_top, "OBJECTLAB", 23, True, ACCENT).pack(anchor="w")
        self._label(side_top, "PHÂN TÍCH VÀ ĐẾM SẢN PHẨM", 10, color=MUTED).pack(anchor="w")

        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent",
                             corner_radius=0, scrollbar_button_color=BORDER, width=330)
        self.sidebar_scroll.grid(row=1, column=0, sticky="nsew")
        ctk.CTkButton(self.sidebar_scroll, text="+  Mở ảnh", height=38, corner_radius=10,
                      fg_color=ACCENT, hover_color="#24B89F", text_color="#071818",
                      font=ctk.CTkFont(size=13,weight="bold"), command=self._open_image).pack(fill="x",padx=14,pady=(4,5))
        # Khôi phục đồng thời toàn bộ tham số, giữ nguyên ảnh đang mở.
        ctk.CTkButton(self.sidebar_scroll, text="↺  Khôi phục mặc định", height=34,
                      corner_radius=10, fg_color="transparent", hover_color=SURFACE_2,
                      border_width=1, border_color=BORDER, text_color=FG,
                      font=ctk.CTkFont(size=12), command=self.reset_defaults).pack(
                          fill="x", padx=14, pady=(3,8))

        # Auto thực sự phân tích lại ảnh (khác việc Otsu chỉ tính lại ngưỡng T).
        auto_box = ctk.CTkFrame(self.sidebar_scroll, fg_color=SURFACE_2,
                                corner_radius=10, border_width=1, border_color=BORDER)
        auto_box.pack(fill="x", padx=14, pady=(5,4))
        ctk.CTkSwitch(auto_box, text="Tự động theo ảnh mới", variable=self.auto_mode,
                      command=self._auto_changed, progress_color="#168479",
                      button_color=FG, font=ctk.CTkFont(size=12,weight="bold")).pack(
                          anchor="w", padx=11, pady=(11,4))
        self.auto_detail_label = self._label(
            auto_box,
            "Tự gợi ý Morphology, Kernel và cách kết nối dựa trên ảnh.",
            10, color=MUTED, wraplength=275, justify="left")
        self.auto_detail_label.pack(anchor="w", padx=11, pady=(0,10))

        self._section_title("TÁCH NỀN")
        ctk.CTkSegmentedButton(self.sidebar_scroll, variable=self.threshold_method,
               values=["Manual", "Otsu", "Adaptive"], selected_color="#146E69",
               selected_hover_color="#168479", fg_color=SURFACE_2,
               command=lambda _: self._user_threshold_changed()).pack(fill="x",padx=14)
        tr = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        tr.pack(fill="x", padx=14,pady=(10,3))
        self.threshold_caption = self._label(tr,"Ngưỡng T tự động (Otsu)",11,color=MUTED)
        self.threshold_caption.pack(side="left")
        self.threshold_display=self._label(tr,"—",12,True,ACCENT)
        self.threshold_display.pack(side="right")
        self.threshold_slider=ctk.CTkSlider(self.sidebar_scroll,from_=0,to=255,number_of_steps=255,
                progress_color=ACCENT, button_color=ACCENT,
                command=self._on_threshold_slider)
        self.threshold_slider.pack(fill="x",padx=14,pady=(0,8))
        self.threshold_slider.set(127)
        ctk.CTkSwitch(self.sidebar_scroll,text="Đảo nền / sản phẩm (Binary INV)",
               variable=self.invert,progress_color="#168479",button_color=FG,
               command=self._user_changed, font=ctk.CTkFont(size=11)).pack(anchor="w",padx=14,pady=(0,4))

        self._section_title("PHÉP TOÁN MORPHOLOGY")
        for val,text in OPERATIONS:
            ctk.CTkRadioButton(self.sidebar_scroll,text=text,variable=self.operation,value=val,
                  command=self._user_changed,font=ctk.CTkFont(size=11),radiobutton_width=17,
                  radiobutton_height=17, border_color="#72849A", fg_color=ACCENT,
                  hover_color=ACCENT).pack(anchor="w",padx=16,pady=4)

        self._section_title("HÌNH DẠNG KERNEL")
        shape_row=ctk.CTkFrame(self.sidebar_scroll,fg_color="transparent")
        shape_row.pack(fill="x",padx=14)
        for value,name in SHAPES:
            ctk.CTkRadioButton(shape_row,text=name,variable=self.shape,value=value,
                   command=self._user_changed, radiobutton_width=16,radiobutton_height=16,
                   fg_color=ACCENT,hover_color=ACCENT,font=ctk.CTkFont(size=11)).pack(side="left",padx=(0,13))
        self._slider_group("Kích thước Kernel", "kernel", 0,3,0,3,"3×3     5×5     7×7     9×9")
        self._slider_group("Số lần lặp", "iterations", 1,5,1,4,"1                3                5")

        self._section_title("KIỂU ẢNH XỬ LÝ")
        for value,name in [("binary","Mask nhị phân (0/255)"),("gray","Ảnh xám (Grayscale)")]:
            ctk.CTkRadioButton(self.sidebar_scroll,text=name,variable=self.morph_input,
                 value=value,command=self._user_changed, fg_color=ACCENT,hover_color=ACCENT,
                 radiobutton_width=16,radiobutton_height=16,
                 font=ctk.CTkFont(size=11)).pack(anchor="w",padx=16,pady=4)

        self._section_title("PHÂN VÙNG ĐỐI TƯỢNG")
        labelrow = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        labelrow.pack(fill="x",padx=14,pady=(0,8))
        self._label(labelrow,"Connectivity",11,color=MUTED).pack(side="left")
        ctk.CTkSegmentedButton(labelrow, variable=self.connectivity,values=["4","8"],
                 width=94,fg_color=SURFACE_2,selected_color="#146E69",
                 command=lambda _:self._user_changed()).pack(side="right")
        methodrow = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        methodrow.pack(fill="x",padx=14)
        self._label(methodrow,"Thuật toán",11,color=MUTED).pack(side="left")
        ctk.CTkSegmentedButton(methodrow, variable=self.algorithm,values=["OpenCV","Scratch"],
                 width=155,fg_color=SURFACE_2,selected_color="#146E69",
                 command=lambda _:self._user_changed()).pack(side="right")

        # Chế độ tách vật thể chạm nhau mới từ Quang, mặc định tắt.
        # Khi bật, Watershed sẽ gán nhãn trực tiếp cho vùng đã tách;
        # CCL và Connectivity vẫn được giữ cho chế độ thông thường.
        waterframe = ctk.CTkFrame(self.sidebar_scroll, fg_color=SURFACE_2,
                                  corner_radius=10, border_width=1, border_color=BORDER)
        waterframe.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkSwitch(waterframe, text="Tách sản phẩm chạm nhau (Watershed)",
                      variable=self.use_watershed, progress_color="#168479",
                      button_color=FG, font=ctk.CTkFont(size=11, weight="bold"),
                      command=self._user_changed).pack(anchor="w", padx=11, pady=(10, 6))
        waterrow = ctk.CTkFrame(waterframe, fg_color="transparent")
        waterrow.pack(fill="x", padx=11)
        self._label(waterrow, "Mức tách (Distance Ratio)", 10, color=MUTED).pack(side="left")
        self.water_ratio_display = self._label(waterrow, "0.35", 10, True, ACCENT)
        self.water_ratio_display.pack(side="right")
        self.water_ratio_slider = ctk.CTkSlider(
            waterframe, from_=0.15, to=0.70, number_of_steps=11,
            progress_color=ACCENT, button_color=ACCENT,
            command=self._on_watershed_ratio)
        self.water_ratio_slider.pack(fill="x", padx=11, pady=(4, 6))
        self.water_ratio_slider.set(0.35)
        self._label(waterframe,
                    "Bật khi các vật thể dính nhau. Chế độ này gán nhãn bằng Watershed thay cho CCL 4/8.",
                    10, color=MUTED, wraplength=272, justify="left").pack(
                        anchor="w", padx=11, pady=(0, 10))

        self._section_title("THỐNG KÊ SẢN PHẨM")
        statsfields=ctk.CTkFrame(self.sidebar_scroll,fg_color="transparent")
        statsfields.pack(fill="x",padx=14)
        statsfields.grid_columnconfigure((0,1),weight=1)
        self._label(statsfields,"Min Area",10,color=MUTED).grid(row=0,column=0,sticky="w")
        self._label(statsfields,"Max Area (trống = ∞)",10,color=MUTED).grid(row=0,column=1,sticky="w",padx=(8,0))
        self.min_area_entry=ctk.CTkEntry(statsfields,placeholder_text="100",height=30,fg_color=SURFACE_2,border_color=BORDER)
        self.min_area_entry.insert(0,"100")
        self.min_area_entry.grid(row=1,column=0,sticky="ew",pady=(5,2))
        self.max_area_entry=ctk.CTkEntry(statsfields,placeholder_text="Không giới hạn",height=30,fg_color=SURFACE_2,border_color=BORDER)
        self.max_area_entry.grid(row=1,column=1,sticky="ew",padx=(8,0),pady=(5,2))
        for e in (self.min_area_entry,self.max_area_entry):
            e.bind("<Return>", lambda _: self.request_process())
            e.bind("<FocusOut>", lambda _: self.request_process())
        self._label(self.sidebar_scroll,
                    "Chỉ đếm các vùng có diện tích trong khoảng Min / Max Area.",
                    10,color=MUTED,wraplength=295,justify="left").pack(anchor="w",padx=14,pady=(6,8))
        self.details_button = ctk.CTkButton(
            self.sidebar_scroll, text="Xem chi tiết sản phẩm", height=34,
            corner_radius=10, fg_color=SURFACE_2, hover_color="#213A4A",
            border_width=1, border_color=BORDER, text_color=FG,
            font=ctk.CTkFont(size=12), command=self._show_object_details,
            state="disabled")
        self.details_button.pack(fill="x", padx=14, pady=(0,18))
        self._threshold_changed(run=False)

        # Bên phải: thanh tiêu đề, các số liệu, 4 canvas và trạng thái
        self.main_panel=ctk.CTkFrame(self,corner_radius=0,fg_color=BG)
        self.main_panel.grid(row=0,column=1,sticky="nsew",padx=(18,18),pady=(18,16))
        self.main_panel.grid_columnconfigure(0,weight=1)
        self.main_panel.grid_rowconfigure(2,weight=1)

        header=ctk.CTkFrame(self.main_panel,fg_color="transparent")
        header.grid(row=0,column=0,sticky="ew",pady=(0,16))
        header.grid_columnconfigure(0,weight=1)
        self._label(header,"Bảng phân tích ảnh",23,True).grid(row=0,column=0,sticky="w")
        self.file_label=self._label(header,"Chưa chọn ảnh",11,color=MUTED)
        self.file_label.grid(row=1,column=0,sticky="w",pady=(3,0))
        # Hiển thị ngay cấu hình đang áp dụng, thay vì thông báo kỹ thuật cho người viết code.
        self.active_config_label=self._label(
            header,"Opening  •  Elip 3×3  •  1 lần  •  Otsu tự động  •  8 hướng",
            11,color=MUTED,wraplength=850,justify="left"
        )
        self.active_config_label.grid(row=2,column=0,columnspan=2,sticky="w",pady=(8,0))
        act=ctk.CTkFrame(header,fg_color="transparent")
        act.grid(row=0,column=1,rowspan=2)
        ctk.CTkButton(act,text="Vừa khung",width=92,height=34,fg_color=SURFACE_2,
                      border_color=BORDER,border_width=1,command=self.fit_all).pack(side="left",padx=(0,9))
        ctk.CTkButton(act,text="Lưu kết quả",width=112,height=34,fg_color=ACCENT,
                      hover_color="#25B99F",text_color="#0B211B",command=self._save_image).pack(side="left")

        metrics=ctk.CTkFrame(self.main_panel,fg_color="transparent")
        metrics.grid(row=1,column=0,sticky="ew",pady=(0,13))
        metrics.grid_columnconfigure((0,1,2),weight=1)
        self.raw_value = self._metric(metrics,0,"VÙNG LIÊN THÔNG", "—", "Tổng vùng được phát hiện")
        self.count_value = self._metric(metrics,1,"SẢN PHẨM HỢP LỆ", "—", "Sau lọc diện tích")
        self.time_value = self._metric(metrics,2,"THỜI GIAN XỬ LÝ", "—", "Tổng thời gian xử lý")

        pics=ctk.CTkFrame(self.main_panel, fg_color="transparent")
        pics.grid(row=2,column=0,sticky="nsew")
        pics.grid_columnconfigure((0,1),weight=1,uniform="view")
        pics.grid_rowconfigure((0,1),weight=1,uniform="view")
        self.views={}
        self._viewer_card(pics,0,0,"original","ẢNH GỐC", "Ảnh đã chọn")
        self._viewer_card(pics,0,1,"binary","ẢNH NHỊ PHÂN", "Kết quả tách nền")
        self._viewer_card(pics,1,0,"morphology","SAU MORPHOLOGY", "Kết quả xử lý hình thái")
        self._viewer_card(pics,1,1,"result","KẾT QUẢ ĐẾM", "Khung bao sản phẩm hợp lệ")

        status=ctk.CTkFrame(self.main_panel,fg_color=SURFACE,corner_radius=12,border_color=BORDER,border_width=1)
        status.grid(row=3,column=0,sticky="ew",pady=(12,0))
        status.grid_columnconfigure(0,weight=1)
        self.status_label=self._label(status,"Chọn ảnh để bắt đầu.",
                                11,color=MUTED,wraplength=750,justify="left")
        self.status_label.grid(row=0,column=0,sticky="w",padx=14,pady=10)

    def _metric(self,parent,column,title,value,foot):
        outer=ctk.CTkFrame(parent,fg_color=SURFACE,corner_radius=12,border_width=1,border_color=BORDER)
        outer.grid(row=0,column=column,sticky="ew",padx=(0,10 if column<2 else 0))
        self._label(outer,title,10,True,MUTED).pack(anchor="w",padx=15,pady=(13,2))
        value_label=self._label(outer,value,23,True,ACCENT)
        value_label.pack(anchor="w",padx=15)
        self._label(outer,foot,10,color=MUTED).pack(anchor="w",padx=15,pady=(1,13))
        return value_label

    def _viewer_card(self,parent,row,col,key,title,subtitle):
        frame=ctk.CTkFrame(parent,fg_color=SURFACE,corner_radius=12,
                           border_width=1,border_color=BORDER)
        frame.grid(row=row,column=col,sticky="nsew",padx=(0,7) if col==0 else (7,0),
                   pady=(0,7) if row==0 else (7,0))
        frame.grid_columnconfigure(0,weight=1)
        frame.grid_rowconfigure(1,weight=1)
        top=ctk.CTkFrame(frame,fg_color="transparent")
        top.grid(row=0,column=0,sticky="ew",padx=11,pady=(8,7))
        top.grid_columnconfigure(0,weight=1)
        self._label(top,title,11,True).grid(row=0,column=0,sticky="w")
        sub=self._label(top,subtitle,10,color=MUTED)
        sub.grid(row=1,column=0,sticky="w")
        ctk.CTkButton(top,text="Vừa khung",width=70,height=24,fg_color=SURFACE_2,
             hover_color="#213A4A",command=lambda k=key:self.views[k]["canvas"].fit_image()).grid(row=0,column=1,rowspan=2,padx=(8,0))
        canvas=ZoomCanvas(frame)
        canvas.grid(row=1,column=0,sticky="nsew",padx=8,pady=(0,8))
        self.views[key]={"canvas":canvas,"subtitle":sub}

    def _slider_group(self,title,kind,low,high,initial,steps,ends):
        container=ctk.CTkFrame(self.sidebar_scroll,fg_color="transparent")
        container.pack(fill="x",padx=14,pady=(13,0))
        h=ctk.CTkFrame(container,fg_color="transparent")
        h.pack(fill="x")
        self._label(h,title,11,True).pack(side="left")
        val_label=self._label(h,"3 × 3" if kind=="kernel" else "1",11,True,ACCENT)
        val_label.pack(side="right")
        if kind=="kernel":self.kernel_display=val_label
        else:self.iter_display=val_label
        slider=ctk.CTkSlider(container,from_=low,to=high,number_of_steps=steps,
             progress_color=ACCENT,button_color=ACCENT,command=lambda val,k=kind:self._on_slider(k,val))
        slider.pack(fill="x",pady=(7,0))
        # Lưu tham chiếu để nút Khôi phục mặc định đưa cả vị trí nút kéo về đúng nấc.
        if kind == "kernel":
            self.kernel_slider = slider
        else:
            self.iterations_slider = slider
        slider.set(initial)
        self._label(container,ends,9,color=MUTED).pack(fill="x",pady=(2,0))

    def _on_watershed_ratio(self, value):
        if self._restoring_defaults:
            return
        value = round(float(value), 2)
        self.distance_ratio.set(value)
        self.water_ratio_display.configure(text=f"{value:.2f}")
        self._user_changed()

    def _auto_changed(self):
        """Bật lại auto thì phân tích ảnh đang mở; tắt auto giữ mọi giá trị."""
        if self._restoring_defaults:
            return
        self._manual_override = not self.auto_mode.get()
        if self.auto_mode.get():
            self.auto_detail_label.configure(text="Đang phân tích thông số phù hợp với ảnh...")
        else:
            self.auto_detail_label.configure(text="Chỉnh tay: giữ các thông số khi mở ảnh mới.")
        self.request_process(immediate=self.auto_mode.get())

    def _user_changed(self):
        """Chỉnh tay ảnh này; nếu Auto bật thì ảnh MỚI lại được tự gợi ý."""
        if self._restoring_defaults:
            return
        self._manual_override = True
        if self.auto_mode.get():
            self.auto_detail_label.configure(
                text="Đang chỉnh tay ảnh này. Mở ảnh khác sẽ tự chọn lại thông số.")
        else:
            self.auto_detail_label.configure(text="Chỉnh tay: giữ các thông số khi mở ảnh mới.")
        self.request_process()

    def _user_threshold_changed(self):
        self._user_changed()
        self._threshold_changed(run=False)

    def _sync_auto_controls(self, cfg, info):
        """Đưa các lựa chọn thực tế lên radio, segmented button và slider."""
        self._restoring_defaults = True
        try:
            self.operation.set(cfg.operation)
            self.shape.set(cfg.shape)
            self.morph_input.set(cfg.morphology_input)
            self.threshold_method.set(cfg.threshold_method.capitalize())
            self.invert.set(cfg.invert)
            self.connectivity.set(str(cfg.connectivity))
            self.algorithm.set("OpenCV" if cfg.algorithm == "opencv" else "Scratch")
            self.kernel_value.set(cfg.kernel_size)
            self.iterations_value.set(cfg.iterations)
            self.kernel_slider.set((cfg.kernel_size - 3)//2)
            self.iterations_slider.set(cfg.iterations)
            self.kernel_display.configure(text=f"{cfg.kernel_size} × {cfg.kernel_size}")
            self.iter_display.configure(text=str(cfg.iterations))
            self._threshold_changed(run=False)
            self.auto_detail_label.configure(
                text=f"Đã chọn: {info.changed}. {info.reason}.")
        finally:
            self._restoring_defaults = False

    def reset_defaults(self):
        """Đặt lại toàn bộ tùy chọn xử lý theo PipelineConfig; giữ nguyên ảnh đang mở.

        Các thay đổi bằng .set() có thể kích hoạt callback của widget. Tạm chặn
        việc xử lý ảnh đến khi mọi thông số đã nhất quán, sau đó chạy lại một lần.
        """
        defaults = PipelineConfig()
        self._restoring_defaults = True
        try:
            self.auto_mode.set(True)
            self._manual_override = False
            self.auto_detail_label.configure(text="Đang chờ gợi ý cho ảnh hiện tại...")
            # Tách nền và Morphology
            self.threshold_method.set(defaults.threshold_method.capitalize())
            self.threshold_value.set(defaults.threshold_value)
            self.invert.set(defaults.invert)
            self.operation.set(defaults.operation)
            self.shape.set(defaults.shape)
            self.morph_input.set(defaults.morphology_input)
            self.use_watershed.set(defaults.use_watershed)
            self.distance_ratio.set(defaults.distance_ratio)
            self.water_ratio_slider.set(defaults.distance_ratio)
            self.water_ratio_display.configure(text=f"{defaults.distance_ratio:.2f}")

            # Connected Components
            self.connectivity.set(str(defaults.connectivity))
            self.algorithm.set("OpenCV" if defaults.algorithm == "opencv" else "Scratch")

            # Đồng bộ giá trị xử lý VÀ vị trí thanh trượt GUI
            self.kernel_value.set(defaults.kernel_size)
            self.iterations_value.set(defaults.iterations)
            self.kernel_slider.set((defaults.kernel_size - 3) // 2)
            self.iterations_slider.set(defaults.iterations)
            self.kernel_display.configure(text=f"{defaults.kernel_size} × {defaults.kernel_size}")
            self.iter_display.configure(text=str(defaults.iterations))

            # Ngưỡng Manual được lưu riêng, không bị Otsu tự động ghi đè.
            self._otsu_threshold = None
            self.threshold_slider.configure(state="normal")
            self.threshold_slider.set(defaults.threshold_value)
            self._threshold_changed(run=False)

            # Tham số thống kê chuẩn bị cho module Minh Anh
            self.min_area_entry.delete(0, "end")
            self.min_area_entry.insert(0, str(defaults.min_area))
            self.max_area_entry.delete(0, "end")
            if defaults.max_area is not None:
                self.max_area_entry.insert(0, str(defaults.max_area))
        finally:
            self._restoring_defaults = False

        if self.image_bgr is not None:
            # Cập nhật lại 4 ảnh và số vùng từ chính ảnh đang mở, không tải ảnh khác.
            self.request_process(immediate=True)
        else:
            self.auto_detail_label.configure(text="Tự gợi ý khi mở ảnh.")
            self.active_config_label.configure(
                text="Đang áp dụng: Opening  •  Elip 3×3  •  1 lần  •  Otsu tự động  •  8 hướng  •  OpenCV"
            )
            self.status_label.configure(
                text="Đã khôi phục cài đặt mặc định. Chọn ảnh để bắt đầu.", text_color=MUTED
            )

    def _on_slider(self, kind, val):
        if kind=="kernel":
            n=3+2*round(val)
            self.kernel_value.set(n)
            self.kernel_display.configure(text=f"{n} × {n}")
        else:
            n=round(val)
            self.iterations_value.set(n)
            self.iter_display.configure(text=str(n))
        self._user_changed()

    def _on_threshold_slider(self, value):
        # Ở chế độ tự động, giá trị thanh trượt chỉ phản ánh kết quả thuật toán.
        # Không được ghi ngược T tự động vào giá trị Manual mà người dùng đã đặt.
        if self.threshold_method.get() != "Manual":
            return
        n = round(value)
        self.threshold_value.set(n)
        self.threshold_display.configure(text=str(n))
        self._user_changed()

    def _threshold_changed(self, run=True):
        method = self.threshold_method.get()
        if method == "Manual":
            self.threshold_caption.configure(text="Ngưỡng T (tùy chỉnh)")
            self.threshold_slider.configure(state="normal")
            self.threshold_slider.set(self.threshold_value.get())
            self.threshold_display.configure(text=str(self.threshold_value.get()))
        elif method == "Otsu":
            self.threshold_caption.configure(text="Ngưỡng T tự động (Otsu)")
            self.threshold_slider.configure(state="disabled")
            # Sẽ cập nhật sang T thực tế khi tính xong ảnh đang mở.
            self.threshold_display.configure(text="Đang tính..." if self.image_bgr is not None else "—")
        else:
            self.threshold_caption.configure(text="Ngưỡng T (Adaptive)")
            self.threshold_slider.configure(state="disabled")
            self.threshold_display.configure(text="Theo vùng")
        if run:
            self.request_process()

    def _display_threshold_result(self, result):
        """Đồng bộ trị số T và mô tả ảnh với chế độ đang được chọn."""
        cfg = result["config"]
        method = cfg.threshold_method
        if method == "manual":
            info = f"Tùy chỉnh • T = {cfg.threshold_value}"
            self.threshold_display.configure(text=str(cfg.threshold_value))
        elif method == "otsu":
            # Với Morphology ảnh xám, T hiển thị là ngưỡng của mask SAU xử lý.
            # Với ảnh nhị phân, đây là ngưỡng Otsu đầu vào.
            info_used = result.get("count_threshold_info", result["threshold_info"])
            threshold = round(float(info_used.split("T=")[-1]))
            self._otsu_threshold = threshold
            self.threshold_slider.set(threshold)
            self.threshold_display.configure(text=str(threshold))
            info = f"Otsu tự động • T = {threshold}"
        else:
            self.threshold_display.configure(text="Theo vùng")
            info = "Adaptive • Ngưỡng theo vùng"
        if cfg.invert:
            info += " • Đảo màu"
        self.views["binary"]["subtitle"].configure(text=info)
        return info

    def _open_image(self):
        path=filedialog.askopenfilename(title="Chọn ảnh sản phẩm",
            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.webp"),("Tất cả", "*.*")])
        if path:self._load_image(path)

    def _load_image(self,path):
        try:
            image=imread_unicode(path)
        except Exception as e:
            messagebox.showerror("Không mở được ảnh",str(e))
            return
        self.image_path=Path(path)
        self.image_bgr=image
        # Ảnh mới chưa có kết quả: không hiển thị/thao tác với số liệu của ảnh cũ.
        self.last_result = None
        self.details_button.configure(state="disabled")
        self._otsu_threshold = None
        self._manual_override = not self.auto_mode.get()  # Ảnh mới được auto lại dù ảnh trước có chỉnh tay.
        self._threshold_changed(run=False)
        h,w=image.shape[:2]
        self.file_label.configure(text=f"{self.image_path.name}    •    {w} × {h} px")
        self.views["original"]["canvas"].set_image(image)
        if self.auto_mode.get():
            self.auto_detail_label.configure(text="Đang phân tích ảnh mới để gợi ý thông số...")
        self.request_process(immediate=True)

    def _get_config(self):
        min_txt=self.min_area_entry.get().strip()
        max_txt=self.max_area_entry.get().strip()
        try:
            min_area=int(min_txt) if min_txt else 0
            max_area=int(max_txt) if max_txt else None
        except ValueError:
            raise ValueError("Min Area / Max Area phải là số nguyên hoặc để trống")
        return PipelineConfig(
           operation=self.operation.get(),shape=self.shape.get(),
           kernel_size=self.kernel_value.get(), iterations=self.iterations_value.get(),
           morphology_input=self.morph_input.get(),
           threshold_method=self.threshold_method.get().lower(),
           threshold_value=self.threshold_value.get(),invert=self.invert.get(),
           connectivity=int(self.connectivity.get()),
           algorithm=self.algorithm.get().lower(),
           min_area=min_area,max_area=max_area,
           use_watershed=self.use_watershed.get(),
           distance_ratio=self.distance_ratio.get(),
        )

    def request_process(self,immediate=False):
        if self._closing or self._restoring_defaults or self.image_bgr is None:return
        self._generation += 1  # Kết quả ảnh cũ hết hiệu lực ngay khi người dùng đổi cấu hình.
        if self._pending_process is not None:
            self.after_cancel(self._pending_process)
            self._pending_process=None
        if immediate:
            self._start_process()
        else:
            self._pending_process=self.after(180,self._start_process)

    def _start_process(self):
        self._pending_process=None
        if self.image_bgr is None:return
        try:
            cfg=self._get_config()
            cfg.validate()
        except Exception as e:
            self.status_label.configure(text=f"Lỗi tham số: {e}",text_color="#FF938E")
            return
        job_id=self._generation
        # Không cần deepcopy ảnh: chỉ đọc, không ghi đè pixels gốc
        img=self.image_bgr
        if self._job is not None and not self._job.done():
            self._job.cancel()  # Hủy job chưa chạy, kết quả cũ sẽ bị bỏ qua
        self.status_label.configure(text="Đang xử lý ảnh...",text_color=MUTED)
        self._job=self._executor.submit(self._worker_process,img,cfg,job_id, bool(self.auto_mode.get() and not self._manual_override))

    def _worker_process(self,image,cfg,job_id,auto_enabled):
        try:
            auto_info = None
            if auto_enabled:
                cfg, auto_info = suggest_parameters(image, cfg)
            result = process_image(image,cfg)
            result["auto_info"] = auto_info
            self._requests.put((job_id,result,None))
        except Exception as e:
            self._requests.put((job_id,None,str(e)))

    def _poll_worker(self):
        try:
            while True:
                id_,result,error=self._requests.get_nowait()
                if id_ != self._generation:
                    continue
                if error:
                    self.status_label.configure(text=f"Không xử lý được: {error}", text_color="#FF938E")
                else:
                    self._render_result(result)
        except queue.Empty:
            pass
        if not self._closing:self.after(80,self._poll_worker)

    def _render_result(self,result):
        self.last_result=result
        cfg=result["config"]
        if result.get("auto_info") is not None and self.auto_mode.get() and not self._manual_override:
            self._sync_auto_controls(cfg, result["auto_info"])
        threshold_desc = self._display_threshold_result(result)
        # Luôn trình bày cấu hình đúng với kết quả ảnh vừa xử lý.
        operation_name = next(name.split(" — ")[0] for val,name in OPERATIONS if val == cfg.operation)
        shape_name = dict(SHAPES)[cfg.shape]
        source_name = "Nhị phân" if cfg.morphology_input == "binary" else "Ảnh xám"
        self.active_config_label.configure(
            text=f"{'Tự động theo ảnh' if result.get('auto_info') is not None else 'Chỉnh tay'}: {operation_name}  •  {shape_name} {cfg.kernel_size}×{cfg.kernel_size}"
                 f"  •  {cfg.iterations} lần  •  {threshold_desc}  •  {cfg.connectivity} hướng"
                 f"  •  {'OpenCV' if cfg.algorithm == 'opencv' else 'Scratch BFS'}"
                 + (f"  •  Watershed {cfg.distance_ratio:.2f}" if result.get("used_watershed") else "")
        )
        # Với Morphology ảnh xám, mask đem đi đếm được cắt ngưỡng SAU Morphology.
        # Vì vậy, khi đó hiển thị mask cuối cùng để người dùng thấy đúng ảnh đưa vào CCL.
        binary_preview = result["count_mask"] if cfg.morphology_input == "gray" and result["countable"] else result["binary"]
        self.views["binary"]["canvas"].set_image(binary_preview)
        if cfg.morphology_input == "gray" and result["countable"]:
            self.views["binary"]["subtitle"].configure(text=threshold_desc + " • Sau Morphology")
        self.views["morphology"]["canvas"].set_image(
            result["count_mask"] if result.get("used_watershed") else result["morphology"]
        )
        self.views["morphology"]["subtitle"].configure(
            text=(f"Sau tách Watershed • Ratio {cfg.distance_ratio:.2f}"
                  if result.get("used_watershed") else
                  f"{operation_name} • {shape_name} {cfg.kernel_size}×{cfg.kernel_size} • {cfg.iterations} lần • {source_name}"))
        if result["countable"]:
            if result["annotated_bgr"] is not None:
                self.views["result"]["canvas"].set_image(result["annotated_bgr"])
                self.views["result"]["subtitle"].configure(
                    text="Sản phẩm hợp lệ • Tách Watershed" if result.get("used_watershed")
                    else "Sản phẩm hợp lệ đã được đánh dấu")
            else:
                self.views["result"]["canvas"].set_image(result["colored_rgb"],rgb=True)
                self.views["result"]["subtitle"].configure(text=f"{result['raw_count']} vùng • Kết nối {cfg.connectivity} hướng • {cfg.algorithm.upper() if cfg.algorithm == 'opencv' else 'Scratch BFS'}")
            self.raw_value.configure(text=str(result["raw_count"]))
        else:
            self.views["result"]["canvas"].set_image(None)
            self.views["result"]["subtitle"].configure(text="Chọn phép co, giãn, Opening hoặc Closing để phân vùng")
            self.raw_value.configure(text="—")
        count=result["count"]
        self.count_value.configure(text="—" if count is None else str(count))
        # Nút chỉ khả dụng khi module Statistics đã xử lý thành công một ảnh đếm được.
        self.details_button.configure(state="normal" if count is not None else "disabled")
        self.time_value.configure(text=f"{result['times']['total_ms']:.1f} ms")
        if not result["countable"]:
            status_text = (
                "Phép toán này dùng để làm nổi bật chi tiết ảnh. "
                "Chọn Erosion, Dilation, Opening hoặc Closing để phân vùng đối tượng."
            )
        elif count is None:
            status_text = "Không có kết quả thống kê. Vui lòng kiểm tra module Statistics."
        else:
            raw = result["raw_count"]
            status_text = (f"Đã đếm {count} sản phẩm hợp lệ từ {raw} vùng sau Watershed."
                           if result.get("used_watershed") else
                           f"Đã đếm {count} sản phẩm hợp lệ từ {raw} vùng liên thông.")
        self.status_label.configure(
            text=status_text,
            text_color=WARNING if not result["countable"] else MUTED,
        )

    def _show_object_details(self):
        """Hiển thị các thống kê do Minh Anh trả về, không tính toán lại nhãn."""
        result = self.last_result
        if result is None or result.get("stats") is None:
            messagebox.showinfo("Chưa có thống kê", "Hãy mở và xử lý ảnh trước khi xem chi tiết.")
            return
        stats = result["stats"]
        objects = stats["objects"]
        if getattr(self, "_details_window", None) is not None and self._details_window.winfo_exists():
            self._details_window.destroy()

        dialog = ctk.CTkToplevel(self)
        self._details_window = dialog
        dialog.title("Thống kê sản phẩm")
        dialog.geometry("850x560")
        dialog.minsize(650, 400)
        dialog.configure(fg_color=BG)
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(2, weight=1)

        self._label(dialog, f"{stats['count']} sản phẩm hợp lệ", 21, True).grid(
            row=0, column=0, sticky="w", padx=22, pady=(20, 4))
        self._label(dialog, "Diện tích tính bằng pixel² · Tâm và khung bao tính bằng pixel",
                    11, color=MUTED).grid(row=1, column=0, sticky="w", padx=22, pady=(0, 8))

        text_box = ctk.CTkTextbox(dialog, fg_color=SURFACE, text_color=FG,
                                  border_color=BORDER, border_width=1,
                                  font=("Consolas", 12), wrap="none")
        text_box.grid(row=2, column=0, sticky="nsew", padx=22, pady=10)
        lines = [f"{'STT':>4}  {'ID':>6}  {'AREA':>9}  {'TÂM (X,Y)':>18}  {'BBOX (X,Y,W,H)':>22}", "─" * 77]
        for i, obj in enumerate(objects[:300], 1):
            cx, cy = obj["centroid"]
            x, y, width, height = obj["bbox"]
            lines.append(f"{i:>4}  {obj['id']:>6}  {obj['area']:>9}  "
                         f"{f'({cx:.1f}, {cy:.1f})':>18}  {f'({x}, {y}, {width}, {height})':>22}")
        if len(objects) > 300:
            lines.append(f"\nChỉ hiển thị 300/{len(objects)} đối tượng. Xuất CSV để xem toàn bộ.")
        if not objects:
            lines.append("Không có vùng nào thỏa điều kiện Min/Max Area.")
        text_box.insert("1.0", "\n".join(lines))
        text_box.configure(state="disabled")

        actions = ctk.CTkFrame(dialog, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="e", padx=22, pady=(4, 18))
        ctk.CTkButton(actions, text="Xuất CSV", fg_color=ACCENT,
                      hover_color="#25B99F", text_color="#0B211B",
                      command=self._export_statistics_csv).pack(side="left", padx=(0, 10))
        ctk.CTkButton(actions, text="Đóng", fg_color=SURFACE_2,
                      command=dialog.destroy).pack(side="left")
        dialog.lift()
        dialog.focus_force()

    def _export_statistics_csv(self):
        """Xuất đầy đủ đối tượng hợp lệ sang bảng CSV có thể mở trong Excel."""
        result = self.last_result
        if result is None or result.get("stats") is None:
            messagebox.showinfo("Chưa có thống kê", "Chưa có dữ liệu để xuất CSV.")
            return
        stem = self.image_path.stem if self.image_path else "result"
        path = filedialog.asksaveasfilename(
            title="Xuất thống kê sản phẩm", defaultextension=".csv",
            initialfile=f"{stem}_statistics.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["STT", "Label ID", "Area (pixel)", "Centroid X", "Centroid Y",
                                 "BBox X", "BBox Y", "BBox Width", "BBox Height"])
                for i, obj in enumerate(result["stats"]["objects"], 1):
                    cx, cy = obj["centroid"]
                    x, y, width, height = obj["bbox"]
                    writer.writerow([i, obj["id"], obj["area"], cx, cy, x, y, width, height])
            self.status_label.configure(text=f"Đã xuất thống kê: {path}", text_color=ACCENT)
        except Exception as exc:
            messagebox.showerror("Không xuất được CSV", str(exc))

    def fit_all(self):
        for view in self.views.values():view["canvas"].fit_image()

    def _save_image(self):
        result=self.last_result
        if result is None:
            messagebox.showinfo("Chưa có ảnh", "Mở và xử lý ảnh trước khi lưu.")
            return
        if result["annotated_bgr"] is not None:
            image=result["annotated_bgr"]
            suffix="_counted.png"
        elif result["colored_rgb"] is not None:
            image=cv2.cvtColor(result["colored_rgb"],cv2.COLOR_RGB2BGR)
            suffix="_components.png"
        else:
            image=cv2.cvtColor(result["morphology"],cv2.COLOR_GRAY2BGR)
            suffix="_morphology.png"
        stem=self.image_path.stem if self.image_path else "result"
        path=filedialog.asksaveasfilename(title="Lưu ảnh kết quả",defaultextension=".png",
              initialfile=f"{stem}{suffix}",
              filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp")])
        if path:
            try:
                imwrite_unicode(path,image)
                self.status_label.configure(text=f"Đã lưu: {path}",text_color=ACCENT)
            except Exception as e:
                messagebox.showerror("Lỗi lưu ảnh",str(e))

    def _on_close(self):
        self._closing=True
        if self._pending_process is not None:self.after_cancel(self._pending_process)
        self._executor.shutdown(wait=False,cancel_futures=True)
        self.destroy()
