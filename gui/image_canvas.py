"""Canvas ảnh hỗ trợ zoom bằng con lăn, kéo ảnh (pan), vừa khung (fit).
Chỉ thay đổi cách hiển thị: không làm thay đổi pixel hay ảnh được đếm.
"""
from __future__ import annotations
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class ZoomCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, background="#101822", highlightthickness=0,
                         borderwidth=0, cursor="fleur", **kwargs)
        self._image: Image.Image | None = None
        self._tk_image = None
        self._image_id = None
        self._zoom = 1.0
        self._offset_x = 0.
        self._offset_y = 0.
        self._fitted = True
        self._drag = None
        self.bind("<Configure>", self._resized)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.12))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1/1.12))
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Double-Button-1>", lambda e: self.fit_image())
        self._draw_message()

    def _draw_message(self):
        self.delete("all")
        self.create_text(max(self.winfo_width()/2,140), max(self.winfo_height()/2,90),
                         text="Chưa có ảnh", fill="#6F8094", font=("Segoe UI", 12))

    def set_image(self, image: np.ndarray | None, rgb: bool = False):
        """Đầu vào ndarray grayscale, RGB (rgb=True), hoặc BGR (rgb=False)."""
        if image is None:
            self._image = None
            self._tk_image = None
            self._draw_message()
            return
        if image.ndim == 2:
            pil = Image.fromarray(image, mode="L")
        elif image.ndim == 3 and image.shape[2] == 3:
            if not rgb:
                image = image[:, :, ::-1]
            pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        else:
            raise ValueError("Ảnh xem trước phải là gray hoặc 3 kênh")
        self._image = pil
        self._fitted = True
        self.fit_image()

    def _resized(self, _event=None):
        if self._image is None:
            self._draw_message()
        elif self._fitted:
            self.fit_image()
        else:
            self._render()

    def fit_image(self):
        if self._image is None:
            return
        width = max(60, self.winfo_width() - 24)
        height = max(60, self.winfo_height() - 24)
        self._zoom = min(width/self._image.width, height/self._image.height, 4.0)
        self._offset_x = self.winfo_width()/2 - (self._image.width*self._zoom)/2
        self._offset_y = self.winfo_height()/2 - (self._image.height*self._zoom)/2
        self._fitted = True
        self._render()

    def zoom_step(self, factor: float):
        self._zoom_at(self.winfo_width()/2, self.winfo_height()/2, factor)

    def _wheel(self, event):
        if event.delta:
            self._zoom_at(event.x, event.y, 1.12 if event.delta > 0 else 1/1.12)

    def _zoom_at(self, x: int, y: int, factor: float):
        if self._image is None:
            return
        newzoom = max(0.03, min(8.0, 6000/max(self._image.size), self._zoom * factor))
        if abs(newzoom - self._zoom) < 1e-8:
            return
        # Điểm ảnh dưới con trỏ không đổi vị trí khi zoom
        src_x = (x-self._offset_x) / self._zoom
        src_y = (y-self._offset_y) / self._zoom
        self._zoom = newzoom
        self._offset_x = x-src_x*newzoom
        self._offset_y = y-src_y*newzoom
        self._fitted = False
        self._render()

    def _on_press(self, event):
        self._drag = (event.x, event.y)

    def _on_drag(self, event):
        if self._image is None or self._drag is None:
            return
        x, y = self._drag
        self._offset_x += event.x-x
        self._offset_y += event.y-y
        self._drag = (event.x,event.y)
        self._fitted = False
        # Pan chỉ dịch chuyển ảnh đang hiển thị, không resize ảnh mỗi pixel kéo.
        if self._image_id is not None:
            self.move(self._image_id, event.x-x, event.y-y)

    def _on_release(self, _event):
        self._drag = None

    def _render(self):
        if self._image is None:
            self._draw_message()
            return
        width = max(1, int(self._image.width*self._zoom))
        height = max(1, int(self._image.height*self._zoom))
        # Giới hạn bộ nhớ thumbnail khi zoom lớn với ảnh độ phân giải cao
        if width > 6000 or height > 6000:
            ratio = min(6000/width, 6000/height)
            width, height = max(1,int(width*ratio)), max(1,int(height*ratio))
        im = self._image.resize((width,height), Image.Resampling.NEAREST if self._zoom>2 else Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(im)
        self.delete("all")
        self._image_id = self.create_image(self._offset_x, self._offset_y,
                                           anchor="nw", image=self._tk_image)
