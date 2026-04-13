from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .base_module import FeatureModule


@dataclass
class ProfileSettings:
    profile_mode: str = "axis"
    fixed_axis: str = "width"
    fixed_ratio: float = 0.50
    scan_start_ratio: float = 0.25
    scan_end_ratio: float = 0.75
    line_start: Optional[Tuple[float, float]] = None
    line_end: Optional[Tuple[float, float]] = None
    line_samples: int = 300
    window_size: int = 11


@dataclass
class ProfileResult:
    name: str
    fixed_index: Optional[int]
    scan_start: Optional[int]
    scan_end: Optional[int]
    indices: np.ndarray
    values: np.ndarray


class ImageData:
    def __init__(self, path: Path, display: np.ndarray, intensity: np.ndarray, is_rgb: bool):
        self.path = path
        self.display = display
        self.intensity = intensity
        self.is_rgb = is_rgb

    @property
    def shape(self) -> Tuple[int, int]:
        return self.intensity.shape


class ImageLoader:
    @staticmethod
    def load_image(path: Path) -> ImageData:
        arr = plt.imread(path)

        if arr.ndim == 2:
            intensity = np.asarray(arr, dtype=np.float32)
            display = arr
            is_rgb = False
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                intensity = np.asarray(arr[:, :, 0], dtype=np.float32)
                display = arr[:, :, 0]
                is_rgb = False
            else:
                rgb = arr[:, :, :3]
                display = rgb
                intensity = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
                is_rgb = True
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        return ImageData(path=path, display=display, intensity=np.asarray(intensity, dtype=np.float32), is_rgb=is_rgb)


class IntensityProfileAnalyzer:
    @staticmethod
    def bilinear_sample(image: np.ndarray, y: float, x: float) -> float:
        h, w = image.shape
        x = float(np.clip(x, 0.0, w - 1))
        y = float(np.clip(y, 0.0, h - 1))

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        dx = x - x0
        dy = y - y0

        v00 = float(image[y0, x0])
        v01 = float(image[y0, x1])
        v10 = float(image[y1, x0])
        v11 = float(image[y1, x1])

        top = (1.0 - dx) * v00 + dx * v01
        bottom = (1.0 - dx) * v10 + dx * v11
        return (1.0 - dy) * top + dy * bottom

    @staticmethod
    def resolve_fixed_index(length: int, fixed_ratio: float) -> int:
        idx = int(length * fixed_ratio)
        return int(np.clip(idx, 0, length - 1))

    @staticmethod
    def resolve_scan_range(length: int, start_ratio: float, end_ratio: float) -> Tuple[int, int]:
        start = int(np.clip(int(length * start_ratio), 0, length - 1))
        end = int(np.clip(int(length * end_ratio), 0, length - 1))

        if end < start:
            start, end = end, start
        if end == start and end < length - 1:
            end += 1

        return start, end

    @staticmethod
    def averaged_profile(
        image: np.ndarray,
        fixed_axis: str,
        fixed_index: int,
        scan_start: int,
        scan_end: int,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 1")

        half = window_size // 2
        h, w = image.shape

        if fixed_axis == "width":
            if not (0 <= fixed_index < w):
                raise ValueError(f"fixed_index out of range for width axis: {fixed_index}")
            scan_start = int(np.clip(scan_start, 0, h - 1))
            scan_end = int(np.clip(scan_end, 0, h - 1))
            w0 = max(0, fixed_index - half)
            w1 = min(w - 1, fixed_index + half)

            indices = np.arange(scan_start, scan_end + 1)
            values = np.array([np.mean(image[h_idx, w0 : w1 + 1]) for h_idx in indices], dtype=np.float64)

        elif fixed_axis == "height":
            if not (0 <= fixed_index < h):
                raise ValueError(f"fixed_index out of range for height axis: {fixed_index}")
            scan_start = int(np.clip(scan_start, 0, w - 1))
            scan_end = int(np.clip(scan_end, 0, w - 1))
            h0 = max(0, fixed_index - half)
            h1 = min(h - 1, fixed_index + half)

            indices = np.arange(scan_start, scan_end + 1)
            values = np.array([np.mean(image[h0 : h1 + 1, w_idx]) for w_idx in indices], dtype=np.float64)

        else:
            raise ValueError("fixed_axis must be 'height' or 'width'")

        return indices, values

    def compute_axis(self, image: np.ndarray, settings: ProfileSettings) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
        h, w = image.shape

        if settings.fixed_axis == "width":
            fixed_index = self.resolve_fixed_index(w, settings.fixed_ratio)
            scan_start, scan_end = self.resolve_scan_range(h, settings.scan_start_ratio, settings.scan_end_ratio)
        elif settings.fixed_axis == "height":
            fixed_index = self.resolve_fixed_index(h, settings.fixed_ratio)
            scan_start, scan_end = self.resolve_scan_range(w, settings.scan_start_ratio, settings.scan_end_ratio)
        else:
            raise ValueError("fixed_axis must be 'height' or 'width'")

        indices, values = self.averaged_profile(
            image=image,
            fixed_axis=settings.fixed_axis,
            fixed_index=fixed_index,
            scan_start=scan_start,
            scan_end=scan_end,
            window_size=settings.window_size,
        )

        return fixed_index, scan_start, scan_end, indices, values

    def compute_line(self, image: np.ndarray, settings: ProfileSettings) -> Tuple[None, None, None, np.ndarray, np.ndarray]:
        if settings.line_start is None or settings.line_end is None:
            raise ValueError("Please select line start and end points.")

        if settings.window_size < 1 or settings.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 1")

        h, w = image.shape
        x0 = settings.line_start[0] * max(w - 1, 1)
        y0 = settings.line_start[1] * max(h - 1, 1)
        x1 = settings.line_end[0] * max(w - 1, 1)
        y1 = settings.line_end[1] * max(h - 1, 1)

        dx = x1 - x0
        dy = y1 - y0
        length = float(np.hypot(dx, dy))
        if length < 1e-8:
            raise ValueError("Line length is too small. Please select two different points.")

        nx = -dy / length
        ny = dx / length

        samples = max(int(settings.line_samples), 2)
        t = np.linspace(0.0, 1.0, samples)
        values = np.empty(samples, dtype=np.float64)
        half = settings.window_size // 2

        for i, tv in enumerate(t):
            cx = x0 + dx * tv
            cy = y0 + dy * tv
            local_vals = []
            for offset in range(-half, half + 1):
                sx = cx + nx * offset
                sy = cy + ny * offset
                local_vals.append(self.bilinear_sample(image, sy, sx))
            values[i] = float(np.mean(local_vals))

        return None, None, None, t, values

    def compute(self, image: np.ndarray, settings: ProfileSettings) -> Tuple[Optional[int], Optional[int], Optional[int], np.ndarray, np.ndarray]:
        if settings.profile_mode == "axis":
            return self.compute_axis(image, settings)
        if settings.profile_mode == "line":
            return self.compute_line(image, settings)
        raise ValueError("profile_mode must be 'axis' or 'line'")


class IntensityProfileModule(FeatureModule):
    name = "Intensity Profile"

    def __init__(self, notebook: ttk.Notebook) -> None:
        super().__init__(notebook)
        self.images: list[ImageData] = []
        self.current_image_index = 0
        self.analyzer = IntensityProfileAnalyzer()

        self.profile_mode_var = tk.StringVar(value="axis")
        self.fixed_axis_var = tk.StringVar(value="width")
        self.fixed_ratio_var = tk.DoubleVar(value=0.50)
        self.start_ratio_var = tk.DoubleVar(value=0.25)
        self.end_ratio_var = tk.DoubleVar(value=0.75)
        self.line_samples_var = tk.IntVar(value=300)
        self.window_size_var = tk.IntVar(value=11)
        self.save_mode_var = tk.StringVar(value="Composite")
        self.status_var = tk.StringVar(value="Ready. Load an image to start.")

        self.line_start: Optional[Tuple[float, float]] = None
        self.line_end: Optional[Tuple[float, float]] = None

        self.last_plot_settings: Optional[ProfileSettings] = None
        self.last_fixed_index: Optional[int] = None
        self.last_scan_start: Optional[int] = None
        self.last_scan_end: Optional[int] = None
        self.profile_results: list[ProfileResult] = []

    def build(self) -> None:
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main, text="Control Panel", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.build_controls(control_frame)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.fig, (self.ax_image, self.ax_profile) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            dpi=120,
            gridspec_kw={"width_ratios": [1, 1]},
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        status_label = ttk.Label(self.frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=tk.X, pady=(8, 0))

        self.draw_empty()

    def build_controls(self, parent: ttk.LabelFrame) -> None:
        ttk.Button(parent, text="Add Images (1+)", command=self.on_load_images).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(parent, text="Clear Images", command=self.on_clear_images).pack(fill=tk.X, pady=(0, 10))

        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(nav_frame, text="Prev Image", command=self.on_prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next Image", command=self.on_next_image).pack(side=tk.RIGHT)
        self.image_nav_label = ttk.Label(parent, text="Current image: N/A")
        self.image_nav_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="Profile Mode").pack(anchor="w")
        mode_combo = ttk.Combobox(parent, textvariable=self.profile_mode_var, state="readonly", values=["axis", "line"])
        mode_combo.pack(fill=tk.X, pady=(0, 10))
        mode_combo.bind("<<ComboboxSelected>>", lambda _: self.update_mode_ui())

        self.axis_frame = ttk.LabelFrame(parent, text="Axis Parameters", padding=8)

        ttk.Label(self.axis_frame, text="Fixed Axis").pack(anchor="w")
        axis_combo = ttk.Combobox(self.axis_frame, textvariable=self.fixed_axis_var, state="readonly", values=["height", "width"])
        axis_combo.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.axis_frame, text="Fixed Ratio [0, 1]").pack(anchor="w")
        ttk.Entry(self.axis_frame, textvariable=self.fixed_ratio_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.axis_frame, text="Scan Start Ratio [0, 1]").pack(anchor="w")
        ttk.Entry(self.axis_frame, textvariable=self.start_ratio_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.axis_frame, text="Scan End Ratio [0, 1]").pack(anchor="w")
        ttk.Entry(self.axis_frame, textvariable=self.end_ratio_var).pack(fill=tk.X, pady=(0, 10))

        self.line_frame = ttk.LabelFrame(parent, text="Line Parameters", padding=8)

        ttk.Label(
            self.line_frame,
            text="Click two points on the left image:\nfirst click=start, second click=end.",
        ).pack(anchor="w", pady=(0, 6))

        self.line_points_label = ttk.Label(self.line_frame, text="Line points: not selected")
        self.line_points_label.pack(anchor="w", pady=(0, 6))

        ttk.Button(self.line_frame, text="Clear Line Points", command=self.on_clear_line_points).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(self.line_frame, text="Line Samples").pack(anchor="w")
        ttk.Spinbox(self.line_frame, from_=50, to=5000, increment=50, textvariable=self.line_samples_var).pack(fill=tk.X, pady=(0, 6))

        ttk.Label(parent, text="Window Size (odd)").pack(anchor="w")
        ttk.Spinbox(parent, from_=1, to=101, increment=2, textvariable=self.window_size_var).pack(fill=tk.X, pady=(0, 14))

        ttk.Label(parent, text="Save Mode").pack(anchor="w")
        ttk.Combobox(
            parent,
            textvariable=self.save_mode_var,
            state="readonly",
            values=["Composite", "ProfileOnly"],
        ).pack(fill=tk.X, pady=(0, 14))

        ttk.Button(parent, text="Plot Profile", command=self.on_plot).pack(fill=tk.X)
        ttk.Button(parent, text="Save Plot", command=self.on_save_plot).pack(fill=tk.X, pady=(8, 0))

        self.update_mode_ui()
        self.update_line_points_label()

    def update_mode_ui(self) -> None:
        if self.profile_mode_var.get() == "line":
            self.axis_frame.pack_forget()
            self.line_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.line_frame.pack_forget()
            self.axis_frame.pack(fill=tk.X, pady=(0, 10))

    def update_line_points_label(self) -> None:
        if self.line_start is None and self.line_end is None:
            text = "Line points: not selected"
        elif self.line_start is not None and self.line_end is None:
            text = f"Start: ({self.line_start[0]:.3f}, {self.line_start[1]:.3f})\nEnd: waiting click"
        else:
            start = self.line_start
            end = self.line_end
            if start is None or end is None:
                text = "Line points: not selected"
                self.line_points_label.config(text=text)
                return
            text = (
                f"Start: ({start[0]:.3f}, {start[1]:.3f})\n"
                f"End: ({end[0]:.3f}, {end[1]:.3f})"
            )
        self.line_points_label.config(text=text)

    def on_clear_line_points(self) -> None:
        self.line_start = None
        self.line_end = None
        self.update_line_points_label()
        self.plot_current_preview_only()

    def on_canvas_click(self, event) -> None:
        if self.profile_mode_var.get() != "line":
            return
        if event.inaxes != self.ax_image:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self.images:
            return

        x = float(np.clip(event.xdata, 0.0, 1.0))
        y = float(np.clip(event.ydata, 0.0, 1.0))

        if self.line_start is None or (self.line_start is not None and self.line_end is not None):
            self.line_start = (x, y)
            self.line_end = None
        else:
            self.line_end = (x, y)

        self.update_line_points_label()
        if self.line_start is not None and self.line_end is not None:
            self.on_plot()
        else:
            self.plot_current_preview_only()

    def draw_empty(self) -> None:
        self.ax_image.clear()
        self.ax_profile.clear()
        self.ax_image.set_title("Image Preview")
        self.ax_image.text(0.5, 0.5, "No image loaded", ha="center", va="center", transform=self.ax_image.transAxes)
        self.ax_image.axis("off")
        self.ax_profile.set_title("Intensity Profile")
        self.ax_profile.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_load_images(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select one or more images",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )

        if not file_paths:
            return

        try:
            new_images = [ImageLoader.load_image(Path(p)) for p in file_paths]

            self.validate_image_shapes(new_images)

            if not self.images:
                self.images = new_images
                self.current_image_index = 0
            else:
                self.images.extend(new_images)
                self.current_image_index = len(self.images) - len(new_images)

            self.update_image_nav_label()
            self.status_var.set(
                f"Added {len(new_images)} image(s). Total images: {len(self.images)}"
            )
            self.on_plot()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def validate_image_shapes(self, new_images: list[ImageData]) -> None:
        if not new_images:
            return

        if self.images:
            ref_h, ref_w = self.images[0].shape
        else:
            ref_h, ref_w = new_images[0].shape

        ref_ratio = ref_h / ref_w
        tolerance = 1e-6

        for image in new_images:
            h, w = image.shape
            ratio = h / w
            if abs(ratio - ref_ratio) > tolerance:
                raise ValueError(
                    "All images must have the same height/width ratio. "
                    f"Expected ratio {ref_h}/{ref_w}, got {h}/{w} ({image.path.name})."
                )

    def on_clear_images(self) -> None:
        self.images = []
        self.current_image_index = 0
        self.profile_results = []
        self.last_plot_settings = None
        self.last_fixed_index = None
        self.last_scan_start = None
        self.last_scan_end = None
        self.update_image_nav_label()
        self.draw_empty()
        self.status_var.set("Cleared all loaded images.")

    def update_image_nav_label(self) -> None:
        if not self.images:
            self.image_nav_label.config(text="Current image: N/A")
            return

        current = self.images[self.current_image_index]
        self.image_nav_label.config(
            text=(
                f"Current image: {self.current_image_index + 1}/{len(self.images)}\n"
                f"{self.format_display_name(current.path.name)}"
            )
        )

    def format_display_name(self, filename: str) -> str:
        path = Path(filename)
        stem = path.stem[:6]
        suffix = path.suffix
        if suffix:
            return f"{stem}.{suffix}"
        return stem

    def on_prev_image(self) -> None:
        if not self.images:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        self.update_image_nav_label()
        self.plot_current_preview_only()

    def on_next_image(self) -> None:
        if not self.images:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.update_image_nav_label()
        self.plot_current_preview_only()

    def current_image(self) -> Optional[ImageData]:
        if not self.images:
            return None
        return self.images[self.current_image_index]

    def read_settings(self) -> ProfileSettings:
        mode = self.profile_mode_var.get()
        ratio = float(self.fixed_ratio_var.get())
        start = float(self.start_ratio_var.get())
        end = float(self.end_ratio_var.get())
        line_samples = int(self.line_samples_var.get())
        window = int(self.window_size_var.get())

        if mode == "axis":
            if not (0.0 <= ratio <= 1.0 and 0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
                raise ValueError("All ratios must be within [0, 1].")
        elif mode == "line":
            if self.line_start is None or self.line_end is None:
                raise ValueError("Please click two points on the left image for line mode.")
            if line_samples < 2:
                raise ValueError("Line samples must be >= 2.")
        else:
            raise ValueError("Profile mode must be 'axis' or 'line'.")

        if window < 1 or window % 2 == 0:
            raise ValueError("Window size must be an odd integer >= 1.")

        return ProfileSettings(
            profile_mode=mode,
            fixed_axis=self.fixed_axis_var.get(),
            fixed_ratio=ratio,
            scan_start_ratio=start,
            scan_end_ratio=end,
            line_start=self.line_start,
            line_end=self.line_end,
            line_samples=line_samples,
            window_size=window,
        )

    def on_plot(self) -> None:
        if not self.images:
            messagebox.showwarning("No Image", "Please load one or more images first.")
            return

        try:
            settings = self.read_settings()
            self.profile_results = []

            for image in self.images:
                fixed_index, scan_start, scan_end, indices, values = self.analyzer.compute(image.intensity, settings)
                self.profile_results.append(
                    ProfileResult(
                        name=image.path.name,
                        fixed_index=fixed_index,
                        scan_start=scan_start,
                        scan_end=scan_end,
                        indices=indices,
                        values=values,
                    )
                )

            self.last_plot_settings = settings
            self.last_fixed_index = self.profile_results[0].fixed_index
            self.last_scan_start = self.profile_results[0].scan_start
            self.last_scan_end = self.profile_results[0].scan_end

            self.draw_current_image_panel()
            self.draw_profile_panel()
            self.fig.tight_layout()
            self.canvas.draw_idle()

            if settings.profile_mode == "axis":
                self.status_var.set(
                    f"Plotted {len(self.profile_results)} profile(s): mode=axis, axis={settings.fixed_axis}, fixed_index={self.last_fixed_index}, scan=[{self.last_scan_start}, {self.last_scan_end}], window={settings.window_size}"
                )
            else:
                self.status_var.set(
                    f"Plotted {len(self.profile_results)} profile(s): mode=line, samples={settings.line_samples}, window={settings.window_size}"
                )
        except Exception as exc:
            messagebox.showerror("Plot Error", str(exc))

    def plot_current_preview_only(self) -> None:
        if not self.images or not self.profile_results:
            return
        self.draw_current_image_panel()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def draw_current_image_panel(self) -> None:
        image = self.current_image()
        if image is None:
            return

        overlay_mode = self.profile_mode_var.get()
        fixed_index: Optional[int] = None
        scan_start: Optional[int] = None
        scan_end: Optional[int] = None

        if overlay_mode == "axis":
            try:
                preview_settings = self.read_settings()
                fixed_index, scan_start, scan_end, _, _ = self.analyzer.compute(image.intensity, preview_settings)
            except Exception:
                if self.last_plot_settings is not None and self.last_plot_settings.profile_mode == "axis":
                    fixed_index, scan_start, scan_end, _, _ = self.analyzer.compute(image.intensity, self.last_plot_settings)

        h, w = image.shape
        self.ax_image.clear()

        # Force a normalized display frame so different-resolution images occupy
        # the same visual area when switching.
        extent = (0.0, 1.0, 1.0, 0.0)
        if image.is_rgb:
            self.ax_image.imshow(image.display, extent=extent)
        else:
            self.ax_image.imshow(image.display, cmap="gray", extent=extent)
        self.ax_image.set_aspect("auto")

        width_den = max(w - 1, 1)
        height_den = max(h - 1, 1)

        if overlay_mode == "axis":
            if fixed_index is None or scan_start is None or scan_end is None:
                pass
            elif self.fixed_axis_var.get() == "width":
                x_fixed = fixed_index / width_den
                y_start = scan_start / height_den
                y_end = scan_end / height_den
                self.ax_image.plot([x_fixed, x_fixed], [y_start, y_end], color="red", linewidth=2)
            else:
                y_fixed = fixed_index / height_den
                x_start = scan_start / width_den
                x_end = scan_end / width_den
                self.ax_image.plot([x_start, x_end], [y_fixed, y_fixed], color="red", linewidth=2)
        else:
            if self.line_start is not None:
                self.ax_image.plot(self.line_start[0], self.line_start[1], marker="o", color="red", markersize=5)
            if self.line_start is not None and self.line_end is not None:
                self.ax_image.plot(
                    [self.line_start[0], self.line_end[0]],
                    [self.line_start[1], self.line_end[1]],
                    color="red",
                    linewidth=2,
                )
                self.ax_image.plot(self.line_end[0], self.line_end[1], marker="o", color="red", markersize=5)

        self.ax_image.set_xlim(0.0, 1.0)
        self.ax_image.set_ylim(1.0, 0.0)

        self.ax_image.set_title(f"Image")
        self.ax_image.axis("off")

    def draw_profile_panel(self) -> None:
        if not self.profile_results or self.last_plot_settings is None:
            return

        self.ax_profile.clear()

        # Use normalized x-axis when images have different profile lengths.
        lengths = [len(item.indices) for item in self.profile_results]
        use_normalized_x = len(set(lengths)) != 1

        for item in self.profile_results:
            indices = np.asarray(item.indices)
            values = np.asarray(item.values)
            if use_normalized_x:
                x = np.linspace(0.0, 1.0, len(indices))
            else:
                x = indices
            self.ax_profile.plot(x, values, linewidth=1.8, label=self.format_display_name(item.name))

        self.ax_profile.set_title("Averaged Intensity Profile")
        if self.last_plot_settings.profile_mode == "line":
            self.ax_profile.set_xlabel("Line Position (0-1)")
        elif use_normalized_x:
            self.ax_profile.set_xlabel("Relative Position (0-1)")
        else:
            self.ax_profile.set_xlabel(
                "Height Position" if self.last_plot_settings.fixed_axis == "width" else "Width Position"
            )
        self.ax_profile.set_ylabel("Intensity")
        self.ax_profile.grid(True, alpha=0.3)
        if len(self.profile_results) > 1:
            self.ax_profile.legend(fontsize=8)

    def on_save_plot(self) -> None:
        if not self.profile_results:
            messagebox.showwarning("Nothing to Save", "Please plot a profile first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            initialfile=f"intensity-profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All Files", "*.*")],
        )

        if not path:
            return

        try:
            if self.save_mode_var.get() == "ProfileOnly":
                profile_fig = plt.figure(figsize=(7, 4), dpi=180)
                profile_ax = profile_fig.add_subplot(111)

                lengths = [len(item.indices) for item in self.profile_results]
                use_normalized_x = len(set(lengths)) != 1

                for item in self.profile_results:
                    indices = np.asarray(item.indices)
                    values = np.asarray(item.values)
                    if use_normalized_x:
                        x = np.linspace(0.0, 1.0, len(indices))
                    else:
                        x = indices
                    profile_ax.plot(x, values, linewidth=1.8, label=self.format_display_name(item.name))

                profile_ax.set_title("Averaged Intensity Profile")
                if self.last_plot_settings is not None and self.last_plot_settings.profile_mode == "line":
                    profile_ax.set_xlabel("Line Position (0-1)")
                elif use_normalized_x:
                    profile_ax.set_xlabel("Relative Position (0-1)")
                elif self.last_plot_settings is not None:
                    profile_ax.set_xlabel(
                        "Height Position" if self.last_plot_settings.fixed_axis == "width" else "Width Position"
                    )
                profile_ax.set_ylabel("Intensity")
                profile_ax.grid(True, alpha=0.3)
                if len(self.profile_results) > 1:
                    profile_ax.legend(fontsize=8)

                profile_fig.tight_layout()
                profile_fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.1)
                plt.close(profile_fig)
            else:
                self.fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.1)

            self.status_var.set(f"Saved plot to: {path}")
            messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def on_close(self) -> None:
        if hasattr(self, "fig"):
            plt.close(self.fig)
