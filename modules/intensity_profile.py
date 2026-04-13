from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

from .base_module import FeatureModule


@dataclass
class ProfileSettings:
    fixed_axis: str = "y"
    fixed_index: Optional[int] = None
    fixed_ratio: float = 0.50
    scan_start_ratio: float = 0.25
    scan_end_ratio: float = 0.75
    window_size: int = 11


class ImageData:
    def __init__(self, path: Path, gray: np.ndarray):
        self.path = path
        self.gray = gray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.gray.shape


class ImageLoader:
    @staticmethod
    def load_grayscale(path: Path) -> ImageData:
        pil_img = Image.open(path)
        arr = np.asarray(pil_img)

        if arr.ndim == 2:
            gray = arr
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                gray = arr[:, :, 0]
            else:
                rgb = arr[:, :, :3].astype(np.float32)
                gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        return ImageData(path=path, gray=np.asarray(gray, dtype=np.float32))


class IntensityProfileAnalyzer:
    @staticmethod
    def resolve_fixed_index(length: int, fixed_index: Optional[int], fixed_ratio: float) -> int:
        idx = int(fixed_index) if fixed_index is not None else int(length * fixed_ratio)
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

        if fixed_axis == "y":
            if not (0 <= fixed_index < w):
                raise ValueError(f"fixed_index out of range for y-axis: {fixed_index}")
            scan_start = int(np.clip(scan_start, 0, h - 1))
            scan_end = int(np.clip(scan_end, 0, h - 1))
            y0 = max(0, fixed_index - half)
            y1 = min(w - 1, fixed_index + half)

            indices = np.arange(scan_start, scan_end + 1)
            values = np.array([np.mean(image[x, y0 : y1 + 1]) for x in indices], dtype=np.float64)

        elif fixed_axis == "x":
            if not (0 <= fixed_index < h):
                raise ValueError(f"fixed_index out of range for x-axis: {fixed_index}")
            scan_start = int(np.clip(scan_start, 0, w - 1))
            scan_end = int(np.clip(scan_end, 0, w - 1))
            x0 = max(0, fixed_index - half)
            x1 = min(h - 1, fixed_index + half)

            indices = np.arange(scan_start, scan_end + 1)
            values = np.array([np.mean(image[x0 : x1 + 1, y]) for y in indices], dtype=np.float64)

        else:
            raise ValueError("fixed_axis must be 'x' or 'y'")

        return indices, values

    def compute(self, image: np.ndarray, settings: ProfileSettings) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
        h, w = image.shape

        if settings.fixed_axis == "y":
            fixed_index = self.resolve_fixed_index(w, settings.fixed_index, settings.fixed_ratio)
            scan_start, scan_end = self.resolve_scan_range(h, settings.scan_start_ratio, settings.scan_end_ratio)
        elif settings.fixed_axis == "x":
            fixed_index = self.resolve_fixed_index(h, settings.fixed_index, settings.fixed_ratio)
            scan_start, scan_end = self.resolve_scan_range(w, settings.scan_start_ratio, settings.scan_end_ratio)
        else:
            raise ValueError("fixed_axis must be 'x' or 'y'")

        indices, values = self.averaged_profile(
            image=image,
            fixed_axis=settings.fixed_axis,
            fixed_index=fixed_index,
            scan_start=scan_start,
            scan_end=scan_end,
            window_size=settings.window_size,
        )

        return fixed_index, scan_start, scan_end, indices, values


class IntensityProfileModule(FeatureModule):
    name = "Intensity Profile"

    def __init__(self, notebook: ttk.Notebook) -> None:
        super().__init__(notebook)
        self.image_data: Optional[ImageData] = None
        self.analyzer = IntensityProfileAnalyzer()

        self.fixed_axis_var = tk.StringVar(value="y")
        self.fixed_ratio_var = tk.DoubleVar(value=0.50)
        self.start_ratio_var = tk.DoubleVar(value=0.25)
        self.end_ratio_var = tk.DoubleVar(value=0.75)
        self.window_size_var = tk.IntVar(value=11)
        self.fixed_index_mode_var = tk.StringVar(value="ratio")
        self.fixed_index_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Ready. Load an image to start.")

        self.last_profile_indices: Optional[np.ndarray] = None
        self.last_profile_values: Optional[np.ndarray] = None

    def build(self) -> None:
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main, text="Control Panel", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.build_controls(control_frame)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.fig, (self.ax_image, self.ax_profile) = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        status_label = ttk.Label(self.frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=tk.X, pady=(8, 0))

        self.draw_empty()

    def build_controls(self, parent: ttk.LabelFrame) -> None:
        ttk.Button(parent, text="Load Image", command=self.on_load_image).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Fixed Axis").pack(anchor="w")
        axis_combo = ttk.Combobox(parent, textvariable=self.fixed_axis_var, state="readonly", values=["x", "y"])
        axis_combo.pack(fill=tk.X, pady=(0, 10))
        axis_combo.bind("<<ComboboxSelected>>", lambda _: self.refresh_fixed_index_limit())

        mode_frame = ttk.LabelFrame(parent, text="Fixed Index Mode", padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(mode_frame, text="Use Ratio", variable=self.fixed_index_mode_var, value="ratio").pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Use Index", variable=self.fixed_index_mode_var, value="index").pack(anchor="w")

        ttk.Label(parent, text="Fixed Ratio [0, 1]").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.fixed_ratio_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Fixed Index").pack(anchor="w")
        self.index_spin = ttk.Spinbox(parent, from_=0, to=0, textvariable=self.fixed_index_var)
        self.index_spin.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Scan Start Ratio [0, 1]").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.start_ratio_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Scan End Ratio [0, 1]").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.end_ratio_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Window Size (odd)").pack(anchor="w")
        ttk.Spinbox(parent, from_=1, to=101, increment=2, textvariable=self.window_size_var).pack(fill=tk.X, pady=(0, 14))

        ttk.Button(parent, text="Plot Profile", command=self.on_plot).pack(fill=tk.X)
        ttk.Button(parent, text="Save Plot", command=self.on_save_plot).pack(fill=tk.X, pady=(8, 0))

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

    def on_load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            self.image_data = ImageLoader.load_grayscale(Path(file_path))
            self.refresh_fixed_index_limit()
            self.status_var.set(f"Loaded: {self.image_data.path.name}, shape={self.image_data.shape}")
            self.on_plot()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def refresh_fixed_index_limit(self) -> None:
        if self.image_data is None:
            self.index_spin.configure(from_=0, to=0)
            self.fixed_index_var.set(0)
            return

        h, w = self.image_data.shape
        axis = self.fixed_axis_var.get()
        max_idx = (w - 1) if axis == "y" else (h - 1)

        self.index_spin.configure(from_=0, to=max_idx)
        self.fixed_index_var.set(int(np.clip(self.fixed_index_var.get(), 0, max_idx)))

    def read_settings(self) -> ProfileSettings:
        ratio = float(self.fixed_ratio_var.get())
        start = float(self.start_ratio_var.get())
        end = float(self.end_ratio_var.get())
        window = int(self.window_size_var.get())

        if not (0.0 <= ratio <= 1.0 and 0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
            raise ValueError("All ratios must be within [0, 1].")

        if window < 1 or window % 2 == 0:
            raise ValueError("Window size must be an odd integer >= 1.")

        fixed_index = int(self.fixed_index_var.get()) if self.fixed_index_mode_var.get() == "index" else None

        return ProfileSettings(
            fixed_axis=self.fixed_axis_var.get(),
            fixed_index=fixed_index,
            fixed_ratio=ratio,
            scan_start_ratio=start,
            scan_end_ratio=end,
            window_size=window,
        )

    def on_plot(self) -> None:
        if self.image_data is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            settings = self.read_settings()
            fixed_index, scan_start, scan_end, indices, values = self.analyzer.compute(self.image_data.gray, settings)
            self.last_profile_indices = indices
            self.last_profile_values = values

            self.ax_image.clear()
            self.ax_image.imshow(self.image_data.gray, cmap="gray")
            if settings.fixed_axis == "y":
                self.ax_image.plot([fixed_index, fixed_index], [scan_start, scan_end], color="red", linewidth=2)
                x_label = "X Position"
            else:
                self.ax_image.plot([scan_start, scan_end], [fixed_index, fixed_index], color="red", linewidth=2)
                x_label = "Y Position"
            self.ax_image.set_title(f"Image ({self.image_data.path.name})")
            self.ax_image.axis("off")

            self.ax_profile.clear()
            self.ax_profile.plot(indices, values, color="tab:blue", linewidth=2)
            self.ax_profile.set_title("Averaged Intensity Profile")
            self.ax_profile.set_xlabel(x_label)
            self.ax_profile.set_ylabel("Intensity")
            self.ax_profile.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw_idle()

            self.status_var.set(
                f"Plotted profile: axis={settings.fixed_axis}, fixed_index={fixed_index}, scan=[{scan_start}, {scan_end}], window={settings.window_size}"
            )
        except Exception as exc:
            messagebox.showerror("Plot Error", str(exc))

    def on_save_plot(self) -> None:
        if self.last_profile_values is None:
            messagebox.showwarning("Nothing to Save", "Please plot a profile first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All Files", "*.*")],
        )

        if not path:
            return

        try:
            self.fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.1)
            self.status_var.set(f"Saved plot to: {path}")
            messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def on_close(self) -> None:
        if hasattr(self, "fig"):
            plt.close(self.fig)
