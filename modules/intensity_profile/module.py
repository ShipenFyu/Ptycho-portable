from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..base_module import FeatureModule
from .analyzer import IntensityProfileAnalyzer
from .image_loader import ImageLoader
from .models import ImageData, ProfileResult, ProfileSettings


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
        self.multi_line_points: dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

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
        mode_combo = ttk.Combobox(parent, textvariable=self.profile_mode_var, state="readonly", values=["axis", "line", "multi"])
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

        self.line_instruction_var = tk.StringVar(
            value="Click two points on the left image:\nfirst click=start, second click=end."
        )
        self.line_instruction_label = ttk.Label(
            self.line_frame,
            textvariable=self.line_instruction_var,
            wraplength=250,
            justify="left",
        )
        self.line_instruction_label.pack(anchor="w", pady=(0, 6))

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
        mode = self.profile_mode_var.get()
        if mode in ("line", "multi"):
            self.axis_frame.pack_forget()
            self.line_frame.pack(fill=tk.X, pady=(0, 10))
            if mode == "multi":
                self.line_start = None
                self.line_end = None
                self.line_instruction_var.set(
                    "Click two points on the left image:\nrepeat this for each image."
                )
            else:
                self.line_instruction_var.set(
                    "Click two points on the left image:\nfirst click=start, second click=end."
                )
        else:
            self.line_frame.pack_forget()
            self.axis_frame.pack(fill=tk.X, pady=(0, 10))
        self.update_line_points_label()

    def update_line_points_label(self) -> None:
        mode = self.profile_mode_var.get()
        if mode == "multi":
            points = self.multi_line_points.get(self.current_image_index)
            if points is None:
                text = "Current image line: not selected"
            else:
                start, end = points
                text = (
                    f"Current image line\n"
                    f"Start: ({start[0]:.3f}, {start[1]:.3f})\n"
                    f"End: ({end[0]:.3f}, {end[1]:.3f})"
                )
            self.line_points_label.config(text=text)
            return

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
        if self.profile_mode_var.get() == "multi":
            self.multi_line_points.pop(self.current_image_index, None)
            self.line_start = None
            self.line_end = None
        else:
            self.line_start = None
            self.line_end = None
        self.clear_profile_results()
        self.update_line_points_label()
        self.plot_current_preview_only()

    def get_profile_color(self, index: int) -> tuple:
        cmap = plt.get_cmap("tab10")
        return cmap(index % 10)

    def on_canvas_click(self, event) -> None:
        mode = self.profile_mode_var.get()
        if mode not in ("line", "multi"):
            return
        if event.inaxes != self.ax_image:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self.images:
            return

        x = float(np.clip(event.xdata, 0.0, 1.0))
        y = float(np.clip(event.ydata, 0.0, 1.0))

        if mode == "multi":
            if self.line_start is None or self.line_end is not None:
                self.line_start = (x, y)
                self.line_end = None
                self.multi_line_points.pop(self.current_image_index, None)
                self.clear_profile_results()
            else:
                self.line_end = (x, y)
        else:
            if self.line_start is None or (self.line_start is not None and self.line_end is not None):
                self.line_start = (x, y)
                self.line_end = None
                self.clear_profile_results()
            else:
                self.line_end = (x, y)

        if mode == "multi" and self.line_start is not None and self.line_end is not None:
            self.multi_line_points[self.current_image_index] = (self.line_start, self.line_end)

        self.update_line_points_label()
        if self.line_start is not None and self.line_end is not None:
            if mode == "multi":
                missing_count = sum(1 for idx in range(len(self.images)) if idx not in self.multi_line_points)
                if missing_count == 0:
                    self.on_plot()
                else:
                    self.status_var.set(
                        f"Multi mode: line saved for image {self.current_image_index + 1}/{len(self.images)}. Remaining images without lines: {missing_count}."
                    )
                    self.plot_current_preview_only()
            else:
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
            mode = self.profile_mode_var.get()
            if mode == "multi":
                self.line_start = None
                self.line_end = None
                self.clear_profile_results()
                self.status_var.set(
                    f"Added {len(new_images)} image(s). Total images: {len(self.images)}. Multi mode: draw one line per image."
                )
                self.plot_current_preview_only()
            elif mode == "line" and (self.line_start is None or self.line_end is None):
                self.clear_profile_results()
                self.status_var.set(
                    f"Added {len(new_images)} image(s). Total images: {len(self.images)}. Line mode: click two points to plot."
                )
                self.update_line_points_label()
                self.plot_current_preview_only()
            else:
                self.status_var.set(
                    f"Added {len(new_images)} image(s). Total images: {len(self.images)}"
                )
                self.on_plot()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def validate_image_shapes(self, new_images: list[ImageData]) -> None:
        if not new_images:
            return

        mode = self.profile_mode_var.get()
        if mode == "multi":
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
        self.line_start = None
        self.line_end = None
        self.multi_line_points = {}
        self.clear_profile_results()
        self.update_image_nav_label()
        self.draw_empty()
        self.status_var.set("Cleared all loaded images.")

    def clear_profile_results(self) -> None:
        self.profile_results = []
        self.last_plot_settings = None
        self.last_fixed_index = None
        self.last_scan_start = None
        self.last_scan_end = None

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

    def format_plot_label(self, filename: str, index: int) -> str:
        """Return an ASCII-safe label for Matplotlib text rendering."""
        label = self.format_display_name(filename)
        if all(ord(ch) < 128 for ch in label):
            return label
        return f"img_{index + 1}"

    def on_prev_image(self) -> None:
        if not self.images:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        if self.profile_mode_var.get() == "multi":
            self.line_start = None
            self.line_end = None
        self.update_image_nav_label()
        self.update_line_points_label()
        self.plot_current_preview_only()

    def on_next_image(self) -> None:
        if not self.images:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        if self.profile_mode_var.get() == "multi":
            self.line_start = None
            self.line_end = None
        self.update_image_nav_label()
        self.update_line_points_label()
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
        elif mode == "multi":
            if line_samples < 2:
                raise ValueError("Line samples must be >= 2.")
            if not self.images:
                raise ValueError("Please load one or more images first.")
            missing = [
                self.images[idx].path.name
                for idx in range(len(self.images))
                if idx not in self.multi_line_points
            ]
            if missing:
                names = ", ".join(missing[:3])
                if len(missing) > 3:
                    names += f" ... (+{len(missing) - 3} more)"
                raise ValueError(
                    "Multi mode requires one line per image. "
                    f"Missing line on: {names}"
                )
        else:
            raise ValueError("Profile mode must be 'axis', 'line', or 'multi'.")

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

            for idx, image in enumerate(self.images):
                image_settings = settings
                if settings.profile_mode == "multi":
                    start, end = self.multi_line_points[idx]
                    image_settings = ProfileSettings(
                        profile_mode="multi",
                        fixed_axis=settings.fixed_axis,
                        fixed_ratio=settings.fixed_ratio,
                        scan_start_ratio=settings.scan_start_ratio,
                        scan_end_ratio=settings.scan_end_ratio,
                        line_start=start,
                        line_end=end,
                        line_samples=settings.line_samples,
                        window_size=settings.window_size,
                    )

                fixed_index, scan_start, scan_end, indices, values = self.analyzer.compute(image.intensity, image_settings)

                if settings.profile_mode == "multi":
                    vmin = float(np.min(values))
                    vmax = float(np.max(values))
                    if vmax - vmin > 1e-12:
                        values = (values - vmin) / (vmax - vmin)
                    else:
                        values = np.zeros_like(values, dtype=np.float64)

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
            elif settings.profile_mode == "line":
                self.status_var.set(
                    f"Plotted {len(self.profile_results)} profile(s): mode=line, samples={settings.line_samples}, window={settings.window_size}"
                )
            else:
                self.status_var.set(
                    f"Plotted {len(self.profile_results)} profile(s): mode=multi, normalized x/y to [0,1], samples={settings.line_samples}, window={settings.window_size}"
                )
        except Exception as exc:
            messagebox.showerror("Plot Error", str(exc))

    def plot_current_preview_only(self) -> None:
        if not self.images:
            return
        self.draw_current_image_panel()
        if not self.profile_results:
            self.ax_profile.clear()
            self.ax_profile.set_title("Intensity Profile")
            self.ax_profile.grid(True, alpha=0.3)
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
            if overlay_mode == "multi":
                color = "red"
                points = self.multi_line_points.get(self.current_image_index)
                if points is not None:
                    draw_start, draw_end = points
                else:
                    draw_start, draw_end = self.line_start, self.line_end
            else:
                color = "red"
                draw_start, draw_end = self.line_start, self.line_end

            if draw_start is not None:
                self.ax_image.plot(draw_start[0], draw_start[1], marker="o", color=color, markersize=5)
            if draw_start is not None and draw_end is not None:
                self.ax_image.plot(
                    [draw_start[0], draw_end[0]],
                    [draw_start[1], draw_end[1]],
                    color=color,
                    linewidth=2,
                )
                self.ax_image.plot(draw_end[0], draw_end[1], marker="o", color=color, markersize=5)

        self.ax_image.set_xlim(0.0, 1.0)
        self.ax_image.set_ylim(1.0, 0.0)

        self.ax_image.set_title(f"Image")
        self.ax_image.axis("off")

    def draw_profile_panel(self) -> None:
        if not self.profile_results or self.last_plot_settings is None:
            return

        self.ax_profile.clear()

        # Use normalized x-axis when images have different profile lengths,
        # or always in multi mode for explicit cross-image alignment.
        lengths = [len(item.indices) for item in self.profile_results]
        use_normalized_x = self.last_plot_settings.profile_mode == "multi" or len(set(lengths)) != 1

        for idx, item in enumerate(self.profile_results):
            indices = np.asarray(item.indices)
            values = np.asarray(item.values)
            if use_normalized_x:
                x = np.linspace(0.0, 1.0, len(indices))
            else:
                x = indices
            color = self.get_profile_color(idx)
            self.ax_profile.plot(x, values, linewidth=1.8, color=color, label=self.format_plot_label(item.name, idx))

        self.ax_profile.set_title("Averaged Intensity Profile")
        if self.last_plot_settings.profile_mode in ("line", "multi"):
            self.ax_profile.set_xlabel("Line Position (0-1)")
        elif use_normalized_x:
            self.ax_profile.set_xlabel("Relative Position (0-1)")
        else:
            self.ax_profile.set_xlabel(
                "Height Position" if self.last_plot_settings.fixed_axis == "width" else "Width Position"
            )
        if self.last_plot_settings.profile_mode == "multi":
            self.ax_profile.set_ylabel("Normalized Intensity (0-1)")
        else:
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
                use_normalized_x = (
                    self.last_plot_settings is not None
                    and self.last_plot_settings.profile_mode == "multi"
                ) or len(set(lengths)) != 1

                for idx, item in enumerate(self.profile_results):
                    indices = np.asarray(item.indices)
                    values = np.asarray(item.values)
                    if use_normalized_x:
                        x = np.linspace(0.0, 1.0, len(indices))
                    else:
                        x = indices
                    color = self.get_profile_color(idx)
                    profile_ax.plot(x, values, linewidth=1.8, color=color, label=self.format_plot_label(item.name, idx))

                profile_ax.set_title("Averaged Intensity Profile")
                if self.last_plot_settings is not None and self.last_plot_settings.profile_mode in ("line", "multi"):
                    profile_ax.set_xlabel("Line Position (0-1)")
                elif use_normalized_x:
                    profile_ax.set_xlabel("Relative Position (0-1)")
                elif self.last_plot_settings is not None:
                    profile_ax.set_xlabel(
                        "Height Position" if self.last_plot_settings.fixed_axis == "width" else "Width Position"
                    )
                if self.last_plot_settings is not None and self.last_plot_settings.profile_mode == "multi":
                    profile_ax.set_ylabel("Normalized Intensity (0-1)")
                else:
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
