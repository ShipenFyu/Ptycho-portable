from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

from ..base_module import FeatureModule
from .analyzer import KnifeEdgeAnalyzer
from .image_loader import ImageLoader
from .models import ImageData, KnifeEdgeResult


class ResolutionModule(FeatureModule):
    name = "Resolution"

    def __init__(self, notebook: ttk.Notebook) -> None:
        super().__init__(notebook)
        self.image: Optional[ImageData] = None
        self.roi_norm: Optional[Tuple[float, float, float, float]] = None
        self.result: Optional[KnifeEdgeResult] = None
        self.analyzer = KnifeEdgeAnalyzer()

        self.pixel_size_var = tk.StringVar(value="")
        self.pixel_unit_var = tk.StringVar(value="um")
        self.oversampling_var = tk.IntVar(value=8)
        self.smoothing_var = tk.IntVar(value=7)
        self.status_var = tk.StringVar(value="Ready. Load an image and drag a ROI across one knife edge.")
        self.result_var = tk.StringVar(value="No result yet.")

        self.roi_selector: Optional[RectangleSelector] = None
        self.view_xlim: Optional[Tuple[float, float]] = None
        self.view_ylim: Optional[Tuple[float, float]] = None

    def build(self) -> None:
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main, text="Control Panel", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.build_controls(control_frame)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.fig = plt.figure(figsize=(14, 8), dpi=120)
        grid = self.fig.add_gridspec(
            4,
            2,
            width_ratios=[1.18, 1.0],
            height_ratios=[1.1, 1.0, 1.0, 1.0],
            hspace=0.65,
            wspace=0.28,
        )
        self.ax_image = self.fig.add_subplot(grid[:, 0])
        self.ax_roi = self.fig.add_subplot(grid[0, 1])
        self.ax_esf = self.fig.add_subplot(grid[1, 1])
        self.ax_lsf = self.fig.add_subplot(grid[2, 1])
        self.ax_mtf = self.fig.add_subplot(grid[3, 1])
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.08, wspace=0.28, hspace=0.78)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom)

        self.setup_roi_selector()

        status_label = ttk.Label(self.frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=tk.X, pady=(8, 0))

        self.draw_all()

    def build_controls(self, parent: ttk.LabelFrame) -> None:
        ttk.Button(parent, text="Load Image", command=self.on_load_image).pack(fill=tk.X, pady=(0, 6))
        self.image_label = ttk.Label(parent, text="Image: N/A", wraplength=260)
        self.image_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="Pixel Size (optional)").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.pixel_size_var).pack(fill=tk.X, pady=(0, 6))
        ttk.Combobox(parent, textvariable=self.pixel_unit_var, state="readonly", values=["nm", "um", "mm"]).pack(
            fill=tk.X, pady=(0, 12)
        )

        ttk.Label(parent, text="ESF Oversampling").pack(anchor="w")
        ttk.Spinbox(parent, from_=2, to=32, increment=1, textvariable=self.oversampling_var).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="ESF Smooth Window (odd)").pack(anchor="w")
        ttk.Spinbox(parent, from_=1, to=51, increment=2, textvariable=self.smoothing_var).pack(fill=tk.X, pady=(0, 12))

        ttk.Button(parent, text="Analyze ROI", command=self.on_analyze).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(parent, text="Clear ROI", command=self.on_clear_roi).pack(fill=tk.X, pady=(0, 6))
        zoom_frame = ttk.LabelFrame(parent, text="Image View", padding=6)
        zoom_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(zoom_frame, text="Zoom In", command=self.on_zoom_in).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.on_zoom_out).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Button(zoom_frame, text="Reset", command=self.on_reset_view).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Button(parent, text="Save Plot", command=self.on_save_plot).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(parent, text="Save CSV", command=self.on_save_csv).pack(fill=tk.X)

        tips = (
            "ROI selection:\n"
            "1) Drag a box crossing one clean edge\n"
            "2) Keep bright and dark regions on both sides\n"
            "3) Let the edge pass through most of the ROI\n"
            "4) Avoid texture, corners, or multiple edges\n"
            "5) A slight edge tilt improves sampling"
        )
        ttk.Label(parent, text=tips, justify=tk.LEFT, wraplength=260).pack(anchor="w", pady=(14, 10))

        result_frame = ttk.LabelFrame(parent, text="Result", padding=8)
        result_frame.pack(fill=tk.X)
        ttk.Label(result_frame, textvariable=self.result_var, justify=tk.LEFT, wraplength=260).pack(anchor="w")

    def setup_roi_selector(self) -> None:
        self.roi_selector = RectangleSelector(
            self.ax_image,
            self.on_roi_selected,
            useblit=True,
            button=[MouseButton.LEFT],
            interactive=False,
            drag_from_anywhere=False,
        )
        self.update_roi_selector_state()

    def update_roi_selector_state(self) -> None:
        if self.roi_selector is not None:
            self.roi_selector.set_active(self.image is not None)

    def on_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.image = ImageLoader.load_image(Path(path))
            self.roi_norm = None
            self.result = None
            self.view_xlim = None
            self.view_ylim = None
            self.image_label.config(text=f"Image: {self.format_display_name(self.image.path.name)}")
            self.result_var.set("No result yet.")
            self.status_var.set("Image loaded. Drag a ROI around one clean knife edge.")
            self.update_roi_selector_state()
            self.draw_all()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def on_roi_selected(self, eclick, erelease) -> None:
        if self.image is None:
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return

        x0 = float(np.clip(eclick.xdata, 0.0, 1.0))
        y0 = float(np.clip(eclick.ydata, 0.0, 1.0))
        x1 = float(np.clip(erelease.xdata, 0.0, 1.0))
        y1 = float(np.clip(erelease.ydata, 0.0, 1.0))
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if (right - left) <= 1e-4 or (bottom - top) <= 1e-4:
            return

        self.roi_norm = (left, top, right, bottom)
        self.result = None
        self.result_var.set("ROI selected. Click Analyze ROI.")
        self.status_var.set("ROI selected. Check that it contains one clean edge, then analyze.")
        self.draw_all()

    def on_clear_roi(self) -> None:
        self.roi_norm = None
        self.result = None
        self.result_var.set("No result yet.")
        self.status_var.set("ROI cleared.")
        self.draw_all()

    def on_zoom_in(self) -> None:
        self.zoom_image_view(0.5)

    def on_zoom_out(self) -> None:
        self.zoom_image_view(2.0)

    def on_reset_view(self) -> None:
        self.view_xlim = None
        self.view_ylim = None
        self.status_var.set("Image view reset.")
        self.draw_all()

    def on_scroll_zoom(self, event) -> None:
        if self.image is None or event.inaxes != self.ax_image:
            return
        if event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        self.zoom_image_view(scale, center=(float(event.xdata), float(event.ydata)))

    def zoom_image_view(self, scale: float, center: Optional[Tuple[float, float]] = None) -> None:
        if self.image is None:
            return

        if self.view_xlim is None or self.view_ylim is None:
            xlim = (0.0, 1.0)
            ylim = (1.0, 0.0)
        else:
            xlim = self.view_xlim
            ylim = self.view_ylim

        if center is None:
            cx = (xlim[0] + xlim[1]) * 0.5
            cy = (ylim[0] + ylim[1]) * 0.5
        else:
            cx = float(np.clip(center[0], 0.0, 1.0))
            cy = float(np.clip(center[1], 0.0, 1.0))
        width = abs(xlim[1] - xlim[0]) * scale
        height = abs(ylim[1] - ylim[0]) * scale
        width = float(np.clip(width, 0.02, 1.0))
        height = float(np.clip(height, 0.02, 1.0))

        x0 = float(np.clip(cx - width * 0.5, 0.0, 1.0 - width))
        x1 = x0 + width
        y0 = float(np.clip(cy - height * 0.5, 0.0, 1.0 - height))
        y1 = y0 + height
        self.view_xlim = (x0, x1)
        self.view_ylim = (y1, y0)
        self.status_var.set("Image view zoomed. Drag ROI within the enlarged view.")
        self.draw_all()

    def on_analyze(self) -> None:
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        if self.roi_norm is None:
            messagebox.showwarning("No ROI", "Please drag a ROI around one knife edge first.")
            return

        try:
            roi_bounds = self.normalized_roi_to_pixels(self.image.shape)
            result = self.analyzer.analyze(
                self.image.intensity,
                roi_bounds,
                oversampling=int(self.oversampling_var.get()),
                smoothing_window=int(self.smoothing_var.get()),
            )
            self.result = result
            self.result_var.set(self.format_result_text(result))
            self.status_var.set("Knife-edge analysis complete. Confirm the detected edge line in the previews.")
            self.draw_all()
        except Exception as exc:
            messagebox.showerror("Analysis Error", str(exc))

    def normalized_roi_to_pixels(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        if self.roi_norm is None:
            raise ValueError("ROI is not selected.")
        h, w = shape
        left, top, right, bottom = self.roi_norm
        x0 = int(np.floor(left * (w - 1)))
        y0 = int(np.floor(top * (h - 1)))
        x1 = int(np.ceil(right * (w - 1)))
        y1 = int(np.ceil(bottom * (h - 1)))
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 <= x0:
            x1 = min(x0 + 1, w - 1)
        if y1 <= y0:
            y1 = min(y0 + 1, h - 1)
        return x0, y0, x1, y1

    def format_result_text(self, result: KnifeEdgeResult) -> str:
        lines = [
            f"Edge angle: {result.edge_line.angle_deg:.2f} deg",
            f"MTF50: {self.format_frequency(result.mtf50)}",
            f"MTF10: {self.format_frequency(result.mtf10)}",
            f"10-90 width: {self.format_distance(result.edge_width_10_90)}",
        ]
        pixel_size = self.read_pixel_size()
        if pixel_size is not None:
            lines.append(f"Pixel size: {pixel_size:g} {self.pixel_unit_var.get()}/px")
        return "\n".join(lines)

    def read_pixel_size(self) -> Optional[float]:
        text = self.pixel_size_var.get().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value

    def pixel_size_mm(self) -> Optional[float]:
        value = self.read_pixel_size()
        if value is None:
            return None
        unit = self.pixel_unit_var.get()
        if unit == "nm":
            return value * 1e-6
        if unit == "um":
            return value * 1e-3
        if unit == "mm":
            return value
        return None

    def format_frequency(self, cycles_per_pixel: Optional[float]) -> str:
        if cycles_per_pixel is None:
            return "not reached"
        pixel_mm = self.pixel_size_mm()
        if pixel_mm is None:
            return f"{cycles_per_pixel:.4f} cycles/pixel"
        return f"{cycles_per_pixel / pixel_mm:.2f} lp/mm ({cycles_per_pixel:.4f} cycles/pixel)"

    def format_distance(self, pixels: Optional[float]) -> str:
        if pixels is None:
            return "N/A"
        pixel_size = self.read_pixel_size()
        if pixel_size is None:
            return f"{pixels:.3f} px"
        return f"{pixels * pixel_size:.3f} {self.pixel_unit_var.get()} ({pixels:.3f} px)"

    def draw_all(self) -> None:
        self.draw_image_panel()
        self.draw_roi_panel()
        self.draw_curve_panels()
        self.canvas.draw_idle()

    def draw_image_panel(self) -> None:
        self.ax_image.clear()
        self.ax_image.set_title("Image + ROI + Edge Line")
        if self.image is None:
            self.ax_image.text(0.5, 0.5, "No image loaded", ha="center", va="center", transform=self.ax_image.transAxes)
            self.ax_image.axis("off")
            return

        extent = (0.0, 1.0, 1.0, 0.0)
        if self.image.is_rgb:
            self.ax_image.imshow(self.image.display, extent=extent)
        else:
            self.ax_image.imshow(self.image.display, cmap="gray", extent=extent)
        self.ax_image.set_xlim(0.0, 1.0)
        self.ax_image.set_ylim(1.0, 0.0)
        if self.view_xlim is not None and self.view_ylim is not None:
            self.ax_image.set_xlim(*self.view_xlim)
            self.ax_image.set_ylim(*self.view_ylim)
        self.ax_image.axis("off")
        self.draw_roi_overlay(self.ax_image)
        self.draw_edge_line_on_image(self.ax_image)

    def draw_roi_panel(self) -> None:
        self.ax_roi.clear()
        self.ax_roi.set_title("ROI Preview")
        if self.image is None or self.roi_norm is None:
            self.ax_roi.text(0.5, 0.5, "Select ROI", ha="center", va="center", transform=self.ax_roi.transAxes)
            self.ax_roi.axis("off")
            return

        x0, y0, x1, y1 = self.normalized_roi_to_pixels(self.image.shape)
        roi = self.image.display[y0 : y1 + 1, x0 : x1 + 1]
        if self.image.is_rgb:
            self.ax_roi.imshow(roi)
        else:
            self.ax_roi.imshow(roi, cmap="gray")
        self.draw_edge_line_on_roi(self.ax_roi)
        self.ax_roi.axis("off")

    def draw_curve_panels(self) -> None:
        for ax, title in ((self.ax_esf, "ESF"), (self.ax_lsf, "LSF"), (self.ax_mtf, "MTF")):
            ax.clear()
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        if self.result is None:
            self.ax_esf.text(0.5, 0.5, "Analyze ROI", ha="center", va="center", transform=self.ax_esf.transAxes)
            self.ax_lsf.text(0.5, 0.5, "No LSF", ha="center", va="center", transform=self.ax_lsf.transAxes)
            self.ax_mtf.text(0.5, 0.5, "No MTF", ha="center", va="center", transform=self.ax_mtf.transAxes)
            return

        result = self.result
        self.ax_esf.plot(result.distances, result.esf, color="#8a8a8a", linewidth=1.0, label="raw")
        self.ax_esf.plot(result.distances, result.esf_smooth, color="#1f77b4", linewidth=1.5, label="smooth")
        self.ax_esf.set_xlabel("Distance (px)")
        self.ax_esf.set_ylabel("Intensity")
        self.ax_esf.legend(fontsize=7)

        self.ax_lsf.plot(result.distances, result.lsf, color="#2ca02c", linewidth=1.4)
        self.ax_lsf.set_xlabel("Distance (px)")
        self.ax_lsf.set_ylabel("dI/dx")

        self.ax_mtf.plot(result.frequencies, result.mtf, color="#d62728", linewidth=1.4)
        self.ax_mtf.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
        self.ax_mtf.axhline(0.1, color="#999999", linestyle=":", linewidth=0.8)
        if result.mtf50 is not None:
            self.ax_mtf.axvline(result.mtf50, color="#1f77b4", linestyle="--", linewidth=0.9)
        if result.mtf10 is not None:
            self.ax_mtf.axvline(result.mtf10, color="#9467bd", linestyle=":", linewidth=0.9)
        self.ax_mtf.set_xlim(0.0, 0.5)
        self.ax_mtf.set_ylim(0.0, max(1.05, float(np.max(result.mtf)) * 1.05))
        self.ax_mtf.set_xlabel("Spatial Frequency (cycles/pixel)")
        self.ax_mtf.set_ylabel("MTF")

    def draw_roi_overlay(self, ax) -> None:
        if self.roi_norm is None:
            return
        left, top, right, bottom = self.roi_norm
        rect = Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

    def draw_edge_line_on_image(self, ax) -> None:
        if self.image is None or self.result is None:
            return
        x0, y0, x1, y1 = self.result.roi_bounds
        start, end = self.edge_line_segment(width=x1 - x0 + 1, height=y1 - y0 + 1)
        h, w = self.image.shape
        ax.plot(
            [(x0 + start[0]) / max(w - 1, 1), (x0 + end[0]) / max(w - 1, 1)],
            [(y0 + start[1]) / max(h - 1, 1), (y0 + end[1]) / max(h - 1, 1)],
            color="cyan",
            linewidth=2,
        )

    def draw_edge_line_on_roi(self, ax) -> None:
        if self.result is None:
            return
        x0, y0, x1, y1 = self.result.roi_bounds
        start, end = self.edge_line_segment(width=x1 - x0 + 1, height=y1 - y0 + 1)
        ax.plot([start[0], end[0]], [start[1], end[1]], color="cyan", linewidth=2)

    def edge_line_segment(self, width: int, height: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if self.result is None:
            return (0.0, 0.0), (0.0, 0.0)
        line = self.result.edge_line
        candidates: list[Tuple[float, float]] = []
        max_x = float(width - 1)
        max_y = float(height - 1)
        eps = 1e-12

        if abs(line.tangent_x) > eps:
            for x in (0.0, max_x):
                t = (x - line.center_x) / line.tangent_x
                y = line.center_y + t * line.tangent_y
                if 0.0 <= y <= max_y:
                    candidates.append((x, y))

        if abs(line.tangent_y) > eps:
            for y in (0.0, max_y):
                t = (y - line.center_y) / line.tangent_y
                x = line.center_x + t * line.tangent_x
                if 0.0 <= x <= max_x:
                    candidates.append((x, y))

        unique: list[Tuple[float, float]] = []
        for point in candidates:
            if not any(np.hypot(point[0] - old[0], point[1] - old[1]) < 1e-6 for old in unique):
                unique.append(point)

        if len(unique) >= 2:
            return unique[0], unique[1]

        return (
            (float(np.clip(line.center_x, 0.0, max_x)), float(np.clip(line.center_y, 0.0, max_y))),
            (float(np.clip(line.center_x, 0.0, max_x)), float(np.clip(line.center_y, 0.0, max_y))),
        )

    def on_save_plot(self) -> None:
        if self.result is None:
            messagebox.showwarning("Nothing to Save", "Please analyze a ROI first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Knife-Edge Plot",
            defaultextension=".png",
            initialfile=f"resolution_knife-edge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
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

    def on_save_csv(self) -> None:
        if self.result is None:
            messagebox.showwarning("Nothing to Save", "Please analyze a ROI first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Knife-Edge CSV",
            defaultextension=".csv",
            initialfile=f"resolution_knife-edge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            self.write_csv(Path(path))
            self.status_var.set(f"Saved CSV to: {path}")
            messagebox.showinfo("Saved", f"CSV saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def write_csv(self, path: Path) -> None:
        if self.result is None:
            return
        result = self.result
        max_len = max(len(result.distances), len(result.frequencies))
        with path.open("w", encoding="utf-8") as f:
            f.write("distance_px,esf,esf_smooth,lsf,frequency_cycles_per_pixel,mtf\n")
            for i in range(max_len):
                row = [
                    self.csv_value(result.distances, i),
                    self.csv_value(result.esf, i),
                    self.csv_value(result.esf_smooth, i),
                    self.csv_value(result.lsf, i),
                    self.csv_value(result.frequencies, i),
                    self.csv_value(result.mtf, i),
                ]
                f.write(",".join(row) + "\n")

    @staticmethod
    def csv_value(values: np.ndarray, index: int) -> str:
        if index >= len(values):
            return ""
        return f"{float(values[index]):.10g}"

    def format_display_name(self, filename: str) -> str:
        path = Path(filename)
        stem = path.stem[:10]
        return f"{stem}{path.suffix}"

    def on_close(self) -> None:
        if hasattr(self, "fig"):
            plt.close(self.fig)
