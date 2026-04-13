from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import Normalize

from .base_module import FeatureModule


@dataclass
class TensorData:
    name: str
    values: np.ndarray
    display: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape


class TensorAdapter:
    @staticmethod
    def as_2d_float(array: np.ndarray, tensor_name: str) -> np.ndarray:
        arr = np.asarray(array)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Tensor '{tensor_name}' is not 2D after squeeze. Got shape={arr.shape}")
        if np.iscomplexobj(arr):
            raise ValueError(f"Tensor '{tensor_name}' must be real-valued, but got complex dtype={arr.dtype}")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def amplitude_values(array: np.ndarray, tensor_name: str) -> np.ndarray:
        return TensorAdapter.as_2d_float(array, tensor_name)

    @staticmethod
    def phase_values(array: np.ndarray, tensor_name: str) -> np.ndarray:
        return TensorAdapter.as_2d_float(array, tensor_name)
    

class CenteredPowerNorm(Normalize):
    def __init__(self, gamma=0.5, vcenter=0, vmin=-np.pi, vmax=np.pi):
        self.gamma = gamma
        self.vcenter = vcenter
        super().__init__(vmin, vmax)

    def __call__(self, value, clip=None):
        value = np.asarray(value)
        if self.vmin is None or self.vmax is None:
            raise ValueError("Vmin and Vmax must be set before normalization")

        if clip is None:
            clip = self.clip

        vmin = float(self.vmin)
        vmax = float(self.vmax)
        vcenter = float(self.vcenter)

        if vmin >= vmax:
            raise ValueError("vmin must be less than vmax")
        if not (vmin < vcenter < vmax):
            raise ValueError("vcenter must be between vmin and vmax")

        if clip:
            value = np.clip(value, vmin, vmax)

        # Symmetric power mapping around vcenter:
        # vcenter -> 0.5, vmin -> 0, vmax -> 1
        neg_span = vcenter - vmin
        pos_span = vmax - vcenter
        scale = np.where(value < vcenter, neg_span, pos_span)
        scale = np.where(scale == 0, 1.0, scale)

        signed = (value - vcenter) / scale
        signed = np.clip(signed, -1.0, 1.0)
        mag = np.abs(signed) ** self.gamma
        res = 0.5 + 0.5 * np.sign(signed) * mag

        return np.ma.masked_array(res)


class HsvFusionModule(FeatureModule):
    name = "HSV Fusion"

    def __init__(self, notebook: ttk.Notebook) -> None:
        super().__init__(notebook)
        self.source_npz: Optional[Path] = None
        self.amp_tensor: Optional[TensorData] = None
        self.phase_tensor: Optional[TensorData] = None

        self.saturation_var = tk.DoubleVar(value=1.0)
        self.square_roi_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready. Load an NPZ file and select amp/phase tensors.")

        self.current_input_key: str = "amp"
        self.roi_norm: Optional[Tuple[float, float, float, float]] = None
        self.output_rgb: Optional[np.ndarray] = None

        self.input_selector: Optional[RectangleSelector] = None

    def build(self) -> None:
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main, text="Control Panel", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.build_controls(control_frame)

        display_frame = ttk.Frame(main)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        input_group = ttk.LabelFrame(display_frame, text="Input", padding=6)
        input_group.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        output_group = ttk.LabelFrame(display_frame, text="Output", padding=6)
        output_group.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.input_fig, self.ax_input = plt.subplots(1, 1, figsize=(8, 5), dpi=120)
        self.input_canvas = FigureCanvasTkAgg(self.input_fig, master=input_group)
        self.input_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.output_fig, self.ax_output = plt.subplots(1, 1, figsize=(5, 5), dpi=120)
        self.output_canvas = FigureCanvasTkAgg(self.output_fig, master=output_group)
        self.output_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_roi_selector()

        status_label = ttk.Label(self.frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=tk.X, pady=(8, 0))

        self.draw_all()

    def build_controls(self, parent: ttk.LabelFrame) -> None:
        ttk.Button(parent, text="Load NPZ + Select Tensors", command=self.on_load_npz).pack(fill=tk.X, pady=(0, 6))
        self.npz_label = ttk.Label(parent, text="NPZ: N/A", wraplength=260)
        self.npz_label.pack(anchor="w", pady=(0, 10))

        self.amp_label = ttk.Label(parent, text="Amp Tensor: N/A", wraplength=260)
        self.amp_label.pack(anchor="w", pady=(0, 6))
        self.phase_label = ttk.Label(parent, text="Phase Tensor: N/A", wraplength=260)
        self.phase_label.pack(anchor="w", pady=(0, 10))

        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(nav_frame, text="Prev Input", command=self.on_prev_input).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next Input", command=self.on_next_input).pack(side=tk.RIGHT)
        self.input_nav_label = ttk.Label(parent, text="Current input: N/A")
        self.input_nav_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="Saturation [0, 1]").pack(anchor="w")
        self.saturation_entry = ttk.Entry(parent, textvariable=self.saturation_var)
        self.saturation_entry.pack(fill=tk.X, pady=(0, 10))
        self.saturation_entry.bind("<Return>", self.on_saturation_changed)
        self.saturation_entry.bind("<FocusOut>", self.on_saturation_changed)

        ttk.Checkbutton(parent, text="Square ROI (unchecked = Rectangle)", variable=self.square_roi_var).pack(
            anchor="w", pady=(0, 10)
        )

        ttk.Button(parent, text="Clear ROI", command=self.on_clear_roi).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(parent, text="Save HSV Result", command=self.on_save_output).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(parent, text="Clear All", command=self.on_clear_all).pack(fill=tk.X)

        tip = (
            "ROI usage:\n"
            "1) Load one NPZ file\n"
            "2) Select amp/phase tensors in popup\n"
            "3) Use Prev/Next to switch input view\n"
            "4) Drag mouse to select ROI\n"
            "5) Output refreshes automatically"
        )
        ttk.Label(parent, text=tip, justify=tk.LEFT).pack(anchor="w", pady=(12, 0))
        self.update_input_nav_label()

    def setup_roi_selector(self) -> None:
        self.input_selector = RectangleSelector(
            self.ax_input,
            self.on_roi_selected,
            useblit=True,
            button=[MouseButton.LEFT],
            interactive=False,
            drag_from_anywhere=False,
        )
        self.update_roi_selector_state()

    def has_complete_inputs(self) -> bool:
        return self.amp_tensor is not None and self.phase_tensor is not None

    def available_input_keys(self) -> list[str]:
        keys = []
        if self.amp_tensor is not None:
            keys.append("amp")
        if self.phase_tensor is not None:
            keys.append("phase")
        return keys

    def format_display_name(self, filename: str) -> str:
        path = Path(filename)
        stem = path.stem[:6]
        return f"{stem}{path.suffix}"

    def update_input_nav_label(self) -> None:
        available = self.available_input_keys()
        if not available:
            self.input_nav_label.config(text="Current input: N/A")
            return

        if self.current_input_key not in available:
            self.current_input_key = available[0]

        if self.current_input_key == "amp" and self.amp_tensor is not None:
            text = f"Current input: Amplitude\n{self.format_display_name(self.amp_tensor.name)}"
        elif self.current_input_key == "phase" and self.phase_tensor is not None:
            text = f"Current input: Phase\n{self.format_display_name(self.phase_tensor.name)}"
        else:
            text = "Current input: N/A"
        self.input_nav_label.config(text=text)

    def update_roi_selector_state(self) -> None:
        if self.input_selector is not None:
            self.input_selector.set_active(self.has_complete_inputs())

    def on_prev_input(self) -> None:
        available = self.available_input_keys()
        if not available:
            return
        if self.current_input_key not in available:
            self.current_input_key = available[0]
        else:
            idx = available.index(self.current_input_key)
            self.current_input_key = available[(idx - 1) % len(available)]
        self.update_input_nav_label()
        self.draw_input_panel()

    def on_next_input(self) -> None:
        available = self.available_input_keys()
        if not available:
            return
        if self.current_input_key not in available:
            self.current_input_key = available[0]
        else:
            idx = available.index(self.current_input_key)
            self.current_input_key = available[(idx + 1) % len(available)]
        self.update_input_nav_label()
        self.draw_input_panel()

    def select_single_file(self, title: str) -> Optional[Path]:
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("NumPy Archive", "*.npz"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return None
        return Path(file_path)

    def select_tensor_keys_dialog(self, keys: list[str]) -> Optional[Tuple[str, str]]:
        if not keys:
            return None

        dialog = tk.Toplevel(self.frame)
        dialog.title("Select Amp/Phase Tensors")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text="Amplitude tensor key").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        amp_var = tk.StringVar(value=keys[0])
        amp_combo = ttk.Combobox(dialog, textvariable=amp_var, values=keys, state="readonly", width=36)
        amp_combo.grid(row=1, column=0, padx=12, pady=(0, 10))

        ttk.Label(dialog, text="Phase tensor key").grid(row=2, column=0, sticky="w", padx=12, pady=(0, 4))
        phase_default = keys[1] if len(keys) > 1 else keys[0]
        phase_var = tk.StringVar(value=phase_default)
        phase_combo = ttk.Combobox(dialog, textvariable=phase_var, values=keys, state="readonly", width=36)
        phase_combo.grid(row=3, column=0, padx=12, pady=(0, 12))

        result: dict[str, Optional[Tuple[str, str]]] = {"value": None}

        def on_ok() -> None:
            result["value"] = (amp_var.get(), phase_var.get())
            dialog.destroy()

        def on_cancel() -> None:
            result["value"] = None
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=4, column=0, sticky="e", padx=12, pady=(0, 12))
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        amp_combo.focus_set()
        dialog.wait_window()

        return result["value"]

    def on_load_npz(self) -> None:
        path = self.select_single_file("Select NPZ File")
        if path is None:
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = sorted(data.files)
                if not keys:
                    raise ValueError("This NPZ file has no tensors.")
                selected = self.select_tensor_keys_dialog(keys)
                if selected is None:
                    return

                amp_key, phase_key = selected
                if amp_key not in data or phase_key not in data:
                    raise ValueError("Selected key not found in NPZ file.")

                amp_values = TensorAdapter.amplitude_values(data[amp_key], amp_key)
                phase_values = TensorAdapter.phase_values(data[phase_key], phase_key)

            self.source_npz = path
            self.amp_tensor = TensorData(name=amp_key, values=amp_values, display=amp_values)
            self.phase_tensor = TensorData(name=phase_key, values=phase_values, display=phase_values)
            self.ensure_shapes_match()
            self.current_input_key = "amp"
            self.on_clear_roi(update_status=False)
            self.update_roi_selector_state()
            self.update_input_nav_label()
            self.npz_label.config(text=f"NPZ: {self.format_display_name(path.name)}")
            self.amp_label.config(text=f"Amp Tensor: {self.format_display_name(amp_key)}")
            self.phase_label.config(text=f"Phase Tensor: {self.format_display_name(phase_key)}")

            if self.has_complete_inputs():
                self.status_var.set("NPZ loaded. You can drag ROI now.")
            else:
                self.status_var.set("NPZ loaded but tensor selection is incomplete.")
            self.draw_all()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def ensure_shapes_match(self) -> None:
        if self.amp_tensor is None or self.phase_tensor is None:
            return
        if self.amp_tensor.shape != self.phase_tensor.shape:
            amp_shape = self.amp_tensor.shape
            phase_shape = self.phase_tensor.shape
            self.phase_tensor = None
            self.phase_label.config(text="Phase Tensor: N/A")
            self.update_roi_selector_state()
            self.update_input_nav_label()
            raise ValueError(
                f"Amp and phase tensors must have the same shape. "
                f"Got amp={amp_shape}, phase={phase_shape}."
            )

    def on_roi_selected(self, eclick, erelease) -> None:
        if not self.has_complete_inputs():
            self.status_var.set("Please load and select both amp and phase tensors first.")
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return

        x0 = float(np.clip(eclick.xdata, 0.0, 1.0))
        y0 = float(np.clip(eclick.ydata, 0.0, 1.0))
        x1 = float(np.clip(erelease.xdata, 0.0, 1.0))
        y1 = float(np.clip(erelease.ydata, 0.0, 1.0))

        left = min(x0, x1)
        right = max(x0, x1)
        top = min(y0, y1)
        bottom = max(y0, y1)

        if self.square_roi_var.get():
            dx = right - left
            dy = bottom - top
            side = min(dx, dy)
            if side <= 1e-6:
                return
            sx = 1.0 if x1 >= x0 else -1.0
            sy = 1.0 if y1 >= y0 else -1.0
            nx1 = np.clip(x0 + sx * side, 0.0, 1.0)
            ny1 = np.clip(y0 + sy * side, 0.0, 1.0)
            left = min(x0, nx1)
            right = max(x0, nx1)
            top = min(y0, ny1)
            bottom = max(y0, ny1)

        if (right - left) <= 1e-6 or (bottom - top) <= 1e-6:
            return

        self.roi_norm = (left, top, right, bottom)
        self.draw_input_panel()
        self.run_fusion_from_roi(show_dialog=True)

    def on_saturation_changed(self, _event=None) -> None:
        if self.roi_norm is None or not self.has_complete_inputs():
            return
        self.run_fusion_from_roi(show_dialog=False)

    def on_clear_roi(self, update_status: bool = True) -> None:
        self.roi_norm = None
        self.output_rgb = None
        if update_status:
            self.status_var.set("ROI cleared.")
        self.draw_all()

    def on_clear_all(self) -> None:
        self.source_npz = None
        self.amp_tensor = None
        self.phase_tensor = None
        self.current_input_key = "amp"
        self.roi_norm = None
        self.output_rgb = None
        self.npz_label.config(text="NPZ: N/A")
        self.amp_label.config(text="Amp Tensor: N/A")
        self.phase_label.config(text="Phase Tensor: N/A")
        self.update_roi_selector_state()
        self.update_input_nav_label()
        self.status_var.set("Cleared all inputs and outputs.")
        self.draw_all()

    def normalized_roi_to_pixels(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        if self.roi_norm is None:
            return 0, 0, shape[1] - 1, shape[0] - 1

        h, w = shape
        left, top, right, bottom = self.roi_norm

        x0 = int(np.floor(left * (w - 1)))
        y0 = int(np.floor(top * (h - 1)))
        x1 = int(np.ceil(right * (w - 1)))
        y1 = int(np.ceil(bottom * (h - 1)))

        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))

        if x1 <= x0:
            x1 = min(x0 + 1, w - 1)
        if y1 <= y0:
            y1 = min(y0 + 1, h - 1)

        return x0, y0, x1, y1

    def phase_to_hue(self, phase: np.ndarray) -> np.ndarray:
        phase_min = float(np.min(phase))
        phase_max = float(np.max(phase))

        if phase_min >= -np.pi and phase_max <= np.pi:
            hue = (phase + np.pi) / (2 * np.pi)
        else:
            span = phase_max - phase_min
            if span > 1e-12:
                hue = (phase - phase_min) / span
            else:
                hue = np.zeros_like(phase, dtype=np.float32)

        return np.clip(hue, 0.0, 1.0).astype(np.float32)

    def run_fusion_from_roi(self, show_dialog: bool) -> None:
        if self.amp_tensor is None or self.phase_tensor is None or self.roi_norm is None:
            return

        try:
            saturation = float(self.saturation_var.get())
        except Exception:
            if show_dialog:
                messagebox.showerror("Invalid Saturation", "Saturation must be a numeric value.")
            else:
                self.status_var.set("Invalid saturation value.")
            return

        if not (0.0 <= saturation <= 1.0):
            if show_dialog:
                messagebox.showerror("Invalid Saturation", "Saturation must be within [0, 1].")
            else:
                self.status_var.set("Saturation must be within [0, 1].")
            return

        x0, y0, x1, y1 = self.normalized_roi_to_pixels(self.amp_tensor.shape)

        amp_roi = self.amp_tensor.values[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        phase_roi = self.phase_tensor.values[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)

        hsv_image = np.zeros((amp_roi.shape[0], amp_roi.shape[1], 3), dtype=np.float32)
        hsv_image[..., 0] = self.phase_to_hue(phase_roi)
        hsv_image[..., 1] = saturation

        amp_max = float(np.max(amp_roi))
        hsv_image[..., 2] = amp_roi / amp_max if amp_max > 0 else amp_roi

        self.output_rgb = hsv_to_rgb(hsv_image)
        self.draw_output_panel()
        self.status_var.set(
            f"HSV fusion done. ROI pixels: x=[{x0}, {x1}], y=[{y0}, {y1}], saturation={saturation:.3f}"
        )

    def draw_roi_overlay(self, ax) -> None:
        if self.roi_norm is None:
            return
        left, top, right, bottom = self.roi_norm
        rect = Rectangle(
            (left, top),
            right - left,
            bottom - top,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

    def draw_input_panel(self) -> None:
        self.ax_input.clear()

        extent = (0.0, 1.0, 1.0, 0.0)
        if self.current_input_key == "phase":
            tensor = self.phase_tensor
            title = "Phase"
            cmap = "hsv"
            norm = CenteredPowerNorm(gamma=0.6, vcenter=0, vmin=-np.pi, vmax=np.pi)
        else:
            tensor = self.amp_tensor
            title = "Amplitude"
            cmap = "gray"
            norm = None

        if tensor is None:
            self.ax_input.set_title(title)
            self.ax_input.text(0.5, 0.5, "No image", ha="center", va="center", transform=self.ax_input.transAxes)
        else:
            self.ax_input.imshow(tensor.display, cmap=cmap, extent=extent, norm=norm)
            self.ax_input.set_title(title)
            self.draw_roi_overlay(self.ax_input)

        self.ax_input.set_xlim(0.0, 1.0)
        self.ax_input.set_ylim(1.0, 0.0)
        self.ax_input.axis("off")

        self.input_fig.tight_layout()
        self.input_canvas.draw_idle()

    def draw_output_panel(self) -> None:
        self.ax_output.clear()
        self.ax_output.set_title("HSV Fusion Result")
        if self.output_rgb is None:
            self.ax_output.text(0.5, 0.5, "No output yet", ha="center", va="center", transform=self.ax_output.transAxes)
        else:
            self.ax_output.imshow(self.output_rgb)
        self.ax_output.axis("off")
        self.output_fig.tight_layout()
        self.output_canvas.draw_idle()

    def draw_all(self) -> None:
        self.draw_input_panel()
        self.draw_output_panel()

    def on_save_output(self) -> None:
        if self.output_rgb is None:
            messagebox.showwarning("Nothing to Save", "Please select ROI first to generate HSV output.")
            return

        path = filedialog.asksaveasfilename(
            title="Save HSV Fusion Result",
            defaultextension=".png",
            initialfile=f"hsv-fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All Files", "*.*")],
        )

        if not path:
            return

        try:
            out = np.clip(self.output_rgb, 0.0, 1.0)
            plt.imsave(path, out)
            self.status_var.set(f"Saved HSV result to: {path}")
            messagebox.showinfo("Saved", f"HSV result saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def on_close(self) -> None:
        if hasattr(self, "input_fig"):
            plt.close(self.input_fig)
        if hasattr(self, "output_fig"):
            plt.close(self.output_fig)