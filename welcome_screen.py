from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

from modules.base_module import FeatureModule


class WelcomeScreen:
    """Standalone welcome view used as the GUI entry screen."""

    def __init__(self, parent: ttk.Frame, on_open_module: Callable[[int], None]) -> None:
        self.parent = parent
        self.on_open_module = on_open_module

        self.frame = ttk.Frame(parent, padding=20)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.build_ui()

    def build_ui(self) -> None:
        title = ttk.Label(
            self.frame,
            text="Welcome to Ptychography Toolkit",
            font=("Arial", 18, "bold"),
        )
        title.pack(anchor="w", pady=(0, 8))

        desc = ttk.Label(
            self.frame,
            text="Choose a function below to enter the corresponding workspace.",
            font=("Arial", 11),
        )
        desc.pack(anchor="w", pady=(0, 14))

        self.quick_access_frame = ttk.LabelFrame(self.frame, text="Function Entry", padding=12)
        self.quick_access_frame.pack(fill=tk.X, anchor="w")

    def update_module_buttons(self, modules: list[FeatureModule], default_module_index: int = 0) -> None:
        for child in self.quick_access_frame.winfo_children():
            child.destroy()

        if not modules:
            ttk.Label(self.quick_access_frame, text="No function module registered yet.").pack(anchor="w")
            return

        for index, module in enumerate(modules):
            label = f"Open {module.name}"
            if index == default_module_index:
                label = f"{label} (Default)"

            ttk.Button(
                self.quick_access_frame,
                text=label,
                command=lambda i=index: self.on_open_module(i),
            ).pack(anchor="w", fill=tk.X, pady=4)

    def hide(self) -> None:
        self.frame.pack_forget()

    def show(self) -> None:
        self.frame.pack(fill=tk.BOTH, expand=True)
