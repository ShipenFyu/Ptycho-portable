from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from modules import HsvFusionModule, IntensityProfileModule
from modules.base_module import FeatureModule
from welcome_screen import WelcomeScreen


class PtychoImagingGUI:
    """Main GUI shell that hosts pluggable feature modules."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Ptychography Toolkit")
        self.root.geometry("1280x820")
        self.maximize_window()
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)

        self.module_classes = [IntensityProfileModule, HsvFusionModule]
        self.modules: list[FeatureModule] = []
        self.default_module_index = 0
        self.entered_workspace = False
        self.build_ui()
        self.register_modules()
        self.welcome_screen.update_module_buttons(self.modules, self.default_module_index)

    def build_ui(self) -> None:
        self.container = ttk.Frame(self.root, padding=10)
        self.container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            self.container,
            text="Ptychography Toolkit",
            font=("Arial", 14, "bold"),
        )
        header.pack(anchor="w", pady=(0, 8))

        self.content_frame = ttk.Frame(self.container)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(self.content_frame)
        self.welcome_screen = WelcomeScreen(self.content_frame, self.open_module_tab)

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def open_module_tab(self, module_index: int) -> None:
        if not self.entered_workspace:
            self.enter_workspace()
        self.notebook.select(module_index)

    def enter_workspace(self) -> None:
        if self.entered_workspace:
            return

        self.welcome_screen.hide()
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.entered_workspace = True

    def register_modules(self) -> None:
        for module_class in self.module_classes:
            module = module_class(self.notebook)
            module.build()
            self.notebook.add(module.frame, text=module.name)
            self.modules.append(module)

    def maximize_window(self) -> None:
        # On Windows this gives native maximized behavior.
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.attributes("-zoomed", True)

    def on_tab_changed(self, _: tk.Event) -> None:
        if not self.entered_workspace:
            return

        index = self.notebook.index(self.notebook.select())
        if 0 <= index < len(self.modules):
            self.modules[index].on_show()
    
    def on_app_close(self) -> None:
        for module in self.modules:
            try:
                module.on_close()
            except Exception:
                pass

        self.root.quit()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = PtychoImagingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
