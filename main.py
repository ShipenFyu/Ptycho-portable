from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from modules import IntensityProfileModule
from modules.base_module import FeatureModule


class PtychoImagingGUI:
    """Main GUI shell that hosts pluggable feature modules."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Ptychography Toolkit")
        self.root.geometry("1280x820")
        self.maximize_window()
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)

        self.modules: list[FeatureModule] = []
        self.build_ui()
        self.register_default_modules()

    def build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            container,
            text="Ptychography Toolkit",
            font=("Arial", 14, "bold"),
        )
        header.pack(anchor="w", pady=(0, 8))

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def register_default_modules(self) -> None:
        # First feature module: intensity profile plotting.
        self.register_module(IntensityProfileModule(self.notebook))

    def register_module(self, module: FeatureModule) -> None:
        module.build()
        self.notebook.add(module.frame, text=module.name)
        self.modules.append(module)

    def on_tab_changed(self, _: tk.Event) -> None:
        index = self.notebook.index(self.notebook.select())
        if 0 <= index < len(self.modules):
            self.modules[index].on_show()

    def maximize_window(self) -> None:
        # On Windows this gives native maximized behavior.
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.attributes("-zoomed", True)

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
