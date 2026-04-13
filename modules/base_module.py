from __future__ import annotations

from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import ttk


class FeatureModule(ABC):
    """Abstract interface for all feature modules in the GUI app."""

    name: str = "Unnamed Module"

    def __init__(self, notebook: ttk.Notebook) -> None:
        self.parent_notebook = notebook
        self.frame = ttk.Frame(notebook, padding=10)

    @abstractmethod
    def build(self) -> None:
        """Build module UI on self.frame."""

    def on_show(self) -> None:
        """Called when this module tab becomes visible."""
        return

    def on_close(self) -> None:
        """Called before app exits so modules can release resources."""
        return
