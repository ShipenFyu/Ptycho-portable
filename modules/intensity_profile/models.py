from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


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
