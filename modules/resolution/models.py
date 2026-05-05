from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class ImageData:
    def __init__(self, path: Path, display: np.ndarray, intensity: np.ndarray, is_rgb: bool):
        self.path = path
        self.display = display
        self.intensity = intensity
        self.is_rgb = is_rgb

    @property
    def shape(self) -> Tuple[int, int]:
        return self.intensity.shape


@dataclass
class EdgeLine:
    center_x: float
    center_y: float
    tangent_x: float
    tangent_y: float
    normal_x: float
    normal_y: float
    angle_deg: float


@dataclass
class KnifeEdgeResult:
    roi_bounds: Tuple[int, int, int, int]
    edge_line: EdgeLine
    distances: np.ndarray
    esf: np.ndarray
    esf_smooth: np.ndarray
    lsf: np.ndarray
    frequencies: np.ndarray
    mtf: np.ndarray
    mtf50: Optional[float]
    mtf10: Optional[float]
    edge_width_10_90: Optional[float]
    bin_width: float
