from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TensorData:
    name: str
    values: np.ndarray
    display: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape
