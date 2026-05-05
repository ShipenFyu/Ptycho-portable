from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import ImageData


class ImageLoader:
    @staticmethod
    def load_image(path: Path) -> ImageData:
        arr = plt.imread(path)

        if arr.ndim == 2:
            intensity = np.asarray(arr, dtype=np.float32)
            display = intensity
            is_rgb = False
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                intensity = np.asarray(arr[:, :, 0], dtype=np.float32)
                display = intensity
                is_rgb = False
            else:
                rgb = np.asarray(arr[:, :, :3], dtype=np.float32)
                display = rgb
                intensity = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
                is_rgb = True
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        return ImageData(path=path, display=display, intensity=np.asarray(intensity, dtype=np.float32), is_rgb=is_rgb)
