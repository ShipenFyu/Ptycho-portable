from __future__ import annotations

import numpy as np
from matplotlib.colors import hsv_to_rgb


class HsvFusionProcessor:
    @staticmethod
    def phase_to_hue(phase: np.ndarray) -> np.ndarray:
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

    def fuse(self, amplitude: np.ndarray, phase: np.ndarray, saturation: float) -> np.ndarray:
        amp = amplitude.astype(np.float32)
        phase_values = phase.astype(np.float32)

        hsv_image = np.zeros((amp.shape[0], amp.shape[1], 3), dtype=np.float32)
        hsv_image[..., 0] = self.phase_to_hue(phase_values)
        hsv_image[..., 1] = saturation

        amp_max = float(np.max(amp))
        hsv_image[..., 2] = amp / amp_max if amp_max > 0 else amp

        return hsv_to_rgb(hsv_image)
