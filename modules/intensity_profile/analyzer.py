from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .models import ProfileSettings


class IntensityProfileAnalyzer:
    @staticmethod
    def bilinear_sample(image: np.ndarray, y: float, x: float) -> float:
        h, w = image.shape
        x = float(np.clip(x, 0.0, w - 1))
        y = float(np.clip(y, 0.0, h - 1))

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        dx = x - x0
        dy = y - y0

        v00 = float(image[y0, x0])
        v01 = float(image[y0, x1])
        v10 = float(image[y1, x0])
        v11 = float(image[y1, x1])

        top = (1.0 - dx) * v00 + dx * v01
        bottom = (1.0 - dx) * v10 + dx * v11
        return (1.0 - dy) * top + dy * bottom

    @staticmethod
    def resolve_fixed_index(length: int, fixed_ratio: float) -> int:
        idx = int(length * fixed_ratio)
        return int(np.clip(idx, 0, length - 1))

    @staticmethod
    def resolve_scan_range(length: int, start_ratio: float, end_ratio: float) -> Tuple[int, int]:
        start = int(np.clip(int(length * start_ratio), 0, length - 1))
        end = int(np.clip(int(length * end_ratio), 0, length - 1))

        if end < start:
            start, end = end, start
        if end == start and end < length - 1:
            end += 1

        return start, end

    @staticmethod
    def averaged_profile(
        image: np.ndarray,
        fixed_axis: str,
        fixed_index: int,
        scan_start: int,
        scan_end: int,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 1")

        half = window_size // 2
        h, w = image.shape

        if fixed_axis == "width":
            if not (0 <= fixed_index < w):
                raise ValueError(f"fixed_index out of range for width axis: {fixed_index}")
            scan_start = int(np.clip(scan_start, 0, h - 1))
            scan_end = int(np.clip(scan_end, 0, h - 1))
            w0 = max(0, fixed_index - half)
            w1 = min(w - 1, fixed_index + half)

            indices = np.arange(scan_start, scan_end + 1)
            values = np.array([np.mean(image[h_idx, w0 : w1 + 1]) for h_idx in indices], dtype=np.float64)

        elif fixed_axis == "height":
            if not (0 <= fixed_index < h):
                raise ValueError(f"fixed_index out of range for height axis: {fixed_index}")
            scan_start = int(np.clip(scan_start, 0, w - 1))
            scan_end = int(np.clip(scan_end, 0, w - 1))
            h0 = max(0, fixed_index - half)
            h1 = min(h - 1, fixed_index + half)

            indices = np.arange(scan_start, scan_end + 1)
            values = np.array([np.mean(image[h0 : h1 + 1, w_idx]) for w_idx in indices], dtype=np.float64)

        else:
            raise ValueError("fixed_axis must be 'height' or 'width'")

        return indices, values

    def compute_axis(self, image: np.ndarray, settings: ProfileSettings) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
        h, w = image.shape

        if settings.fixed_axis == "width":
            fixed_index = self.resolve_fixed_index(w, settings.fixed_ratio)
            scan_start, scan_end = self.resolve_scan_range(h, settings.scan_start_ratio, settings.scan_end_ratio)
        elif settings.fixed_axis == "height":
            fixed_index = self.resolve_fixed_index(h, settings.fixed_ratio)
            scan_start, scan_end = self.resolve_scan_range(w, settings.scan_start_ratio, settings.scan_end_ratio)
        else:
            raise ValueError("fixed_axis must be 'height' or 'width'")

        indices, values = self.averaged_profile(
            image=image,
            fixed_axis=settings.fixed_axis,
            fixed_index=fixed_index,
            scan_start=scan_start,
            scan_end=scan_end,
            window_size=settings.window_size,
        )

        return fixed_index, scan_start, scan_end, indices, values

    def compute_line(self, image: np.ndarray, settings: ProfileSettings) -> Tuple[None, None, None, np.ndarray, np.ndarray]:
        if settings.line_start is None or settings.line_end is None:
            raise ValueError("Please select line start and end points.")

        if settings.window_size < 1 or settings.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 1")

        h, w = image.shape
        x0 = settings.line_start[0] * max(w - 1, 1)
        y0 = settings.line_start[1] * max(h - 1, 1)
        x1 = settings.line_end[0] * max(w - 1, 1)
        y1 = settings.line_end[1] * max(h - 1, 1)

        dx = x1 - x0
        dy = y1 - y0
        length = float(np.hypot(dx, dy))
        if length < 1e-8:
            raise ValueError("Line length is too small. Please select two different points.")

        nx = -dy / length
        ny = dx / length

        samples = max(int(settings.line_samples), 2)
        t = np.linspace(0.0, 1.0, samples)
        values = np.empty(samples, dtype=np.float64)
        half = settings.window_size // 2

        for i, tv in enumerate(t):
            cx = x0 + dx * tv
            cy = y0 + dy * tv
            local_vals = []
            for offset in range(-half, half + 1):
                sx = cx + nx * offset
                sy = cy + ny * offset
                local_vals.append(self.bilinear_sample(image, sy, sx))
            values[i] = float(np.mean(local_vals))

        return None, None, None, t, values

    def compute(
        self,
        image: np.ndarray,
        settings: ProfileSettings,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], np.ndarray, np.ndarray]:
        if settings.profile_mode == "axis":
            return self.compute_axis(image, settings)
        if settings.profile_mode in ("line", "multi"):
            return self.compute_line(image, settings)
        raise ValueError("profile_mode must be 'axis', 'line', or 'multi'")
