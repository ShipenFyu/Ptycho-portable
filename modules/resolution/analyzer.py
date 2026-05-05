from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .models import EdgeLine, KnifeEdgeResult


class KnifeEdgeAnalyzer:
    def analyze(
        self,
        image: np.ndarray,
        roi_bounds: Tuple[int, int, int, int],
        oversampling: int = 8,
        smoothing_window: int = 7,
    ) -> KnifeEdgeResult:
        x0, y0, x1, y1 = roi_bounds
        roi = np.asarray(image[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float64)
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            raise ValueError("ROI is too small. Please select a larger knife-edge region.")

        roi = self.normalize_image(roi)
        edge_line = self.estimate_edge_line(roi)
        distances, esf = self.compute_esf(roi, edge_line, oversampling)
        if len(esf) < 16:
            raise ValueError("Not enough ESF samples. Please use a larger ROI or a less steep edge.")

        esf_smooth = self.smooth_curve(esf, smoothing_window)
        lsf = np.gradient(esf_smooth, distances)
        frequencies, mtf = self.compute_mtf(lsf, distances)

        return KnifeEdgeResult(
            roi_bounds=roi_bounds,
            edge_line=edge_line,
            distances=distances,
            esf=esf,
            esf_smooth=esf_smooth,
            lsf=lsf,
            frequencies=frequencies,
            mtf=mtf,
            mtf50=self.find_mtf_crossing(frequencies, mtf, 0.50),
            mtf10=self.find_mtf_crossing(frequencies, mtf, 0.10),
            edge_width_10_90=self.compute_edge_width(distances, esf_smooth),
            bin_width=float(np.median(np.diff(distances))),
        )

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        vmin = float(np.percentile(image, 1))
        vmax = float(np.percentile(image, 99))
        if vmax <= vmin:
            return np.zeros_like(image, dtype=np.float64)
        return np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)

    def estimate_edge_line(self, roi: np.ndarray) -> EdgeLine:
        gy, gx = np.gradient(roi)
        mag = np.hypot(gx, gy)
        threshold = float(np.percentile(mag, 85))
        ys, xs = np.nonzero(mag >= threshold)
        if len(xs) < 8:
            raise ValueError("Could not find a strong edge in the ROI.")

        weights = mag[ys, xs]
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("Could not find a strong edge in the ROI.")

        cx = float(np.sum(xs * weights) / total)
        cy = float(np.sum(ys * weights) / total)
        coords = np.column_stack((xs - cx, ys - cy))
        weighted = coords * np.sqrt(weights[:, None])
        _, _, vh = np.linalg.svd(weighted, full_matrices=False)
        tx, ty = vh[0]
        norm = float(np.hypot(tx, ty))
        if norm <= 1e-12:
            raise ValueError("Could not estimate edge direction.")
        tx = float(tx / norm)
        ty = float(ty / norm)
        nx = float(-ty)
        ny = float(tx)

        mean_gx = float(np.sum(gx[ys, xs] * weights) / total)
        mean_gy = float(np.sum(gy[ys, xs] * weights) / total)
        if mean_gx * nx + mean_gy * ny < 0:
            nx = -nx
            ny = -ny

        angle = float(np.degrees(np.arctan2(ty, tx)))
        return EdgeLine(center_x=cx, center_y=cy, tangent_x=tx, tangent_y=ty, normal_x=nx, normal_y=ny, angle_deg=angle)

    def compute_esf(self, roi: np.ndarray, edge_line: EdgeLine, oversampling: int) -> Tuple[np.ndarray, np.ndarray]:
        oversampling = max(2, int(oversampling))
        bin_width = 1.0 / oversampling

        h, w = roi.shape
        yy, xx = np.indices((h, w), dtype=np.float64)
        distances = (xx - edge_line.center_x) * edge_line.normal_x + (yy - edge_line.center_y) * edge_line.normal_y
        values = roi.ravel()
        dist = distances.ravel()

        dmin = float(np.floor(np.min(dist) / bin_width) * bin_width)
        dmax = float(np.ceil(np.max(dist) / bin_width) * bin_width)
        bins = np.arange(dmin, dmax + bin_width, bin_width)
        if len(bins) < 4:
            raise ValueError("ROI does not span enough distance across the edge.")

        sums, _ = np.histogram(dist, bins=bins, weights=values)
        counts, _ = np.histogram(dist, bins=bins)
        valid = counts > 0
        centers = (bins[:-1] + bins[1:]) * 0.5
        esf = np.zeros_like(centers)
        esf[valid] = sums[valid] / counts[valid]
        centers = centers[valid]
        esf = esf[valid]

        if len(esf) >= 10:
            head = float(np.mean(esf[: max(3, len(esf) // 10)]))
            tail = float(np.mean(esf[-max(3, len(esf) // 10) :]))
            if tail < head:
                centers = -centers[::-1]
                esf = esf[::-1]

        return centers, esf

    @staticmethod
    def smooth_curve(values: np.ndarray, window_size: int) -> np.ndarray:
        window_size = max(1, int(window_size))
        if window_size % 2 == 0:
            window_size += 1
        if window_size <= 1 or len(values) < window_size:
            return np.asarray(values, dtype=np.float64)
        kernel = np.ones(window_size, dtype=np.float64) / window_size
        pad = window_size // 2
        padded = np.pad(values, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def compute_mtf(lsf: np.ndarray, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(lsf) < 4:
            raise ValueError("LSF is too short for MTF calculation.")
        spacing = float(np.median(np.diff(distances)))
        if spacing <= 0:
            raise ValueError("Invalid ESF sample spacing.")

        centered = lsf - np.mean(lsf[: max(2, len(lsf) // 10)])
        window = np.hanning(len(centered))
        spectrum = np.abs(np.fft.rfft(centered * window))
        frequencies = np.fft.rfftfreq(len(centered), d=spacing)
        if len(spectrum) == 0 or spectrum[0] <= 1e-12:
            mtf = np.zeros_like(spectrum)
        else:
            mtf = spectrum / spectrum[0]
        return frequencies, mtf

    @staticmethod
    def find_mtf_crossing(frequencies: np.ndarray, mtf: np.ndarray, level: float) -> Optional[float]:
        if len(frequencies) < 2:
            return None
        below = np.nonzero(mtf <= level)[0]
        below = below[below > 0]
        if len(below) == 0:
            return None
        idx = int(below[0])
        x0, x1 = float(frequencies[idx - 1]), float(frequencies[idx])
        y0, y1 = float(mtf[idx - 1]), float(mtf[idx])
        if abs(y1 - y0) <= 1e-12:
            return x1
        return x0 + (level - y0) * (x1 - x0) / (y1 - y0)

    @staticmethod
    def compute_edge_width(distances: np.ndarray, esf: np.ndarray) -> Optional[float]:
        if len(distances) < 4:
            return None
        lo = float(np.percentile(esf, 5))
        hi = float(np.percentile(esf, 95))
        if hi <= lo:
            return None
        normalized = np.clip((esf - lo) / (hi - lo), 0.0, 1.0)

        def crossing(level: float) -> Optional[float]:
            idxs = np.nonzero(normalized >= level)[0]
            if len(idxs) == 0:
                return None
            idx = int(idxs[0])
            if idx == 0:
                return float(distances[0])
            x0, x1 = float(distances[idx - 1]), float(distances[idx])
            y0, y1 = float(normalized[idx - 1]), float(normalized[idx])
            if abs(y1 - y0) <= 1e-12:
                return x1
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0)

        p10 = crossing(0.10)
        p90 = crossing(0.90)
        if p10 is None or p90 is None:
            return None
        return abs(p90 - p10)
