from __future__ import annotations

import numpy as np


class TensorAdapter:
    @staticmethod
    def as_2d_float(array: np.ndarray, tensor_name: str) -> np.ndarray:
        arr = np.asarray(array)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Tensor '{tensor_name}' is not 2D after squeeze. Got shape={arr.shape}")
        if np.iscomplexobj(arr):
            raise ValueError(f"Tensor '{tensor_name}' must be real-valued, but got complex dtype={arr.dtype}")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def amplitude_values(array: np.ndarray, tensor_name: str) -> np.ndarray:
        return TensorAdapter.as_2d_float(array, tensor_name)

    @staticmethod
    def phase_values(array: np.ndarray, tensor_name: str) -> np.ndarray:
        return TensorAdapter.as_2d_float(array, tensor_name)
