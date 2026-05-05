from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize


class CenteredPowerNorm(Normalize):
    def __init__(self, gamma=0.5, vcenter=0, vmin=-np.pi, vmax=np.pi):
        self.gamma = gamma
        self.vcenter = vcenter
        super().__init__(vmin, vmax)

    def __call__(self, value, clip=None):
        value = np.asarray(value)
        if self.vmin is None or self.vmax is None:
            raise ValueError("Vmin and Vmax must be set before normalization")

        if clip is None:
            clip = self.clip

        vmin = float(self.vmin)
        vmax = float(self.vmax)
        vcenter = float(self.vcenter)

        if vmin >= vmax:
            raise ValueError("vmin must be less than vmax")
        if not (vmin < vcenter < vmax):
            raise ValueError("vcenter must be between vmin and vmax")

        if clip:
            value = np.clip(value, vmin, vmax)

        neg_span = vcenter - vmin
        pos_span = vmax - vcenter
        scale = np.where(value < vcenter, neg_span, pos_span)
        scale = np.where(scale == 0, 1.0, scale)

        signed = (value - vcenter) / scale
        signed = np.clip(signed, -1.0, 1.0)
        mag = np.abs(signed) ** self.gamma
        res = 0.5 + 0.5 * np.sign(signed) * mag

        return np.ma.masked_array(res)
