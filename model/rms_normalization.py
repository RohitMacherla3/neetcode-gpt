import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        rms = np.sqrt(np.mean(x**2))
        x_hat = x / (rms + eps)
        out = x_hat * gamma
        return np.round(out, 4)
