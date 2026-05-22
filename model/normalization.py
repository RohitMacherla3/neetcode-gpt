import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        x_hat = (x - np.mean(x)) / np.sqrt(x.std()**2 + 1e-5)
        out = gamma * x_hat + beta
        return np.round(out, 5)
