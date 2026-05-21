import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # to avoid log(0)
        res = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log((1 - y_pred)))
        return np.round(res, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # to avoid log(0)
        summ = np.sum(y_true * np.log(y_pred), axis = 1)
        res = float(-np.mean(summ))
        return np.round(res, 4)
