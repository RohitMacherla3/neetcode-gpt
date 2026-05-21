import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:

        z = z - np.max(z)
        num = np.exp(z)
        return np.round(num / np.sum(num), 4)
