import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        x      = np.array(x,      dtype=np.float64)
        W1     = np.array(W1,     dtype=np.float64)
        b1     = np.array(b1,     dtype=np.float64)
        W2     = np.array(W2,     dtype=np.float64)
        b2     = np.array(b2,     dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)


        z1    = x @ W1.T + b1
        a1    = np.maximum(0, z1)
        y_hat = a1 @ W2.T + b2              
        loss  = np.mean((y_hat - y_true)**2)

        n = len(y_true) if y_true.ndim > 0 else 1
        dy_hat = 2 * (y_hat - y_true) / n
        dW2    = dy_hat.reshape(-1, 1) @ a1.reshape(1, -1)  
        db2    = dy_hat

        da1 = dy_hat.reshape(-1, 1) @ W2
        da1 = da1.flatten()

        dz1 = da1 * (z1 > 0).astype(float)
        dW1    = dz1.reshape(-1, 1) @ x.reshape(1, -1)    
        db1 = dz1

        return {
            'loss': np.round(loss, 4),
            'dW1':  np.round(dW1, 4).tolist(),
            'db1':  np.round(db1, 4).tolist(),
            'dW2':  np.round(dW2, 4).tolist(),
            'db2':  np.round(db2, 4).tolist()
        }

        
