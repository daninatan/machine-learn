import numpy as np
from typing import Tuple


class Importance:
    #Store the importance (VImp) of each attribute

    def __init__(self, degree: int):
        #Number of features
        self.degree: int = degree

        self.max_importance: np.ndarray = np.full(degree, -1.0, dtype=float)
        self.min_importance: np.ndarray = np.full(degree, 1.0, dtype=float)

    def add_importance(self, i: int, importance: float) -> None:
        
        #Update the max and min importance of attribute i
        if importance > self.max_importance[i]:
            self.max_importance[i] = importance

        if importance < self.min_importance[i]:
            self.min_importance[i] = importance

    def export_importance_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        
        #Return max and min vectors
        max_v = self.max_importance.copy()
        min_v = self.min_importance.copy()

        # Normalization Logic:
        # 1. If min > max (unvisited variable), sync them
        mask_unvisited = min_v > max_v
        min_v[mask_unvisited] = max_v[mask_unvisited]

        # 2. Reset unobserved placeholders to zero for clean reporting
        max_v[max_v == -1.0] = 0.0
        min_v[min_v == 1.0] = 0.0

        return max_v, min_v