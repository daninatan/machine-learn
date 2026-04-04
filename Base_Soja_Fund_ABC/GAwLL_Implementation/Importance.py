import numpy as np
from typing import Tuple


class Importance:
    """
    Tracks the empirical importance (VImp) of each variable.

    By storing the historical bounds of performance impact for each gene, 
    this class provides data for posterior XAI analysis.

    Attributes:
        degree (int): Number of features in the chromosome.
        max_importance (np.ndarray): Vector storing the highest observed impact per gene.
        min_importance (np.ndarray): Vector storing the lowest observed impact per gene.
    """

    def __init__(self, degree: int):
        """
        Initializes the importance tracking vectors using sentinel values.

        Args:
            degree (int): The number of features (chromosome size).
        """
        self.degree: int = degree

        # Initialized to extremes (-1.0 and 1.0) to ensure the first 
        # performance observation correctly updates both bounds.
        self.max_importance: np.ndarray = np.full(degree, -1.0, dtype=float)
        self.min_importance: np.ndarray = np.full(degree, 1.0, dtype=float)

    def add_importance(self, i: int, importance: float) -> None:
        """
        Updates the maximum and minimum importance bounds for variable 'i'.

        Args:
            i: Index of the variable being updated.
            importance: The measured performance impact.
        """
        if importance > self.max_importance[i]:
            self.max_importance[i] = importance

        if importance < self.min_importance[i]:
            self.min_importance[i] = importance

    def export_importance_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the VImp vectors after normalizing unobserved variables.

        This method handles variables that were never targeted during mutation, 
        ensuring that sentinel values (-1.0, 1.0) are replaced by 0.0 to 
        maintain data integrity for reporting and XAI visualization.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Max_Importance, Min_Importance)
        """
        # Create copies to avoid modifying the internal state permanently 
        # during the normalization process
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