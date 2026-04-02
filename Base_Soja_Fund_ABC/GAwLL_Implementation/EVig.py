import numpy as np
from typing import Tuple


class eVIG:
    """
    Manages the Interaction Matrix (VInt) for feature linkage discovery.

    The eVIG maintains two symmetric adjacency matrices that store the
    maximum (positive/synergistic) and minimum (negative/antagonistic)
    interaction weights observed between pairs of variables.

    Attributes:
        degree (int): The number of features (dimension of the square matrices).
    """

    def __init__(self, degree: int):
        """
        Initializes the interaction matrices with zeros.

        Args:
            degree (int): Total number of variables/features in the dataset.
        """
        self.degree: int = degree

        # _max_edge_weight: Stores the highest positive interaction (synergy).
        self._max_edge_weight: np.ndarray = np.zeros((degree, degree), dtype=float)

        # _min_edge_weight: Stores the lowest negative interaction (antagonism).
        self._min_edge_weight: np.ndarray = np.zeros((degree, degree), dtype=float)

    def add_edge(self, a: int, b: int, w: float) -> None:
        """
        Records an interaction weight between feature 'a' and feature 'b'.

        This method updates the symmetric matrices if the new weight 'w'
        represents a new maximum or minimum for the pair.

        Args:
            a (int): Index of the first feature.
            b (int): Index of the second feature.
            w (float): The interaction weight (Performance Delta).
        """
        # Symmetric update for the Maximum Interaction Matrix (Synergy)
        if w > self._max_edge_weight[a, b]:
            self._max_edge_weight[a, b] = w
            self._max_edge_weight[b, a] = w

        # Symmetric update for the Minimum Interaction Matrix (Antagonism)
        if w < self._min_edge_weight[a, b]:
            self._min_edge_weight[a, b] = w
            self._min_edge_weight[b, a] = w

    def export_interaction_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns copies of the internal interaction matrices.

        Returns a tuple containing the Max and Min matrices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Max_Matrix, Min_Matrix)
        """
        return self._max_edge_weight.copy(), self._min_edge_weight.copy()