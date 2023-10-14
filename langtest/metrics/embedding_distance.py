import numpy as np
from typing import Callable
import functools


class EmbeddingDistance:
    """
    A utility class for calculating various types of distances and similarities between vectors or arrays.
    This class provides methods for calculating the following distance/similarity measures:
    - Cosine Similarity: Measures the cosine of the angle between two non-zero vectors.
    - Euclidean Distance: Measures the straight-line distance between two points in Euclidean space.
    - Manhattan Distance: Measures the sum of absolute differences between the coordinates of two points.
    - Chebyshev Distance: Measures the maximum absolute difference between coordinates of two points.
    - Hamming Distance: Measures the fraction of differing elements in two binary vectors.
    """

    def validate_input(func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        A decorator function for validating input arrays.

        :param func: The function to be decorated.
        :return: The decorated function.
        """

        @functools.wraps(func)
        def wrapper(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
                raise ValueError(
                    f"Input arrays must be of type 'np.ndarray', but received types: {type(a)} and {type(b)}"
                )
            return func(a, b)

        return wrapper

    def __getitem__(self, name):
        if name in self.available_embedding_distance:
            return self.available_embedding_distance[name]
        else:
            raise KeyError(f"Distance function '{name}' not found")

    @staticmethod
    @validate_input
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the cosine similarity between two arrays of vectors.

        Parameters:
            a (np.ndarray): First array of vectors, shape (n, m).
            b (np.ndarray): Second array of vectors, shape (n, m).

        Returns:
            np.ndarray: An array of cosine similarity values of shape (n,).

        Explanation:
        Cosine similarity measures the cosine of the angle between two vectors.
        Values range from -1 (perfectly dissimilar) to 1 (perfectly similar),
        with 0 indicating orthogonality.
        """
        dot_products = np.einsum("ij,ij->i", a, b)
        magnitudes1 = np.linalg.norm(a, axis=1)
        magnitudes2 = np.linalg.norm(b, axis=1)
        cosine_similarities = dot_products / (magnitudes1 * magnitudes2)
        return cosine_similarities[0]

    @staticmethod
    @validate_input
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two vectors.

        Parameters:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            float: The Euclidean distance.

        Explanation:
        Euclidean distance measures the straight-line distance between two points
        in Euclidean space. It is a non-negative value, with larger values indicating
        greater dissimilarity and smaller values indicating greater similarity.
        """
        return np.linalg.norm(a - b)

    @staticmethod
    @validate_input
    def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the Manhattan distance between two vectors.

        Parameters:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            float: The Manhattan distance.

        Explanation:
        Manhattan distance measures the sum of the absolute differences between the
        coordinates of two points. It provides a measure of similarity based on the
        number of steps required to move from one point to another in a grid-like
        path.
        """
        return np.sum(np.abs(a - b))
