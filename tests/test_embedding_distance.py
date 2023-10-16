import unittest
import numpy as np
from langtest.metrics import EmbeddingDistance


class TestEmbeddingDistance(unittest.TestCase):
    """Test cases for the EmbeddingDistance class and its distance metrics."""

    def setUp(self):
        """
        Set up the test by initializing two sample vectors.

        This method creates two sample vectors for testing the distance metrics.
        """
        self.vector1 = np.array([[1.0, 2.0, 3.0]])
        self.vector2 = np.array([[2.0, 3.0, 4.0]])

    def test_cosine_similarity(self):
        """
        Test the cosine similarity distance metric.

        This test checks the correctness of the cosine similarity distance metric.
        It ensures that the result is a float, within the range [-1, 1], and not None.
        """
        ed = EmbeddingDistance()
        result = ed["cosine"](self.vector1, self.vector2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1)
        self.assertLessEqual(result, 1)
        self.assertIsNotNone(result)

    def test_euclidean_distance(self):
        """
        Test the Euclidean distance metric.

        This test checks the correctness of the Euclidean distance metric.
        It ensures that the result is a float, greater than or equal to 0, and not None.
        """
        ed = EmbeddingDistance()
        result = ed["euclidean"](self.vector1, self.vector2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertIsNotNone(result)

    def test_manhattan_distance(self):
        """
        Test the Manhattan distance metric.

        This test checks the correctness of the Manhattan distance metric.
        It ensures that the result is a float, greater than or equal to 0, and not None.
        """
        ed = EmbeddingDistance()
        result = ed["manhattan"](self.vector1, self.vector2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertIsNotNone(result)

    def test_chebyshev_distance(self):
        """
        Test the Chebyshev distance metric.

        This test checks the correctness of the Chebyshev distance metric.
        It ensures that the result is a float, greater than or equal to 0, and not None.
        """
        ed = EmbeddingDistance()
        result = ed["chebyshev"](self.vector1, self.vector2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertIsNotNone(result)

    def test_hamming_distance(self):
        """
        Test the Hamming distance metric.

        This test checks the correctness of the Hamming distance metric.
        It ensures that the result is a float, greater than or equal to 0, and less than or equal to 1, and not None.
        """
        ed = EmbeddingDistance()
        result = ed["hamming"](self.vector1, self.vector2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)
        self.assertIsNotNone(result)
