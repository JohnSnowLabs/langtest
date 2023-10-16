import unittest
from langtest.metrics import StringDistance


class TestStringDistance(unittest.TestCase):
    def setUp(self):
        self.text1 = "hello"
        self.text2 = "hola"

    def assert_normalized_distance(self, distance_name, result):
        """
        Assert that the result is a float, not None, and falls within the range [0, 1].

        Args:
            distance_name (str): The name of the distance metric for documentation purposes.
            result (float): The result of the distance calculation.
        """
        self.assertIsInstance(result, float)
        self.assertIsNotNone(result)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_normalized_jaro_distance(self):
        """
        Test Jaro distance calculation.

        Ensure that the result is a float, not None, and falls within the range [0, 1].
        """
        sd = StringDistance()
        result = sd["jaro"](self.text1, self.text2)
        self.assert_normalized_distance("jaro", result)

    def test_normalized_jaro_winkler_distance(self):
        """
        Test Jaro-Winkler distance calculation.

        Ensure that the result is a float, not None, and falls within the range [0, 1].
        """
        sd = StringDistance()
        result = sd["jaro_winkler"](self.text1, self.text2)
        self.assert_normalized_distance("jaro_winkler", result)

    def test_normalized_hamming_distance(self):
        """
        Test Hamming distance calculation.

        Ensure that the result is a float, not None, and falls within the range [0, 1].
        """
        sd = StringDistance()
        result = sd["hamming"](self.text1, self.text2)
        self.assert_normalized_distance("hamming", result)

    def test_normalized_levenshtein_distance(self):
        """
        Test Levenshtein distance calculation.

        Ensure that the result is a float, not None, and falls within the range [0, 1].
        """
        sd = StringDistance()
        result = sd["levenshtein"](self.text1, self.text2)
        self.assert_normalized_distance("levenshtein", result)

    def test_normalized_damerau_levenshtein_distance(self):
        """
        Test Damerau-Levenshtein distance calculation.

        Ensure that the result is a float, not None, and falls within the range [0, 1].
        """
        sd = StringDistance()
        result = sd["damerau_levenshtein"](self.text1, self.text2)
        self.assert_normalized_distance("damerau_levenshtein", result)

    def test_normalized_indel_distance(self):
        """
        Test Indel distance calculation.

        Ensure that the result is a float, not None, and falls within the range [0, 1].
        """
        sd = StringDistance()
        result = sd["indel"](self.text1, self.text2)
        self.assert_normalized_distance("indel", result)
