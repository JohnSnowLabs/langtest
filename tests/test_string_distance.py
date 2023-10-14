import unittest
from langtest.metrics.string_distance import StringDistance


class TestStringDistance(unittest.TestCase):
    def setUp(self):
        self.text1 = "hello"
        self.text2 = "hola"

    def test_normalized_jaro_distance(self):
        """
        Test Jaro distance calculation.

        Ensure that the result is a float and not None.
        """
        sd = StringDistance()
        result = sd["jaro"](self.text1, self.text2)
        self.assertIsInstance(result, float)
        self.assertIsNotNone(result)

    def test_normalized_jaro_winkler_distance(self):
        """
        Test Jaro-Winkler distance calculation.

        Ensure that the result is a float and not None.
        """
        sd = StringDistance()
        result = sd["jaro_winkler"](self.text1, self.text2)
        self.assertIsInstance(result, float)
        self.assertIsNotNone(result)

    def test_normalized_hamming_distance(self):
        """
        Test Hamming distance calculation.

        Ensure that the result is a float and not None.
        """
        sd = StringDistance()
        result = sd["hamming"](self.text1, self.text2)
        self.assertIsInstance(result, float)
        self.assertIsNotNone(result)

    def test_normalized_levenshtein_distance(self):
        """
        Test Levenshtein distance calculation.

        Ensure that the result is a float and not None.
        """
        sd = StringDistance()
        result = sd["levenshtein"](self.text1, self.text2)
        self.assertIsInstance(result, float)
        self.assertIsNotNone(result)

    def test_normalized_damerau_levenshtein_distance(self):
        """
        Test Damerau-Levenshtein distance calculation.

        Ensure that the result is a float and not None.
        """
        sd = StringDistance()
        result = sd["damerau_levenshtein"](self.text1, self.text2)
        self.assertIsInstance(result, float)
        self.assertIsNotNone(result)

    def test_normalized_indel_distance(self):
        """
        Test Indel distance calculation.

        Ensure that the result is a float and not None.
        """
        sd = StringDistance()
        result = sd["indel"](self.text1, self.text2)
        self.assertIsInstance(result, float)
        self.assertIsNotNone(result)
