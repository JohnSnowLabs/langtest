import unittest
from langtest.utils.gender_classifier import GenderClassifier


class GenderClassifierTestCase(unittest.TestCase):
    """
    Test case for the Spacy integration in the langtest module.
    """

    def setUp(self) -> None:
        """
        Set up the test case.

        Initializes the parameters for the Harness class.
        """
        self.classifier = GenderClassifier()

        self.examples = [
            ("He is running.", "male"),
            ("She is a nurse.", "female"),
            ("Emily is our top employee.", "female"),
            ("Jack didn't came home last night.", "male"),
        ]

    def test_examples(self):
        """
        Test examples in gender classifier.
        """
        for sentence, true in self.examples:
            pred = self.classifier.predict(sentence)
            self.assertEqual(pred, true)
