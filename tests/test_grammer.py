import unittest
from langtest import Harness
import pandas as pd
from langtest.transform.grammar import GrammarTestFactory, Paraphrase, BaseGrammar


class TestGrammar(unittest.TestCase):
    """A test case class for testing the `GrammarTestFactory` class."""

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.data = [
            "This is a test sentence.",
            "This is another test sentence.",
        ]
        self.supported_tests = GrammarTestFactory.available_tests()
        self.harness = Harness(
            task="text-classification",
            model={"model": "lvwerra/distilbert-imdb", "hub": "huggingface"},
        )
        self.harness.configure(
            {
                "tests": {
                    "defaults": {"min_pass_rate": 1.0},
                    "grammar": {
                        "paraphrase": {"min_pass_rate": 0.7},
                    },
                }
            }
        )

    def test_harness_with_grammer(self):
        """
        Test the `Harness` class with the grammar tests.
        """
        self.harness.data = self.harness.data[:4]
        self.harness.generate().run()
        self.assertIsInstance(self.harness.testcases(), pd.DataFrame)
        self.assertIsInstance(self.harness.generated_results(), pd.DataFrame)
        self.assertIsInstance(self.harness.report(), pd.DataFrame)

    def test_grammar(self):
        """
        Test the `GrammarTestFactory` class.
        """
        sens = Paraphrase.transform(self.data)
        self.assertIsInstance(sens, list)
        self.assertIsInstance(sens[0], str)

    def test_grammar_testfactory(self):
        """
        Test the `GrammarTestFactory` class.
        """
        self.assertIsInstance(self.supported_tests, dict)
        # check if the paraphrase test is from the BaseGrammar class
        self.assertTrue(Paraphrase in BaseGrammar.__subclasses__())
        self.assertIsInstance(self.supported_tests["paraphrase"].alias_name, str)
        self.assertIsInstance(self.supported_tests["paraphrase"].supported_tasks, list)

        # check transform, run and async_run attributes in class
        self.assertTrue(hasattr(self.supported_tests["paraphrase"], "transform"))
        self.assertTrue(hasattr(self.supported_tests["paraphrase"], "run"))
        self.assertTrue(hasattr(self.supported_tests["paraphrase"], "async_run"))
