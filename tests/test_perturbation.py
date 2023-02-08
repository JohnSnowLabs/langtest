from typing import List
import unittest

from nlptest.transform.perturbation import *

class PerturbationTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.sentences = [
            "I live in a London since 2019",
            "I cannot live in USA due to torandos."
        ]


    def test_uppercase(self):
        upper_test_cases = UpperCase.transform(
            list_of_strings=self.sentences
        )
        self.assertIsInstance(upper_test_cases, list)
        self.assertIsInstance(upper_test_cases[0], str)
        self.assertTrue(upper_test_cases[0].isupper())
        self.assertEqual(len(self.sentences), len(upper_test_cases))

    
    def test_lowercase(self):
        lower_test_cases = LowerCase.transform(
            list_of_strings = self.sentences
        )

        self.assertIsInstance(lower_test_cases, list)
        self.assertGreater(len(lower_test_cases), 0)
        self.assertTrue(lower_test_cases[0].islower())
        self.assertEqual(len(self.sentences), len(lower_test_cases))
        self.assertIsInstance(lower_test_cases[0], str)

    def test_titlecase(self):
        title_test_cases = TitleCase.transform(
            list_of_strings = self.sentences
        )

        self.assertIsInstance(title_test_cases, list)
        self.assertGreater(len(title_test_cases), 0)
        self.assertTrue(title_test_cases[0].istitle())
        self.assertEqual(len(self.sentences), len(title_test_cases))
        self.assertIsInstance(title_test_cases[0], str)
    

    def test_add_Punctuation(self):
        punc_test_cases = AddPunctuation.transform(
            list_of_strings = self.sentences
        )

        self.assertIsInstance(punc_test_cases, list)
        self.assertGreater(len(punc_test_cases), 0)
        self.assertTrue(punc_test_cases[0][-1].istitle())
        self.assertEqual(len(self.sentences), len(punc_test_cases))
        self.assertIsInstance(punc_test_cases[0], str)
    
    def test_add_context(self):
        start_context = ["Hello"]
        end_context = ["Bye"]
        add_context_test_cases = AddContext.transform(
            list_of_strings=self.sentences,
            starting_context=start_context,
            ending_context=end_context
        )

        self.assertIsInstance(add_context_test_cases, list)
        self.assertGreater(len(add_context_test_cases), 0)
        self.assertIsInstance(add_context_test_cases[0], str)
        self.assertTrue(add_context_test_cases[0][
            0: len(start_context[0])], start_context[0])
        self.assertEqual(len(self.sentences), len(add_context_test_cases))
