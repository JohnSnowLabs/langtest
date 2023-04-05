import unittest
from nlptest.transform.perturbation import *
from nlptest.transform.utils import A2B_DICT


class PerturbationTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.sentences = [
            Sample(original="I live in London, United Kingdom since 2019"),
            Sample(original="I cannot live in USA due to torandos caramelized")
        ]
        self.british_sentences = [
            Sample(original="I live in London, United Kingdom since 2019"),
            Sample(original="I cannot live in USA due to torandos caramelised")
        ]
        self.contraction_sentences = [
            Sample(original="I live in London, United Kingdom since 2019"),
            Sample(original="I can't live in USA due to torandos caramelized")
        ]

        self.labels = [
            ["O", "O", "O", "B-LOC", "B-COUN", "I-COUN", "O", "O", "B-DATE"],
            ["O", "O", "O", "O", "B-COUN", "O", "O", "O", "O", "O"],
        ]

        self.terminology = {
            "LOC": ["London"],
            "COUN": ["United Kingdom", "USA"],
            "DATE": ["2019"],
        }

    def test_uppercase(self) -> None:
        test_cases = UpperCase.transform(self.sentences)
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))
        for test_case in test_cases:
            self.assertTrue(test_case.test_case.isupper())

    def test_lowercase(self) -> None:
        test_cases = LowerCase.transform(self.sentences)
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))
        for test_case in test_cases:
            self.assertTrue(test_case.test_case.islower())

    def test_titlecase(self) -> None:
        test_cases = TitleCase.transform(self.sentences)
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))
        for test_case in test_cases:
            self.assertTrue(test_case.test_case.istitle())

    def test_add_punctuation(self) -> None:
        test_cases = AddPunctuation.transform(self.sentences)
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))
        for test_case in test_cases:
            self.assertFalse(test_case.test_case[-1].isalnum())

    def test_strip_punctuation(self) -> None:
        test_cases = StripPunctuation.transform(self.sentences)
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))
        for test_case in test_cases:
            self.assertTrue(test_case.test_case[-1].isalnum())

    def test_add_typo(self) -> None:
        test_cases = AddTypo.transform(self.sentences)
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))
        for i, test_case in enumerate(test_cases):
            self.assertNotEqual(self.sentences[i], test_case.test_case)

    def test_swap_entities(self) -> None:
        test_cases = SwapEntities.transform(
            sample_list=self.sentences,
            labels=self.labels,
            terminology=self.terminology
        )
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))

    def test_american_to_british(self) -> None:
        test_cases = ConvertAccent.transform(
            sample_list=self.sentences,
            accent_map=A2B_DICT
        )
        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))

    def test_add_context(self) -> None:
        start_context = [["Hello"]]
        end_context = [["Bye"]]
        test_cases = AddContext.transform(
            sample_list=self.sentences,
            starting_context=start_context,
            ending_context=end_context,
            strategy="combined"
        )

        self.assertIsInstance(test_cases, list)
        self.assertEqual(len(self.sentences), len(test_cases))
        for test_case in test_cases:
            self.assertTrue(test_case.test_case.startswith(start_context[0][0]))
            self.assertTrue(test_case.test_case.endswith(end_context[0][0]))

    def test_add_contraction(self) -> None:
        test_cases = AddContraction.transform(self.sentences)
        self.assertIsInstance(test_cases, list)