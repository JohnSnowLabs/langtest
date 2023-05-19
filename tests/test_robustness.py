import unittest
from nlptest.transform.robustness import *
from nlptest.transform.utils import A2B_DICT
from nlptest.utils.custom_types import SequenceClassificationSample


class RobustnessTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.sentences = [
            SequenceClassificationSample(original="I live in London, United Kingdom since 2019"),
            SequenceClassificationSample(original="I cannot live in USA due to torandos caramelized")
        ]
        self.number_sentences = [
            SequenceClassificationSample(original="I live in London, United Kingdom since 2019"),
            SequenceClassificationSample(original="I can't move to the USA because they have an average of 1000 tornadoes a year, and I'm terrified of them")
        ]
        self.sentences_with_punctuation = [
            SequenceClassificationSample(original="I live in London, United Kingdom since 2019."),
            SequenceClassificationSample(original="I cannot live in USA due to torandos caramelized!")
        ]
        self.british_sentences = [
            SequenceClassificationSample(original="I live in London, United Kingdom since 2019"),
            SequenceClassificationSample(original="I cannot live in USA due to torandos caramelised")
        ]
        self.contraction_sentences = [
            SequenceClassificationSample(original="I live in London, United Kingdom since 2019"),
            SequenceClassificationSample(original="I can't live in USA due to torandos caramelized")
        ]
        self.gendered_sentences = [
            SequenceClassificationSample(original="He lives in the USA."),
            SequenceClassificationSample(original="He lives in the USA and his cat is black.")
        ]
        self.ocr_sentences = [
            SequenceClassificationSample(original="This organization's art can win tough acts."),    
            SequenceClassificationSample(original="Anyone can join our community garden.")                                                                                        
        ]
        self.labels = [
            ["O", "O", "O", "B-LOC", "B-COUN", "I-COUN", "O", "B-DATE"],
            ["O", "O", "O", "O", "B-COUN", "O", "O", "O", "O", "O"],
        ]

        self.terminology = {
            "LOC": ["Chelsea"],
            "COUN": ["Spain", "Italy"],
            "DATE": ["2017"],
        }

    def test_uppercase(self) -> None:
        """"""
        transformed_samples = UpperCase.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.isupper())

    def test_lowercase(self) -> None:
        """"""
        transformed_samples = LowerCase.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.islower())

    def test_titlecase(self) -> None:
        """"""
        transformed_samples = TitleCase.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.istitle())

    def test_add_punctuation(self) -> None:
        """"""
        transformed_samples = AddPunctuation.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertFalse(sample.test_case[-1].isalnum())
            self.assertEqual(len(sample.transformations), 1)

    def test_strip_punctuation(self) -> None:
        """"""
        transformed_samples = StripPunctuation.transform(self.sentences_with_punctuation)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)
            self.assertEqual(len(sample.transformations), 1)

    def test_swap_entities(self) -> None:
        """"""
        transformed_samples = SwapEntities.transform(
            sample_list=self.sentences,
            labels=self.labels,
            terminology=self.terminology
        )
        for sample in transformed_samples:
            self.assertEqual(len(sample.transformations), 1)
            self.assertNotEqual(sample.test_case, sample.original)

    def test_american_to_british(self) -> None:
        """"""
        transformed_samples = ConvertAccent.transform(
            sample_list=self.sentences,
            accent_map=A2B_DICT
        )
        self.assertIsInstance(transformed_samples, list)
        self.assertListEqual(
            [sample.test_case for sample in transformed_samples],
            [sample.original for sample in self.british_sentences]
        )

    def test_add_context(self) -> None:
        """"""
        start_context = ["Hello"]
        end_context = ["Bye"]
        transformed_samples = AddContext.transform(
            sample_list=self.sentences,
            starting_context=start_context,
            ending_context=end_context,
            strategy="combined"
        )

        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.startswith(start_context[0]))
            self.assertTrue(sample.test_case.endswith(end_context[0]))

    def test_add_contraction(self) -> None:
        """"""
        transformed_samples = AddContraction.transform(self.sentences)
        self.assertListEqual(
            [sample.test_case for sample in transformed_samples],
            [sample.original for sample in self.contraction_sentences]
        )
        self.assertEqual(
            [len(sample.transformations) for sample in transformed_samples],
            [0, 1]
        )
        
    def test_number_to_word(self) -> None:
        """"""
        transformed_samples = NumberToWord.transform(self.number_sentences)
        # Test that the transformed_samples sentences are in a list
        self.assertIsInstance(transformed_samples, list)


    def test_add_ocr_typo(self) -> None:
        """"""
        expected_corrected_sentences = [ "Tbis organization's a^rt c^an w^in tougb acts.",
                                        "Anyone c^an j0in o^ur communitv gardcn."]
        transformed_samples = AddOcrTypo.transform(self.ocr_sentences)
        
        self.assertIsInstance(transformed_samples, list)
        self.assertListEqual(
            [sample.test_case for sample in transformed_samples],
             expected_corrected_sentences
           )