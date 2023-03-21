import unittest

from nlptest import Harness


class AccuracyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.h_spacy = Harness(
            "ner", model="en_core_web_sm", data="demo/data/test.conll")
        self.h_spacy.configure({
            'tasks': ['ner'],
            'tests_types': ['lowercase'],
            'min_pass_rate': {'default': 0.5}
        })
        self.h_spacy.generate().run().report()

    def test_accuracy_report(self) -> None:
        acc_report = self.h_spacy.accuracy_report()
        self.assertGreaterEqual(acc_report.shape[0], 1)
        self.assertEqual(acc_report.shape[1], 5)
