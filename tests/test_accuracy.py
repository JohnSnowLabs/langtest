import unittest

from nlptest import Harness


class AccuracyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.h_spacy = Harness("ner", model="en_core_web_sm",
                               data="demo/data/test.conll", hub="spacy")
        self.h_spacy.configure(
            {'defaults': {
                'min_pass_rate': 0.65,
                'min_representation': 50
            },
             'tests': {
                 'robustness': {
                     'lowercase': {
                         'min_pass_rate': 0.65
                     }
                 }
             }
            })
        self.h_spacy.generate().run().report()

    def test_accuracy_report(self) -> None:
        acc_report = self.h_spacy.generated_results()
        self.assertGreaterEqual(acc_report.shape[0], 1)
        self.assertEqual(acc_report.shape[1], 7)
