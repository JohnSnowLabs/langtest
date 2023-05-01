import unittest
import pandas as pd
from nlptest import Harness


class AccuracyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        """"""

        self.h_spacy = Harness(
            task="ner",
            model="en_core_web_sm",
            data="nlptest/data/conll/sample.conll",
            hub="spacy"
        )
        self.h_spacy.configure(
                {'tests': {
                    'defaults': {
                        'min_pass_rate': 0.65,
                    },
                    'accuracy': {
                        'min_f1_score': {
                            'min_score': 0.65
                        }
                    }
                }
            })
        self.report = self.h_spacy.generate().run().report()

    def test_report(self):
        """"""
        self.assertIsInstance(self.report, pd.DataFrame)
