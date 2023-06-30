import unittest
from langtest import Harness


class TranslationTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = Harness(
            task="translation",
            model='t5-base',
            hub="huggingface",
            data="Translation-test"
        )

        # configure the harness
        self.harness.configure({
            'model_parameters': {
                'target_language': 'de'
                }, 
            'tests': {
                'defaults': {'min_pass_rate': '1.0,'},
                'robustness': {
                    'add_typo': {'min_pass_rate': 0.7}, 
                    'lowercase': {'min_pass_rate': 0.7}
                    }
                }
        })
        self.harness.data = self.harness.data[:5]

    def test_translation_workflow(self):
        """Test translation"""
        self.harness.generate().run().report()
    