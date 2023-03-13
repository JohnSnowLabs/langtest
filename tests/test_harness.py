import sys
import unittest

sys.path.insert(0, '..')

from nlptest import Harness
from nlptest.modelhandler.modelhandler import ModelFactory
from nlptest.utils.custom_types import Sample


class HarnessTestCase(unittest.TestCase):

    def setUp(self) -> None:
        """"""
        self.data_path = "./demo/data/test.conll"
        self.config_path = "./demo/data/config.yml"
        self.harness = Harness(
            task='ner',
            model='dslim/bert-base-NER',
            data=self.data_path,
            config=self.config_path,
            hub="transformers"
        )

    def test_Harness(self):
        """"""
        self.assertIsInstance(self.harness, Harness)

    def test_missing_parameter(self):
        """"""
        with self.assertRaises(OSError) as _:
            Harness(task='ner', model='dslim/bert-base-NER', data=self.data_path, config=self.config_path)

    def test_attributes(self):
        """
        Testing Attributes of Harness Class
        """
        self.assertIsInstance(self.harness.task, str)
        self.assertIsInstance(self.harness.model, (str, ModelFactory))
        # self.assertIsInstance(self.harness.data, (str, DataFactory)) 
        self.assertIsInstance(self.harness._config, (str, dict))

    def test_generate_testcases(self):
        """"""
        load_testcases = self.harness.generate().load_testcases
        self.assertIsInstance(load_testcases, list)
        self.assertIsInstance(load_testcases[0], Sample)

    def test_run_testcases(self):
        """"""
        self.harness.generate()
        robustness_results = self.harness.run().generated_results
        self.assertIsInstance(robustness_results, list)
        self.assertIsInstance(robustness_results[0], Sample)

    def test_report(self):
        """"""
        self.harness.generate()
        df = self.harness.run().accuracy_results
        self.assertCountEqual(df.columns.tolist(), ['test_type', 'test_case', 'actual_result'])

    def test_duplicate_tasks(self):
        """"""
        with self.assertRaises(AssertionError):
            Harness(
                task="text-classifer",
                model=ModelFactory("ner", "dslim/bert-base-NER"),
                data=self.data_path,
                config=self.config_path
            )
