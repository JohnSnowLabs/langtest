import os
import unittest
from nlptest import Harness
from nlptest.utils.custom_types import Sample
from nlptest.modelhandler.modelhandler import ModelFactory


class HarnessTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        """"""
        cls.data_path = "tests/fixtures/test.conll"
        cls.config_path = "tests/fixtures/config_ner.yaml"
        cls.harness = Harness(
            task='ner',
            model='dslim/bert-base-NER',
            data=cls.data_path,
            config=cls.config_path,
            hub="huggingface"
        )

        cls.harness.generate().run()

    def test_Harness(self):
        """"""
        self.assertIsInstance(self.harness, Harness)

    def test_missing_parameter(self):
        """"""
        with self.assertRaises(OSError) as _:
            Harness(task='ner', model='dslim/bert-base-NER',
                    data=self.data_path, config=self.config_path)

    def test_attributes(self):
        """
        Testing Attributes of Harness Class
        """
        self.assertIsInstance(self.harness.task, str)
        self.assertIsInstance(self.harness.model, (str, ModelFactory))
        self.assertIsInstance(self.harness._config, (str, dict))

    def test_generate_testcases(self):
        """"""
        load_testcases = self.harness._testcases
        self.assertIsInstance(load_testcases, list)
        self.assertIsInstance(load_testcases[0], Sample)

    def test_run_testcases(self):
        """"""
        robustness_results = self.harness._generated_results
        self.assertIsInstance(robustness_results, list)
        self.assertIsInstance(robustness_results[0], Sample)

    def test_report(self):
        """"""
        df = self.harness.report()
        self.assertCountEqual(
            df.columns.tolist(),
            ['category', 'test_type', 'fail_count', 'pass_count', 'pass_rate',
       'minimum_pass_rate', 'pass'] )

    def test_save(self):
        """"""
        save_dir = "/tmp/saved_harness_test"
        self.harness.save(save_dir)

        self.assertCountEqual(os.listdir(save_dir), [
                              'config.yaml', 'test_cases.pkl', 'data.pkl'])
