import os
import unittest
from nlptest import Harness
from nlptest.utils.custom_types import Sample
from nlptest.modelhandler.modelhandler import ModelFactory


class HarnessTestCase(unittest.TestCase):
    # TODO: add tests for the different combination of hub/task
    def setUp(self) -> None:
        """"""
        self.data_path = "tests/fixtures/test.conll"
        self.config_path = "tests/fixtures/config_ner.yaml"
        self.harness = Harness(
            task='ner',
            model='dslim/bert-base-NER',
            data=self.data_path,
            config=self.config_path,
            hub="huggingface"
        )

    def test_Harness(self):
        """"""
        self.assertIsInstance(self.harness, Harness)

    def test_missing_parameter(self):
        """"""
        with self.assertRaises(OSError) as _:
            Harness(task='ner', model='dslim/bert-base-NER',
                    data=self.data_path, config=self.config_path, hub="huggingface")

    def test_attributes(self):
        """
        Testing Attributes of Harness Class
        """
        self.assertIsInstance(self.harness.task, str)
        self.assertIsInstance(self.harness.model, (str, ModelFactory))
        self.assertIsInstance(self.harness._config, (str, dict))

    def test_generate_testcases(self):
        """"""
        load_testcases = self.harness.generate()._testcases
        self.assertIsInstance(load_testcases, list)
        self.assertIsInstance(load_testcases[0], Sample)

    def test_run_testcases(self):
        """"""
        self.harness.generate()
        robustness_results = self.harness.run()._generated_results
        self.assertIsInstance(robustness_results, list)
        self.assertIsInstance(robustness_results[0], Sample)

    def test_report(self):
        """"""
        self.harness.generate()
        df = self.harness.run().accuracy_results
        self.assertCountEqual(df.columns.tolist(), [
                              'test_type', 'test_case', 'actual_result'])

    def test_duplicate_tasks(self):
        """"""
        with self.assertRaises(AssertionError):
            Harness(
                task="text-classifer",
                model=ModelFactory("ner", "dslim/bert-base-NER"),
                data=self.data_path,
                config=self.config_path,
                hub="huggingface"
            )

    def test_save(self):
        """"""
        save_dir = "/tmp/saved_harness_test"
        self.harness.generate()
        self.harness.save(save_dir)

        self.assertCountEqual(os.listdir(save_dir), [
                              'config.yaml', 'test_cases.pkl', 'data.pkl'])

    def test_load_ner(self):
        """"""
        save_dir = "/tmp/saved_ner_harness_test"
        self.harness.generate()
        self.harness.save(save_dir)

        loaded_harness = Harness.load(
            save_dir=save_dir,
            task="ner",
            model="bert-base-cased",
            hub="huggingface"
        )
        self.assertEqual(self.harness._testcases, loaded_harness._testcases)
        self.assertEqual(self.harness._config, loaded_harness._config)
        self.assertEqual(self.harness.data, loaded_harness.data)
        self.assertNotEqual(self.harness.model, loaded_harness.model)

    def test_load_text_classification(self):
        """"""
        save_dir = "/tmp/saved_text_classification_harness_test"
        tc_harness = Harness(
            task='text-classification',
            model='bert-base-cased',
            data="tests/fixtures/text_classification.csv",
            config="tests/fixtures/config_text_classification.yaml",
            hub="huggingface"
        )
        tc_harness.generate()
        tc_harness.save(save_dir)

        loaded_tc_harness = Harness.load(
            save_dir=save_dir,
            task="text-classification",
            model="bert-base-uncased",
            hub="huggingface"
        )
        self.assertEqual(tc_harness._testcases, loaded_tc_harness._testcases)
        self.assertEqual(tc_harness._config, loaded_tc_harness._config)
        self.assertEqual(tc_harness.data, loaded_tc_harness.data)
        self.assertNotEqual(tc_harness.model, loaded_tc_harness.model)
