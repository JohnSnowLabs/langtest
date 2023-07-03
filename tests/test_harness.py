import os
import unittest
import pandas as pd
from langtest import Harness
from langtest.modelhandler.modelhandler import ModelFactory
from langtest.utils.custom_types import Sample

class HarnessTestCase(unittest.TestCase):
    """
    Test case for the Harness class.
    """
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
        """
        Test the Harness instance.
        """
        self.assertIsInstance(self.harness, Harness)

    def test_missing_parameter(self):
        """
        Test handling of missing parameters in Harness instantiation.
        """
        with self.assertRaises(ValueError) as _:
            Harness(task='ner', model='dslim/bert-base-NER',
                    data=self.data_path, config=self.config_path)

    def test_attributes(self):
        """
        Test the attributes of the Harness class.
        """
        self.assertIsInstance(self.harness.task, str)
        self.assertIsInstance(self.harness.model, (str, ModelFactory))
        self.assertIsInstance(self.harness._config, (str, dict))

    def test_generate_testcases(self):
        """
        Test generating test cases.
        """
        load_testcases = self.harness._testcases
        self.assertIsInstance(load_testcases, list)
        self.assertIsInstance(load_testcases[0], Sample.__constraints__)

    def test_run_testcases(self):
        """
        Test running test cases.
        """
        robustness_results = self.harness._generated_results
        self.assertIsInstance(robustness_results, list)
        self.assertIsInstance(robustness_results[0], Sample.__constraints__)

    def test_report(self):
        """
        Test generating a report.
        """
        df = self.harness.report()
        self.assertCountEqual(
            df.columns.tolist(),
            ['category', 'test_type', 'fail_count', 'pass_count', 'pass_rate',
             'minimum_pass_rate', 'pass'])

    def test_incompatible_tasks(self):
        """
        Test handling of incompatible tasks.
        """
        with self.assertRaises(ValueError):
            Harness(
                task="text-classifer",
                model="dslim/bert-base-NER",
                data=self.data_path,
                config=self.config_path,
                hub="huggingface"
            )

    def test_unsupported_test_for_task(self):
        """
        Test handling of unsupported tests for a task.
        """
        with self.assertRaises(ValueError):
            h = Harness(
                task="text-classification",
                model="textcat_imdb",
                hub="spacy",
                config={'tests':{'robustness':{'swap_entities':{'min_pass_rate':0.5}}}}
            )
            h.generate()

    def test_save(self):
        """
        Test saving the Harness object.
        """
        save_dir = "/tmp/saved_harness_test"
        self.harness.save(save_dir)

        self.assertCountEqual(os.listdir(save_dir), [
            'config.yaml', 'test_cases.pkl', 'data.pkl'])

    def test_load_ner(self):
        """
        Test loading a saved Harness object for NER task.
        """
        save_dir = "/tmp/saved_ner_harness_test"
        self.harness.save(save_dir)

        loaded_harness = Harness.load(
            save_dir=save_dir,
            task="ner",
            model="bert-base-cased",
            hub="huggingface"
        )
        self.assertEqual(self.harness._config, loaded_harness._config)
        self.assertEqual(self.harness.data, loaded_harness.data)
        self.assertNotEqual(self.harness.model, loaded_harness.model)

    def test_load_text_classification(self):
        """
        Test loading a saved Harness object for text classification task.
        """
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
        self.assertEqual(tc_harness._config, loaded_tc_harness._config)
        self.assertEqual(tc_harness.data, loaded_tc_harness.data)
        self.assertNotEqual(tc_harness.model, loaded_tc_harness.model)

    def test_load_HF_data_text_classification(self):
        """
        Test loading a saved Harness object with Hugging Face data for text classification task.
        """
        save_dir = "/tmp/saved_HF_data_text_classification_harness_test"
        tc_harness = Harness(task="text-classification", hub="huggingface",
                            model="aychang/roberta-base-imdb",
                            data={"name":'imdb'}
                                )
        tc_harness.data=tc_harness.data[:10]
        tc_harness.generate()
        tc_harness.save(save_dir)

        loaded_tc_harness = Harness.load(
            save_dir=save_dir,
            task="text-classification",
            model="aychang/roberta-base-imdb",
            hub="huggingface"
        )
        self.assertEqual(tc_harness._config, loaded_tc_harness._config)
        self.assertEqual(tc_harness.data, loaded_tc_harness.data)
        self.assertNotEqual(tc_harness.model, loaded_tc_harness.model)
    
    def test_harness_edit_import_testcases(self):
        """"""
        save_dir = "/tmp/saved_harness_edit_import_testcases"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        harness = Harness(
            task="ner",
            model="bert-base-cased",
            data="tests/fixtures/test.conll",
            hub="huggingface"
        )
        harness.data = harness.data[:10]
        harness.generate()

        # edit the testcases
        harness.edit_testcases(save_dir+"/test_cases.csv")

        # test the file is generated or not
        self.assertTrue(os.path.exists(save_dir+"/test_cases.csv"))

        # remove the testcases randomly
        df = pd.read_csv(save_dir+"/test_cases.csv")
        df = df.sample(frac=0.5)
        df.to_csv(save_dir+"/test_cases.csv", index=False)

        # import the testcases
        harness.import_edited_testcases(save_dir+"/test_cases.csv")

        # test working of the harness
        harness.run().report()

class DefaultCodeBlocksTestCase(unittest.TestCase):
    """
    Test case for the default code blocks.
    """
    def test_non_existing_default(self):
        """
        Test handling of non-existing default models.
        """
        with self.assertRaises(ValueError):
            h = Harness(task="ner", model="xxxxxxxxx", hub="spacy")

    def test_ner_spacy(self):
        """
        Test NER task with Spacy model.
        """
        h = Harness(task="ner", model="en_core_web_sm", hub="spacy")
        h.generate().run().report()

    def test_ner_hf(self):
        """
        Test NER task with Hugging Face model.
        """
        h = Harness(task="ner", model="dslim/bert-base-NER", hub="huggingface")
        h.generate().run().report()

    def test_ner_jsl(self):
        """
        Test NER task with John Snow Labs model.
        """
        h = Harness(task="ner", model="ner_dl_bert", hub="johnsnowlabs")
        h.generate().run().report()

    def test_text_classification_spacy(self):
        """
        Test text classification task with Spacy model.
        """
        h = Harness(task="text-classification", model="textcat_imdb", hub="spacy")
        h.generate().run().report()

    def test_text_classification_hf(self):
        """
        Test text classification task with Hugging Face model.
        """
        h = Harness(task="text-classification", model="lvwerra/distilbert-imdb",
                    hub="huggingface")
        h.generate().run().report()

    def test_text_classification_jsl(self):
        """
        Test text classification task with John Snow Labs model.
        """
        try:
            h = Harness(task="text-classification", model="en.sentiment.imdb.glove", hub="johnsnowlabs")
            h.generate().run().report()
        except Exception as e:
            self.fail(f"Test failed with the following error:\n{e}")
