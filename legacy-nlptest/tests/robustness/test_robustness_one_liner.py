import unittest
import os
from nlptest.robustness.robustness_fixing import test_and_augment_robustness
import sparknlp
from sparknlp.annotator import *
from pyspark.ml import Pipeline


class TestRobustnessOneLiner(unittest.TestCase):

    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        # Create a SparkSession object
        self.spark = sparknlp.start()

        # Create a pipeline model object
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        embeddings = WordEmbeddingsModel.pretrained('glove_100d') \
            .setInputCols(["document", 'token']) \
            .setOutputCol("embeddings")

        ner = NerDLModel.pretrained("ner_dl", 'en') \
            .setInputCols(["document", "token", "embeddings"]) \
            .setOutputCol("ner")

        ner_pipeline = Pipeline().setStages([document_assembler, tokenizer, embeddings, ner])

        self.pipeline_model = ner_pipeline.fit(self.spark.createDataFrame([[""]]).toDF("text"))

        # Set the test file path
        self.test_file_path = os.path.join(THIS_DIR, os.pardir, 'resources/test.conll')

        # Set the augmented file path
        self.augmented_file_path = os.path.join(THIS_DIR, os.pardir, 'resources/augmented_test.conll')

        # Delete the augmented file if it has been created already
        if os.path.exists(self.augmented_file_path):
            os.remove(self.augmented_file_path)

    def test_one_liner(self):
        robustness_test_results = test_and_augment_robustness(spark=self.spark,
                                                              pipeline_model=self.pipeline_model,
                                                              test_file_path=self.test_file_path,
                                                              conll_path_to_augment=self.test_file_path,
                                                              conll_save_path=self.augmented_file_path,
                                                              noise_prob=0.5,
                                                              metrics_output_format='dictionary')

        # Assert that the test_robustness function returns a dictionary
        self.assertIsInstance(robustness_test_results, dict)

        # Assert that test_robustness output has keys 'metrics', 'comparison_df', and 'test_details' in the dictionary
        self.assertIn('metrics', robustness_test_results)
        self.assertIn('comparison_df', robustness_test_results)
        self.assertIn('test_details', robustness_test_results)

        # Assert that test_robustness 'metrics' output has keys for each perturbation
        for test_name in ['capitalization_upper', 'capitalization_lower', 'capitalization_title',
                          'add_punctuation', 'strip_punctuation', 'introduce_typos', 'add_contractions', 'add_context',
                          'american_to_british', 'swap_entities', 'swap_cohyponyms']:
            self.assertIn(test_name, robustness_test_results['metrics'])

        self.assertTrue(os.path.exists(self.augmented_file_path))


if __name__ == '__main__':
    unittest.main()
