import unittest
import mlflow
from langtest import Harness


class MlFlowTesting(unittest.TestCase):
    """
    Test case for the MLflow integration in the langtest module.
    """

    def setUp(self) -> None:
        """
        Set up the test case.

        Initializes the parameters for the Harness class.
        """
        self.params = {
            "task": "ner",
            "model": {"model": "dslim/bert-base-NER", "hub": "huggingface"},
            "data": {"data_source": "tests/fixtures/test.conll"},
            "config": "tests/fixtures/config_ner.yaml",
        }

    def test_mlflow(self):
        """
        Testing mlflow integration
        """
        harness = Harness(**self.params)
        harness.data = harness.data[0:5]
        harness.generate().run().report(mlflow_tracking=True)
        experiment_id = mlflow.get_experiment_by_name(self.params["model"])
        self.assertIsNotNone(experiment_id)
