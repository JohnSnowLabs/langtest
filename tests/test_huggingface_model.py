import unittest
from langtest.modelhandler import ModelAPI
from langtest.tasks import TaskManager
from langtest.utils.custom_types.helpers import HashableDict


class HuggingFaceTestCase(unittest.TestCase):
    """
    A test case for Hugging Face models.

    This test case performs tests related to loading and handling Hugging Face models using the `ModelAPI` class.

    Attributes:
        models (List[str]): A list of Hugging Face model names.
        tasks (List[str]): A list of tasks associated with the models.
    """

    def setUp(self) -> None:
        """
        Set up the test case.

        This method initializes the list of Hugging Face model names and the associated tasks.
        """
        self.models = [
            "dslim/bert-base-NER",
            "Jean-Baptiste/camembert-ner",
            "d4data/biomedical-ner-all",
        ]

        self.tasks = ["ner", "text-classifier"]

    def test_transformers_ner_models(self):
        """
        Test loading Hugging Face models.

        This method tests the loading of a Hugging Face model using the `ModelAPI` class.
        It asserts that the loaded model is an instance of `ModelAPI`.
        """
        task = TaskManager(self.tasks[0])
        model = task.model(model_path=self.models[0], model_hub="huggingface")
        self.assertIsInstance(model, ModelAPI)

    def test_transformers_QA_models(self):
        """
        Test loading Hugging Face models.

        This method tests the loading of a Hugging Face model using the `ModelAPI` class.
        It asserts that the loaded model is an instance of `ModelAPI`.
        """
        task = TaskManager("question-answering")
        model = task.model(model_path="gpt2", model_hub="huggingface")

        self.assertIsInstance(model, ModelAPI)

    def test_unsupported_task(self):
        """
        Test unsupported task.

        This method tests the behavior of `ModelAPI` when an unsupported task is provided.
        It expects an `AssertionError` to be raised when attempting to create a model with an unsupported task.
        """
        # Raises with unsupported task to model Factory
        with self.assertRaises(AssertionError):
            task = TaskManager(self.tasks[1])
            task.model(model_path=self.models[0], model_hub="huggingface")


class TestTransformersModels(unittest.TestCase):
    """
    Test loading Hugging Face models.

    This class tests the loading of Hugging Face models using the `ModelAPI` class.
    It asserts that the loaded model is an instance of `ModelAPI`.
    """

    def setUp(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        # Define the model identifier
        self.model_id = "gpt2"

        # Load tokenizer and model using transformers library
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.gpt2_model = AutoModelForCausalLM.from_pretrained(self.model_id)

        # Create a text-generation pipeline for reference
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            tokenizer=self.tokenizer,
            max_new_tokens=20,
        )

        # Initialize TaskManager for question-answering task
        self.task = TaskManager("question-answering")

    def _assert_result_conditions(self, model, result):
        """
        Assert conditions for the model result.
        """
        self.assertIsInstance(model, ModelAPI)
        self.assertIsInstance(result, str, "Expected string result")

    def _test_loading_and_prediction(self, model_path):
        """
        Test loading a model and making a prediction.
        """
        # Test loading model
        model = self.task.model(model_path=model_path, model_hub="huggingface")

        # Test model prediction with specific input variables
        result = model.predict(
            prompt=HashableDict(
                **{
                    "template": "Generate a {adjective} sentence about {noun}.",
                    "input_variables": ["adjective", "noun"],
                }
            ),
            text=HashableDict(**{"adjective": "beautiful", "noun": "cats"}),
        )
        # Assert conditions for the result
        self._assert_result_conditions(model, result)

    def test_load_and_predict_gpt2_model(self):
        """Test loading and predicting with GPT-2 model."""
        self._test_loading_and_prediction(self.model_id)

    def test_load_and_predict_pipeline_model(self):
        """Test loading and predicting with a pipeline model."""
        self._test_loading_and_prediction(self.pipe)

    def test_load_and_predict_preloaded_model(self):
        """Test loading and predicting with a pre-loaded model instance."""
        self._test_loading_and_prediction(self.gpt2_model)
