from typing import Any, Dict, Optional


class BaseEnd2EndPipeline:
    """Base end to end pipeline leveraging langtest capabilities to improve a model

    It executes the following workflow in a sequential order:
    - train a model on a given dataset
    - evaluate the model on a given test dataset
    - test the trained model on a set of tests
    - augment the training set based on the tests outcome
    - retrain the model on a the freshly generated augmented training set
    - evaluate the retrained model on the test dataset
    - compare the performance of the two models
    """

    def __init__(
        self,
        task: str,
        model_name: str,
        hub: Optional[str],
        train_data: str,
        eval_data: str,
        config: Dict[str, Any],
        *args,
        **kwargs
    ):
        """Constructor method

        Args:
            task (str): name of the task to perform
            model_name (str): name of the pretrained model to load
            hub (str): name of the hun to load the model from
            train_data (str): path to the train dataset
            eval_data (str): path to the evaluation dataset
            config (Dict[str, Any]): tests configuration
        """
        self.task = task
        self.model_name = model_name
        self.hub = hub
        self.train_data = train_data
        self.eval_data = eval_data
        self.config = config

    def setup(self):
        """Performs all the necessary set up steps"""
        raise NotImplementedError()

    def train(self):
        """Performs the training procedure of the model"""
        raise NotImplementedError()

    def evaluate(self):
        """Performs the evaluation procedure on the given test set"""
        raise NotImplementedError()

    def test(self):
        """Performs the testing procedure of the model on a set of tests using langtest"""
        raise NotImplementedError()

    def augment(self):
        """Performs the data augmentation procedure based on langtest"""
        raise NotImplementedError()

    def retrain(self):
        """Performs the training procedure using the augmented data created by langtest"""
        raise NotImplementedError()

    def reevaluate(self):
        """Performs the evaluation procedure of the model training on the augmented dataset"""
        raise NotImplementedError()

    def compare(self):
        """Performs the comparison between the two trained models"""
        raise NotImplementedError()
