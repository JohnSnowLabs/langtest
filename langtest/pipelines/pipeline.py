from metaflow import FlowSpec, JSONType, Parameter, step


class BaseEnd2EndPipeline(FlowSpec):
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

    task = Parameter("task", help="Name of the task to perform", type=str, required=True)
    model_name = Parameter(
        "model_name", help="Name of the pretrained model to load", type=str, required=True
    )
    hub = Parameter(
        "hub", help="Name of the hub to load the model from", type=str, required=True
    )
    train_data = Parameter(
        "train_data", help="Path to the train dataset", type=str, required=True
    )
    eval_data = Parameter(
        "eval_data", help="Path to the evaluation dataset", type=str, required=True
    )
    config = Parameter("config", help="Tests configuration", type=JSONType, required=True)

    @step
    def start(self):
        """Starting step of the flow (required by Metaflow)"""
        self.next(self.setup)

    @step
    def setup(self):
        """Performs all the necessary set up steps"""
        raise NotImplementedError()

    @step
    def train(self):
        """Performs the training procedure of the model"""
        raise NotImplementedError()

    @step
    def evaluate(self):
        """Performs the evaluation procedure on the given test set"""
        raise NotImplementedError()

    @step
    def test(self):
        """Performs the testing procedure of the model on a set of tests using langtest"""
        raise NotImplementedError()

    @step
    def augment(self):
        """Performs the data augmentation procedure based on langtest"""
        raise NotImplementedError()

    @step
    def retrain(self):
        """Performs the training procedure using the augmented data created by langtest"""
        raise NotImplementedError()

    @step
    def reevaluate(self):
        """Performs the evaluation procedure of the model training on the augmented dataset"""
        raise NotImplementedError()

    @step
    def compare(self):
        """Performs the comparison between the two trained models"""
        raise NotImplementedError()

    @step
    def end(self):
        """Ending step of the flow (required by Metaflow)"""
        raise NotImplementedError()
