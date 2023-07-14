from typing import Any, Dict

from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

from langtest import Harness
from langtest.datahandler.datasource import ConllDataset
from langtest.pipelines.utils.data_helpers import NERDataset
from .pipeline import BaseEnd2EndPipeline


class NEREnd2EndPipeline(BaseEnd2EndPipeline):
    """NER pipeline for Huggingface models"""

    def __init__(
        self,
        model_name: str,
        train_data: str,
        eval_data: str,
        config: Dict,
        training_args: Dict[str, Any],
        **kwargs,
    ):
        """Constructor method

        Args:
            model_name (str): name of the pretrained model to load
            train_data (str): path to the train dataset
            eval_data (str): path to the evaluation dataset
            config (Union[str, Dict[str, Any]]): tests configuration
            training_args (Dict[str, Any]): training arguments to pass to the Trainer
        """
        super().__init__("ner", model_name, "huggingface", train_data, eval_data, config)

        if "output_dir" not in training_args:
            training_args["output_dir"] = "checkpoints"

        self.training_args = training_args

    def setup(self):
        """Performs all the necessary set up steps"""
        self.train_datasource = ConllDataset(file_path=self.train_data, task=self.task)
        self.eval_datasource = ConllDataset(file_path=self.eval_data, task=self.task)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train(self):
        """Performs the training procedure of the model"""
        tokens, labels = self.train_datasource.load_raw_data()
        self.train_dataset = NERDataset(
            tokens=tokens, labels=labels, tokenizer=self.tokenizer
        )

        tokens, labels = self.eval_datasource.load_raw_data()
        self.eval_dataset = NERDataset(
            tokens=tokens, labels=labels, tokenizer=self.tokenizer
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.train_dataset.label_map),
            id2label=self.train_dataset.id2label,
            label2id=self.train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
        self.trainer.train()
        self.model.save_pretrained(self.training_args["output_dir"])
        self.tokenizer.save_pretrained(self.training_args["output_dir"])

    def evaluate(self):
        """Performs the evaluation procedure on the given test set"""
        self.metrics = self.trainer.evaluate()

    def test(self):
        """Performs the testing procedure of the model on a set of tests using langtest"""
        self.harness = Harness(
            task=self.task,
            model=self.training_args["output_dir"],
            hub=self.hub,
            data=self.train_data,
        )
        self.harness.configure(self.config)
        self.harness.generate()
        self.harness.save(save_dir="saved_harness")
        self.harness.run()
        self.harness.report(format="dataframe", save_dir="first_report")

    def augment(self):
        """Performs the data augmentation procedure based on langtest"""
        self.harness.augment(
            input_path=self.train_data,
            output_path=f"augmented_{self.train_data}",
            export_mode="add",
        )

    def retrain(self):
        """Performs the training procedure using the augmented data created by langtest"""
        self.augmented_train_datasource = ConllDataset(
            file_path=f"augmented_{self.train_data}", task=self.task
        )
        tokens, labels = self.augmented_train_datasource.load_raw_data()

        self.augmented_train_dataset = NERDataset(
            tokens=tokens, labels=labels, tokenizer=self.tokenizer
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.augmented_train_dataset.label_map),
            id2label=self.augmented_train_dataset.id2label,
            label2id=self.augmented_train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.augmented_train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
        self.trainer.train()
        self.model.save_pretrained(f"augmented_{self.training_args['output_dir']}")
        self.tokenizer.save_pretrained(f"augmented_{self.training_args['output_dir']}")

    def reevaluate(self):
        """Performs the evaluation procedure of the model training on the augmented dataset"""
        self.augmented_metrics = self.trainer.evaluate()

    def compare(self):
        """Performs the comparison between the two trained models"""
        print("Metrics before augmentation:", self.metrics)
        print("Metrics after augmentation:", self.augmented_metrics)
