import logging

from metaflow import FlowSpec, JSONType, Parameter, step
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from langtest import Harness
from langtest.datahandler.datasource import ConllDataset
from langtest.pipelines.utils.data_helpers import NERDataset
from langtest.pipelines.utils.metrics import compute_metrics


class NEREnd2EndPipeline(FlowSpec):
    """NER pipeline for Huggingface models"""

    model_name = Parameter(
        "model-name", help="Name of the pretrained model to load", type=str, required=True
    )
    train_data = Parameter(
        "train-data", help="Path to the train dataset", type=str, required=True
    )
    eval_data = Parameter(
        "eval-data", help="Path to the evaluation dataset", type=str, required=True
    )
    config = Parameter(
        "config", help="Tests configuration", type=JSONType, required=False, default=None
    )
    training_args = Parameter(
        "training-args",
        help="Training arguments to pass to the Trainer",
        type=JSONType,
        required=True,
    )

    @step
    def start(self):
        """Starting step of the flow (required by Metaflow)"""
        self.next(self.setup)

    @step
    def setup(self):
        """Performs all the necessary set up steps"""
        self.task = "ner"
        self.hub = "huggingface"
        self.output_dir = "checkpoints/"

        self.train_datasource = ConllDataset(file_path=self.train_data, task=self.task)
        self.eval_datasource = ConllDataset(file_path=self.eval_data, task=self.task)

        self.next(self.train)

    @step
    def train(self):
        """Performs the training procedure of the model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=self.output_dir, **self.training_args),
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Performs the evaluation procedure on the given test set"""
        trained_model = AutoModelForTokenClassification.from_pretrained(
            self.output_dir,
            num_labels=len(self.train_dataset.label_map),
            id2label=self.train_dataset.id2label,
            label2id=self.train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )
        self.metrics = Trainer(
            model=trained_model,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics(list(self.train_dataset.label_map.keys())),
        ).evaluate()

        self.next(self.test)

    @step
    def test(self):
        """Performs the testing procedure of the model on a set of tests using langtest"""
        self.harness = Harness(
            task=self.task,
            model=self.output_dir,
            hub=self.hub,
            data=self.train_data,
        )
        if self.config:
            self.harness.configure(self.config)

        _ = self.harness.generate()
        self.harness.save(save_dir="saved_harness")
        _ = self.harness.run()
        self.harness.report(format="dataframe", save_dir="first_report")

        self.next(self.augment)

    @step
    def augment(self):
        """Performs the data augmentation procedure based on langtest"""
        self.harness.augment(
            input_path=self.train_data,
            output_path=f"augmented_{self.train_data}",
            export_mode="add",
        )

        self.next(self.retrain)

    @step
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

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=f"augmented_{self.output_dir}", **self.training_args
            ),
            train_dataset=self.augmented_train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        self.model.save_pretrained(f"augmented_{self.output_dir}")
        self.tokenizer.save_pretrained(f"augmented_{self.output_dir}")

        self.next(self.reevaluate)

    @step
    def reevaluate(self):
        """Performs the evaluation procedure of the model training on the augmented dataset"""
        newly_trained_model = AutoModelForTokenClassification.from_pretrained(
            f"augmented_{self.output_dir}",
            num_labels=len(self.train_dataset.label_map),
            id2label=self.train_dataset.id2label,
            label2id=self.train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )
        self.augmented_metrics = Trainer(
            model=newly_trained_model,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics(list(self.train_dataset.label_map.keys())),
        ).evaluate()

        self.next(self.compare)

    @step
    def compare(self):
        """Performs the comparison between the two trained models"""
        print("Metrics before augmentation:", self.metrics)
        print("Metrics after augmentation:", self.augmented_metrics)

        self.next(self.end)

    @step
    def end(self):
        """Ending step of the flow (required by Metaflow)"""
        logging.info(f"{self.__class__} successfully ran!")


if __name__ == "__main__":
    NEREnd2EndPipeline()
