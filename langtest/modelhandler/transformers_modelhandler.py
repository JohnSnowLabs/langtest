from typing import Dict, List, Tuple, Union

import numpy as np
from transformers import Pipeline, pipeline

from .modelhandler import _ModelHandler
from ..utils.custom_types import (
    NEROutput,
    NERPrediction,
    SequenceClassificationOutput,
    TranslationOutput,
)
from langtest.utils.lib_manager import try_import_lib
import importlib


class PretrainedModelForNER(_ModelHandler):
    """Transformers pretrained model for NER tasks

    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace NER pipeline for predictions.
    """

    def __init__(self, model):
        """Constructor method

        Args:
            model (transformers.pipeline.Pipeline): Pretrained HuggingFace NER pipeline for predictions.
        """
        assert isinstance(model, Pipeline), ValueError(
            f"Invalid transformers pipeline! "
            f"Pipeline should be '{Pipeline}', passed model is: '{type(model)}'"
        )
        self.model = model

    @staticmethod
    def _aggregate_words(predictions: List[Dict]) -> List[Dict]:
        """Aggregates predictions at a word-level by taking the first token label.

        Args:
            predictions (List[Dict]):
                predictions obtained with the pipeline object
        Returns:
            List[Dict]:
                aggregated predictions
        """
        aggregated_words = []
        for prediction in predictions:
            if prediction["word"].startswith("##"):
                aggregated_words[-1]["word"] += prediction["word"][2:]
                aggregated_words[-1]["end"] = prediction["end"]
            elif (
                len(aggregated_words) > 0
                and prediction["start"] == aggregated_words[-1]["end"]
            ):
                aggregated_words[-1]["word"] += prediction["word"]
                aggregated_words[-1]["end"] = prediction["end"]
            else:
                aggregated_words.append(prediction)
        return aggregated_words

    @staticmethod
    def _get_tag(entity_label: str) -> Tuple[str, str]:
        """Retrieve the tag of a BIO label

        Args:
            entity_label (str):
                BIO style label
        Returns:
            Tuple[str,str]:
                tag, label
        """
        if entity_label.startswith("B-") or entity_label.startswith("I-"):
            marker, tag = entity_label.split("-")
            return marker, tag
        return "I", "O"

    @staticmethod
    def _group_sub_entities(entities: List[dict]) -> dict:
        """Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": " ".join(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities(self, entities: List[Dict]) -> List[Dict]:
        """Find and group together the adjacent tokens with the same entity predicted.

        Inspired and adapted from:
        https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/pipelines/token_classification.py#L421

        Args:
            entities (List[Dict]):
                The entities predicted by the pipeline.

        Returns:
            List[Dict]:
                grouped entities
        """
        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            bi, tag = self._get_tag(entity["entity"])
            last_bi, last_tag = self._get_tag(entity_group_disagg[-1]["entity"])

            if tag == "O":
                entity_groups.append(self._group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
            elif tag == last_tag and bi != "B":
                entity_group_disagg.append(entity)
            else:
                entity_groups.append(self._group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            entity_groups.append(self._group_sub_entities(entity_group_disagg))
        return entity_groups

    @classmethod
    def load_model(cls, path: str) -> "Pipeline":
        """Load the NER model into the `model` attribute.

        Args:
            path (str):
                path to model or model name

        Returns:
            'Pipeline':
        """
        return pipeline(model=path, task="ner", ignore_labels=[])

    def predict(self, text: str, **kwargs) -> NEROutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            kwargs: Additional keyword arguments.

        Keyword Args:
            group_entities (bool): Option to group entities.

        Returns:
            NEROutput: A list of named entities recognized in the input text.
        """
        predictions = self.model(text, **kwargs)
        aggregated_words = self._aggregate_words(predictions)
        aggregated_predictions = self.group_entities(aggregated_words)

        return NEROutput(
            predictions=[
                NERPrediction.from_span(
                    entity=prediction.get("entity_group", prediction.get("entity", None)),
                    word=prediction["word"],
                    start=prediction["start"],
                    end=prediction["end"],
                )
                for prediction in aggregated_predictions
            ]
        )

    def predict_raw(self, text: str) -> List[str]:
        """Predict a list of labels.

        Args:
            text (str): Input text to perform NER on.

        Returns:
            List[str]: A list of named entities recognized in the input text.
        """
        prediction = self.model(text)
        if len(prediction) == 0:
            return []

        if prediction[0].get("entity") is not None:
            return [x["entity"] for x in prediction]
        return [x["entity_group"] for x in prediction]

    def __call__(self, text: str, *args, **kwargs) -> NEROutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)


class PretrainedModelForTextClassification(_ModelHandler):
    """Transformers pretrained model for text classification tasks

    Attributes:
        model (transformers.pipeline.Pipeline):
            Loaded Text Classification pipeline for predictions.
    """

    def __init__(
        self,
        model: Pipeline,
    ):
        """Constructor method

        Args:
            model (transformers.pipeline.Pipeline): Pretrained HuggingFace NER pipeline for predictions.
        """
        assert isinstance(model, Pipeline), ValueError(
            f"Invalid transformers pipeline! "
            f"Pipeline should be '{Pipeline}', passed model is: '{type(model)}'"
        )
        self.model = model

    @property
    def labels(self) -> List[str]:
        """Return classification labels of pipeline model."""
        return list(self.model.model.config.id2label.values())

    @classmethod
    def load_model(cls, path: str) -> "Pipeline":
        """Load and return text classification transformers pipeline"""
        return pipeline(model=path, task="text-classification")

    def predict(
        self,
        text: str,
        return_all_scores: bool = False,
        truncation_strategy: str = "longest_first",
        *args,
        **kwargs,
    ) -> SequenceClassificationOutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            return_all_scores (bool): Option to group entities.
            truncation_strategy (str): strategy to use to truncate too long sequences
            kwargs: Additional keyword arguments.

        Returns:
            SequenceClassificationOutput: text classification from the input text.
        """
        if return_all_scores:
            kwargs["top_k"] = len(self.labels)

        output = self.model(text, truncation_strategy=truncation_strategy, **kwargs)
        return SequenceClassificationOutput(text=text, predictions=output)

    def predict_raw(
        self, text: str, truncation_strategy: str = "longest_first"
    ) -> List[str]:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            truncation_strategy (str): strategy to use to truncate too long sequences

        Returns:
            List[str]: Predictions as a list of strings.
        """
        return [
            pred["label"]
            for pred in self.model(text, truncation_strategy=truncation_strategy)
        ]

    def __call__(
        self, text: str, return_all_scores: bool = False, *args, **kwargs
    ) -> SequenceClassificationOutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, return_all_scores=return_all_scores, **kwargs)


class PretrainedModelForTranslation(_ModelHandler):
    """Transformers pretrained model for translation tasks

    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace translation pipeline for predictions.
    """

    def __init__(self, model):
        """Constructor method

        Args:
            model (transformers.pipeline.Pipeline): Pretrained HuggingFace NER pipeline for predictions.
        """
        assert isinstance(model, Pipeline), ValueError(
            f"Invalid transformers pipeline! "
            f"Pipeline should be '{Pipeline}', passed model is: '{type(model)}'"
        )
        self.model = model

    @classmethod
    def load_model(cls, path: str) -> "Pipeline":
        """Load the Translation model into the `model` attribute.

        Args:
            path (str):
                path to model or model name

        Returns:
            'Pipeline':
        """
        from ..langtest import HARNESS_CONFIG as harness_config

        config = harness_config["model_parameters"]
        tgt_lang = config.get("target_language")

        if "t5" in path:
            return pipeline(f"translation_en_to_{tgt_lang}", model=path)
        else:
            return pipeline(model=path, src_lang="en", tgt_lang=tgt_lang)

    def predict(self, text: str, **kwargs) -> TranslationOutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform translation on.
            kwargs: Additional keyword arguments.


        Returns:
            TranslationOutput: Output model for translation tasks
        """
        prediction = self.model(text, **kwargs)[0]["translation_text"]
        return TranslationOutput(translation_text=prediction)

    def __call__(self, text: str, *args, **kwargs) -> TranslationOutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)


class PretrainedModelForQA(_ModelHandler):
    """Transformers pretrained model for QA tasks

    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace QA pipeline for predictions.
    """

    def __init__(self, hub, model, **kwargs):
        """Constructor method

        Args:
            model (transformers.pipeline.Pipeline): Pretrained HuggingFace QA pipeline for predictions.
        """
        assert isinstance(model, Pipeline), ValueError(
            f"Invalid transformers pipeline! "
            f"Pipeline should be '{Pipeline}', passed model is: '{type(model)}'"
        )
        self.model = model

    def _check_langchain_package(self):
        LIB_NAME = "langchain"
        if try_import_lib(LIB_NAME):
            langchain = importlib.import_module(LIB_NAME)
            self.PromptTemplate = getattr(langchain, "PromptTemplate")
        else:
            raise ModuleNotFoundError(
                f"The '{LIB_NAME}' package is not installed. Please install it using 'pip install {LIB_NAME}'."
            )

    @staticmethod
    def load_model(hub: str, path: str, **kwargs) -> "Pipeline":
        """Load the QA model into the `model` attribute.

        Args:
            path (str):
                path to model or model name

        Returns:
            'Pipeline':
        """
        if "task" in kwargs.keys():
            kwargs.pop("task")
        return pipeline(model=path, **kwargs)

    def predict(self, text: Union[str, dict], prompt: dict, **kwargs) -> str:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform QA on.
            kwargs: Additional keyword arguments.


        Returns:
            str: Output model for QA tasks
        """
        prompt_template = self.PromptTemplate(**prompt)
        p = prompt_template.format(**text)
        prediction = self.model(p, **kwargs)
        return prediction[0]["generated_text"][len(p) :]

    def __call__(self, text: Union[str, dict], prompt: dict, **kwargs) -> str:
        """Alias of the 'predict' method"""
        return self.predict(text=text, prompt=prompt, **kwargs)


class PretrainedModelForSummarization(PretrainedModelForQA, _ModelHandler):
    """Transformers pretrained model for summarization tasks

    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace summarization pipeline for predictions.
    """

    pass


class PretrainedModelForToxicity(PretrainedModelForQA, _ModelHandler):
    """Transformers pretrained model for summarization tasks

    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace summarization pipeline for predictions.
    """

    pass


class PretrainedModelForSecurity(PretrainedModelForQA, _ModelHandler):
    """Transformers pretrained model for summarization tasks

    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace summarization pipeline for predictions.
    """

    pass


class PretrainedModelForPolitical(PretrainedModelForQA, _ModelHandler):
    """Transformers pretrained model for summarization tasks

    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace summarization pipeline for predictions.
    """

    pass
