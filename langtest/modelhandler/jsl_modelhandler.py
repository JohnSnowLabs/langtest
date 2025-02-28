import os
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Union, Dict, Tuple
from functools import lru_cache

from langtest.utils.custom_types.helpers import HashableDict
from langtest.utils.custom_types.output import TranslationOutput

from ..modelhandler import ModelAPI
from ..utils.custom_types import NEROutput, NERPrediction, SequenceClassificationOutput
from ..utils.lib_manager import try_import_lib
from ..errors import Errors

if try_import_lib("pyspark"):
    from pyspark.ml import PipelineModel

if try_import_lib("johnsnowlabs"):
    from johnsnowlabs import nlp
    from nlu import NLUPipeline

SUPPORTED_SPARKNLP_NER_MODELS = []
SUPPORTED_SPARKNLP_CLASSIFERS = []
SUPPORTED_SPARKNLP_TRANSLATION = []
if try_import_lib("sparknlp"):
    from sparknlp.annotator import (
        AlbertForTokenClassification,
        BertForTokenClassification,
        CamemBertForTokenClassification,
        DeBertaForTokenClassification,
        DistilBertForTokenClassification,
        LongformerForTokenClassification,
        RoBertaForTokenClassification,
        XlmRoBertaForTokenClassification,
        XlnetForTokenClassification,
        NerDLModel,
        ClassifierDLModel,
        SentimentDLModel,
        AlbertForSequenceClassification,
        BertForSequenceClassification,
        DeBertaForSequenceClassification,
        DistilBertForSequenceClassification,
        LongformerForSequenceClassification,
        RoBertaForSequenceClassification,
        XlmRoBertaForSequenceClassification,
        XlnetForSequenceClassification,
        MarianTransformer,
        MultiClassifierDLModel,
    )
    from sparknlp.base import LightPipeline
    from sparknlp.pretrained import PretrainedPipeline

    SUPPORTED_SPARKNLP_NER_MODELS.extend(
        [
            AlbertForTokenClassification,
            BertForTokenClassification,
            CamemBertForTokenClassification,
            DeBertaForTokenClassification,
            DistilBertForTokenClassification,
            LongformerForTokenClassification,
            RoBertaForTokenClassification,
            XlmRoBertaForTokenClassification,
            XlnetForTokenClassification,
            NerDLModel,
        ]
    )

    SUPPORTED_SPARKNLP_CLASSIFERS.extend(
        [
            MultiClassifierDLModel,
            ClassifierDLModel,
            SentimentDLModel,
            AlbertForSequenceClassification,
            BertForSequenceClassification,
            DeBertaForSequenceClassification,
            DistilBertForSequenceClassification,
            LongformerForSequenceClassification,
            RoBertaForSequenceClassification,
            XlmRoBertaForSequenceClassification,
            XlnetForSequenceClassification,
        ]
    )
    SUPPORTED_SPARKNLP_TRANSLATION.extend(
        [
            MarianTransformer,
        ]
    )

if try_import_lib("sparknlp_jsl"):
    from sparknlp_jsl.legal import (
        LegalBertForTokenClassification,
        LegalNerModel,
        LegalBertForSequenceClassification,
        LegalClassifierDLModel,
    )

    from sparknlp_jsl.finance import (
        FinanceBertForTokenClassification,
        FinanceNerModel,
        FinanceBertForSequenceClassification,
        FinanceClassifierDLModel,
    )

    from sparknlp_jsl.annotator import (
        MedicalBertForTokenClassifier,
        MedicalNerModel,
        MedicalBertForSequenceClassification,
        MedicalDistilBertForSequenceClassification,
        MedicalQuestionAnswering,
        MedicalSummarizer,
        ExtractiveSummarization,
        MedicalTextGenerator,
    )

    SUPPORTED_SPARKNLP_NER_MODELS.extend(
        [
            LegalBertForTokenClassification,
            LegalNerModel,
            FinanceBertForTokenClassification,
            FinanceNerModel,
            MedicalBertForTokenClassifier,
            MedicalNerModel,
        ]
    )

    SUPPORTED_SPARKNLP_CLASSIFERS.extend(
        [
            LegalBertForSequenceClassification,
            LegalClassifierDLModel,
            FinanceBertForSequenceClassification,
            FinanceClassifierDLModel,
            MedicalBertForSequenceClassification,
            MedicalDistilBertForSequenceClassification,
        ]
    )

    SUPPORTED_SPARKNLP_QA = [MedicalQuestionAnswering]

    SUPPORTED_SPARKNLP_SUMMARIZATION = [
        MedicalSummarizer,
        ExtractiveSummarization,
    ]

    SUPPORTED_SPARKNLP_TEXT_GENERATION = [MedicalTextGenerator]


class PretrainedJSLModel(ABC):
    """PretrainedJSLModel is an abstract class for handling SparkNLP models.

    Attributes:
        model (Union["NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"]):
            Loaded SparkNLP LightPipeline for inference.
    """

    hub = "johnsnowlabs"

    @abstractmethod
    def __init__(
        self,
        model: Union[
            "NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"
        ],
    ):
        """Constructor method

        Args:
            model (Union["NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"]):
                Loaded SparkNLP LightPipeline for inference.
        """
        if model.__class__.__name__ == "PipelineModel":
            self.model = model

        elif model.__class__.__name__ == "LightPipeline":
            self.model = model.pipeline_model

        elif model.__class__.__name__ == "PretrainedPipeline":
            self.model = model.model

        elif model.__class__.__name__ == "NLUPipeline":
            stages = [comp.model for comp in model.components]
            _pipeline = nlp.Pipeline().setStages(stages)
            tmp_df = model.spark.createDataFrame([[""]]).toDF("text")
            self.model = _pipeline.fit(tmp_df)

        else:
            raise ValueError(Errors.E038(model_type=type(model)))

        self.predict.cache_clear()

    @classmethod
    def load_model(cls, path) -> "NLUPipeline":
        """Load the NER model into the `model` attribute.

        Args:
            path (str): Path to pretrained local or NLP Models Hub SparkNLP model
        """
        if isinstance(path, str):
            if os.path.exists(path):
                if try_import_lib("johnsnowlabs"):
                    loaded_model = nlp.load(path=path)
                else:
                    loaded_model = PipelineModel.load(path)
            else:
                if try_import_lib("johnsnowlabs"):
                    loaded_model = nlp.load(path)
                else:
                    raise ValueError(Errors.E039())

            return cls(loaded_model)
        return cls(path)

    @abstractmethod
    @lru_cache(maxsize=128)
    def predict(self, text: str, *args, **kwargs) -> Any:
        """Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform translation on.
        """
        raise NotImplementedError()

    def __call__(
        self, text: str, *args, **kwargs
    ) -> Union[NEROutput, SequenceClassificationOutput, TranslationOutput]:
        """Alias of the 'predict' method"""
        if isinstance(text, dict):
            text = HashableDict(**text)
        kwargs = {
            k: HashableDict(**v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return self.predict(text=text, *args, **kwargs)


class PretrainedModelForNER(PretrainedJSLModel, ModelAPI):
    """Pretrained model for NER tasks."""

    task = "ner"

    def __init__(
        self,
        model: Union[
            "NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"
        ],
    ):
        """Constructor method

        Args:
            model (LightPipeline):
                Loaded SparkNLP LightPipeline for inference.
        """
        super().__init__(model)
        #   there can be multiple ner model in the pipeline
        #   but at first I will set first as default one. Later we can adjust Harness to test multiple model
        ner_model = None
        for annotator in self.model.stages:
            if self.is_ner_annotator(annotator):
                ner_model = annotator
                break

        if ner_model is None:
            raise ValueError(Errors.E040(var="NER"))

        self.output_col = ner_model.getOutputCol()

        #   in order to overwrite configs, light pipeline should be reinitialized.
        self.model = LightPipeline(self.model)

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
        for i in range(0, len(predictions)):
            aggregated_words.append(
                {
                    "entity": predictions[i].result,
                    "index": i + 1,
                    "word": predictions[i].metadata["word"],
                    "start": predictions[i].begin,
                    "end": (predictions[i].end) + 1,
                }
            )

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
            return entity_label.split("-")
        return "I", "O"

    @staticmethod
    def _group_sub_entities(entities: List[dict]) -> dict:
        """Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
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

            entities = self._get_tag(entity["entity"])
            bi = entities[0]
            tag = "-".join(entities[1:])

            last_entities = self._get_tag(entity_group_disagg[-1]["entity"])
            last_tag = "-".join(last_entities[1:])

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

    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs) -> NEROutput:
        """Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform NER on.

        Returns:
            NEROutput: A list of named entities recognized in the input text.
        """
        prediction = self.model.fullAnnotate(text)[0][self.output_col]
        aggregated_words = self._aggregate_words(prediction)
        aggregated_predictions = self.group_entities(aggregated_words)

        return NEROutput(
            predictions=[
                NERPrediction.from_span(
                    entity=ent["entity_group"],
                    word=ent["word"],
                    start=ent["start"],
                    end=ent["end"],
                )
                for ent in aggregated_predictions
            ]
        )

    def predict_raw(self, text: str) -> List[str]:
        """Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform NER on.

        Returns:
            List[str]: Predicted labels.
        """
        return self.model.annotate(text)[self.output_col]

    @staticmethod
    def is_ner_annotator(model_instance) -> bool:
        """Check ner model instance is supported by langtest"""
        for model in SUPPORTED_SPARKNLP_NER_MODELS:
            if isinstance(model_instance, model):
                return True
        return False


class PretrainedModelForTextClassification(PretrainedJSLModel, ModelAPI):
    """Pretrained model for text classification tasks"""

    task = "text-classification"

    def __init__(
        self,
        model: Union[
            "NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"
        ],
    ):
        """Constructor class

        Args:
            model (LightPipeline):
                Loaded SparkNLP LightPipeline for inference.
        """
        super().__init__(model)

        _classifier = None
        self.multi_label_classifier = False
        for annotator in self.model.stages:
            if self.is_classifier(annotator):
                _classifier = annotator
                break

        if _classifier is None:
            raise ValueError(Errors.E040(var="classifier"))

        if isinstance(_classifier, MultiClassifierDLModel):
            self.multi_label_classifier = True
            self.threshold = _classifier.getThreshold()

        self.output_col = _classifier.getOutputCol()
        self.classes = _classifier.getClasses()
        self.model = LightPipeline(self.model)

    @staticmethod
    def is_classifier(model_instance) -> bool:
        """Check classifier model instance is supported by langtest"""
        for model in SUPPORTED_SPARKNLP_CLASSIFERS:
            if isinstance(model_instance, model):
                return True
        return False

    @lru_cache(maxsize=102400)
    def predict(
        self, text: str, return_all_scores: bool = False, *args, **kwargs
    ) -> SequenceClassificationOutput:
        """Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform NER on.
            return_all_scores (bool): Option to return score for all labels.

        Returns:
            SequenceClassificationOutput: Classification output from SparkNLP LightPipeline.
        """
        prediction_metadata = self.model.fullAnnotate(text)[0][self.output_col]
        prediction = []

        if len(prediction_metadata) > 0:
            prediction_metadata = prediction_metadata[0].metadata
            prediction = [
                {"label": x, "score": y} for x, y in prediction_metadata.items()
            ]

        if self.multi_label_classifier:
            prediction = [x for x in prediction if float(x["score"]) > self.threshold]

            return SequenceClassificationOutput(
                text=text,
                predictions=prediction,
                multi_label=self.multi_label_classifier,
            )

        if not return_all_scores:
            prediction = [max(prediction, key=lambda x: x["score"])]

        return SequenceClassificationOutput(text=text, predictions=prediction)

    def predict_raw(self, text: str) -> List[str]:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform text classification on.

        Returns:
            List[str]: Predictions as a list of strings.
        """
        prediction_metadata = self.model.fullAnnotate(text)[0][self.output_col][
            0
        ].metadata
        prediction = [{"label": x, "score": y} for x, y in prediction_metadata.items()]
        prediction = [max(prediction, key=lambda x: x["score"])]
        return [x["label"] for x in prediction]


class PretrainedModelForTranslation(PretrainedJSLModel, ModelAPI):
    """Pretrained model for translations tasks"""

    task = "translation"

    def __init__(
        self,
        model: Union[
            "NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"
        ],
    ):
        """Constructor class

        Args:
            model (LightPipeline):
                Loaded SparkNLP LightPipeline for inference.
        """
        super().__init__(model)

        _translator = None
        for annotator in self.model.stages:
            if self.is_translator(annotator):
                _translator = annotator
                break

        if _translator is None:
            raise ValueError(Errors.E040(var="translator"))

        self.output_col = _translator.getOutputCol()
        self.model = LightPipeline(self.model)

    @staticmethod
    def is_translator(model_instance) -> bool:
        """Check translator model instance is supported by langtest"""
        for model in SUPPORTED_SPARKNLP_TRANSLATION:
            if isinstance(model_instance, model):
                return True
        return False

    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs) -> TranslationOutput:
        """Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform translation on.

        Returns:
            TranslationOutput: Translation output from SparkNLP LightPipeline.
        """
        prediction_metadata = self.model.fullAnnotate(text)[0]["translation"]
        prediction = [x.result for x in prediction_metadata]

        return TranslationOutput(translation_text=" ".join(prediction))

    def predict_raw(self, text: str) -> List[str]:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform translation on.

        Returns:
            List[str]: Predictions as a list of strings.
        """
        prediction_metadata = self.model.fullAnnotate(text)[0]["translation"]
        prediction = [x.result for x in prediction_metadata]
        return prediction


class PretrainedModelForQA(PretrainedJSLModel, ModelAPI):
    """Pretrained model for question answering tasks"""

    task = "question-answering"

    def __init__(
        self,
        model: Union[
            "NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"
        ],
    ):
        """Constructor class

        Args:
            model (LightPipeline):
                Loaded SparkNLP LightPipeline for inference.
        """
        super().__init__(model)

        self.model_type: Literal["QA", "TextGen"] = "QA"

        _qa_model = None
        for annotator in self.model.stages:
            is_supported, model_type = self.is_qa_annotator(annotator)
            if is_supported:
                break
            self.model_type = model_type

        if _qa_model is None:
            raise ValueError(Errors.E040(var="QA"))

        self.output_col = "answer"
        self.model = LightPipeline(self.model)

    @staticmethod
    def is_qa_annotator(model_instance) -> bool:
        """Check QA model instance is supported by langtest"""
        for model in SUPPORTED_SPARKNLP_QA:
            if isinstance(model_instance, model):
                return True, "QA"
        for model in SUPPORTED_SPARKNLP_TEXT_GENERATION:
            if isinstance(model_instance, model):
                return True, "TextGen"
        return False, None

    @lru_cache(maxsize=102400)
    def predict(self, text: Union[str, Dict], *args, **kwargs) -> Dict[str, Any]:
        """Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform question answering on.

        Returns:
            Dict[str, Any]: Question answering output from SparkNLP LightPipeline.
        """
        if self.model_type == "QA" and isinstance(text, dict) and "context" in text:
            context = text["context"]
            question = text["question"]
            prediction_metadata = self.model.fullAnnotate([context], [question])[0][
                self.output_col
            ]
        elif self.model_type == "TextGen":
            question = text
            prediction_metadata = self.model.fullAnnotate([question])[0][self.output_col]
        else:
            question = text
            context = ""
            prediction_metadata = self.model.fullAnnotate([context], [question])[0][
                self.output_col
            ]
        prediction = prediction_metadata[0].result

        return prediction

    def predict_raw(self, text: str) -> List[str]:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform question answering on.

        Returns:
            List[str]: Predictions as a list of strings.
        """
        prediction_metadata = self.model.fullAnnotate(text)[0][self.output_col]
        prediction = prediction_metadata[0].result
        return prediction


class PretrainedModelForSummarization(PretrainedJSLModel, ModelAPI):
    """Pretrained model for summarization tasks"""

    task = "summarization"

    def __init__(
        self,
        model: Union[
            "NLUPipeline", "PretrainedPipeline", "LightPipeline", "PipelineModel"
        ],
    ):
        """Constructor class

        Args:
            model (LightPipeline):
                Loaded SparkNLP LightPipeline for inference.
        """
        super().__init__(model)

        _summarizer = None
        for annotator in self.model.stages:
            if self.is_summarizer(annotator):
                _summarizer = annotator
                break

        if _summarizer is None:
            raise ValueError(Errors.E040(var="summarizer"))

        self.output_col = "summary"
        self.model = LightPipeline(self.model)

    @staticmethod
    def is_summarizer(model_instance) -> bool:
        """Check summarizer model instance is supported by langtest"""
        for model in SUPPORTED_SPARKNLP_SUMMARIZATION:
            if isinstance(model_instance, model):
                return True
        return False

    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs) -> str:
        """Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform summarization on.

        Returns:
            str: Summarization output from SparkNLP LightPipeline.
        """
        if isinstance(text, dict):
            text = text["context"]
        prediction_metadata = self.model.fullAnnotate(text)[0][self.output_col]
        prediction = prediction_metadata[0].result

        return prediction

    def predict_raw(self, text: str) -> List[str]:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform summarization on.

        Returns:
            List[str]: Predictions as a list of strings.
        """
        if isinstance(text, dict):
            text = text["context"]
        prediction_metadata = self.model.fullAnnotate(text)[0][self.output_col]
        prediction = prediction_metadata[0].result
        return prediction
