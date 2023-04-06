import os
from typing import List, Union, Dict, Tuple
import numpy as np

from .modelhandler import _ModelHandler
from ..utils.custom_types import NEROutput, NERPrediction, SequenceClassificationOutput
from ..utils.lib_manager import try_import_lib

if try_import_lib('pyspark'):
    from pyspark.ml import PipelineModel

if try_import_lib('johnsnowlabs'):
    from johnsnowlabs import nlp
    from nlu import NLUPipeline

SUPPORTED_SPARKNLP_NER_MODELS = []
SUPPORTED_SPARKNLP_CLASSIFERS = []
if try_import_lib("sparknlp"):
    from sparknlp.annotator import *
    from sparknlp.base import LightPipeline
    from sparknlp.pretrained import PretrainedPipeline

    SUPPORTED_SPARKNLP_NER_MODELS.extend([
        AlbertForTokenClassification,
        BertForTokenClassification,
        CamemBertForTokenClassification,
        DeBertaForTokenClassification,
        DistilBertForTokenClassification,
        LongformerForTokenClassification,
        RoBertaForTokenClassification,
        XlmRoBertaForTokenClassification,
        XlnetForTokenClassification,
        NerDLModel
    ])

    SUPPORTED_SPARKNLP_CLASSIFERS.extend([
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
    ])

if try_import_lib("sparknlp_jsl"):
    from sparknlp_jsl.legal import (LegalBertForTokenClassification, LegalNerModel,
                                    LegalBertForSequenceClassification, LegalClassifierDLModel)

    from sparknlp_jsl.finance import (FinanceBertForTokenClassification, FinanceNerModel,
                                      FinanceBertForSequenceClassification, FinanceClassifierDLModel)

    from sparknlp_jsl.annotator import (MedicalBertForTokenClassifier, MedicalNerModel,
                                        MedicalBertForSequenceClassification,
                                        MedicalDistilBertForSequenceClassification)

    SUPPORTED_SPARKNLP_NER_MODELS.extend([
        LegalBertForTokenClassification, LegalNerModel,
        FinanceBertForTokenClassification, FinanceNerModel,
        MedicalBertForTokenClassifier, MedicalNerModel
    ])

    SUPPORTED_SPARKNLP_CLASSIFERS.extend([
        LegalBertForSequenceClassification, LegalClassifierDLModel,
        FinanceBertForSequenceClassification, FinanceClassifierDLModel,
        MedicalBertForSequenceClassification, MedicalDistilBertForSequenceClassification
    ])


class PretrainedModelForNER(_ModelHandler):
    """"""

    def __init__(
            self,
            model: Union['NLUPipeline', 'PretrainedPipeline', 'LightPipeline', 'PipelineModel']
    ):
        """
        Attributes:
            model (LightPipeline):
                Loaded SparkNLP LightPipeline for inference.
        """

        if model.__class__.__name__ == 'PipelineModel':
            model = model

        elif model.__class__.__name__ == 'LightPipeline':
            model = model.pipeline_model

        elif model.__class__.__name__ == 'PretrainedPipeline':
            model = model.model

        elif model.__class__.__name__ == 'NLUPipeline':
            stages = [comp.model for comp in model.components]
            _pipeline = nlp.Pipeline().setStages(stages)
            tmp_df = model.spark.createDataFrame([['']]).toDF('text')
            model = _pipeline.fit(tmp_df)

        else:
            raise ValueError(f'Invalid SparkNLP model object: {type(model)}. '
                             f'John Snow Labs model handler accepts: '
                             f'[NLUPipeline, PretrainedPipeline, PipelineModel, LightPipeline]')

        #   there can be multiple ner model in the pipeline
        #   but at first I will set first as default one. Later we can adjust Harness to test multiple model
        ner_model = None
        for annotator in model.stages:
            if self.is_ner_annotator(annotator):
                ner_model = annotator
                break

        if ner_model is None:
            raise ValueError('Invalid PipelineModel! There should be at least one NER component.')

        self.output_col = ner_model.getOutputCol()

        #   in order to overwrite configs, light pipeline should be reinitialized.
        self.model = LightPipeline(model)
   
    @staticmethod
    def _aggregate_words(prediction: List[Dict]) -> List[Dict]:
        """
        Aggregates predictions at a word-level by taking the first token label.

        Args:
            predictions (List[Dict]):
                predictions obtained with the pipeline object
        Returns:
            List[Dict]:
                aggregated predictions
        """
        aggregated_words = []
        for i in range(0,len(prediction)):
            aggregated_words.append(
                {
                    'entity': prediction[i].result,
                    'index':i+1,
                    'word' : prediction[i].metadata['word'],
                    'start': prediction[i].begin,
                    'end' :  (prediction[i].end)+1
                    
                }
            )
            
        return aggregated_words
    
    @staticmethod
    def _get_tag(entity_label: str) -> Tuple[str, str]:
        """"
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
        """
        Group together the adjacent tokens with the same entity predicted.
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
        """
        Find and group together the adjacent tokens with the same entity predicted.
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
    def load_model(cls, path: str) -> 'NLUPipeline':
        """
        Load the NER model into the `model` attribute.
        Args:
            path (str): Path to pretrained local or NLP Models Hub SparkNLP model
        """
        if os.path.exists(path):
            if try_import_lib('johnsnowlabs'):
                loaded_model = nlp.load(path=path)
            else:
                loaded_model = PipelineModel.load(path)
        else:
            if try_import_lib('johnsnowlabs'):
                loaded_model = nlp.load(path)
            else:
                raise ValueError(f'johnsnowlabs is not installed. '
                                 f'In order to use NLP Models Hub, johnsnowlabs should be installed!')

        return loaded_model

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
                    entity=ent['entity_group'],
                    word=ent['word'],
                    start=ent['start'],
                    end=ent['end'],
                ) for ent in aggregated_predictions
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

    def __call__(self, text: str) -> NEROutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text)

    @staticmethod
    def is_ner_annotator(model_instance) -> bool:
        """Check ner model instance is supported by nlptest"""
        for model in SUPPORTED_SPARKNLP_NER_MODELS:
            if isinstance(model_instance, model):
                return True
        return False


class PretrainedModelForTextClassification(_ModelHandler):
    """"""

    def __init__(
            self,
            model: Union['NLUPipeline', 'PretrainedPipeline', 'LightPipeline', 'PipelineModel']
    ):
        """
        Attributes:
            model (LightPipeline):
                Loaded SparkNLP LightPipeline for inference.
        """

        if model.__class__.__name__ == 'PipelineModel':
            model = model

        elif model.__class__.__name__ == 'LightPipeline':
            model = model.pipeline_model

        elif model.__class__.__name__ == 'PretrainedPipeline':
            model = model.model

        elif model.__class__.__name__ == 'NLUPipeline':
            stages = [comp.model for comp in model.components]
            _pipeline = nlp.Pipeline().setStages(stages)
            tmp_df = model.spark.createDataFrame([['']]).toDF('text')
            model = _pipeline.fit(tmp_df)

        else:
            raise ValueError(f'Invalid SparkNLP model object: {type(model)}. '
                             f'John Snow Labs model handler accepts: '
                             f'[NLUPipeline, PretrainedPipeline, PipelineModel, LightPipeline]')

        _classifier = None
        for annotator in model.stages:
            if self.is_classifier(annotator):
                _classifier = annotator
                break

        if _classifier is None:
            raise ValueError('Invalid PipelineModel! There should be at least one classifier component.')

        self.output_col = _classifier.getOutputCol()
        self.classes = _classifier.getClasses()
        self.model = LightPipeline(model)

    @staticmethod
    def is_classifier(model_instance) -> bool:
        """Check classifier model instance is supported by nlptest"""
        for model in SUPPORTED_SPARKNLP_CLASSIFERS:
            if isinstance(model_instance, model):
                return True
        return False

    @classmethod
    def load_model(cls, path) -> 'NLUPipeline':
        """
        Load the NER model into the `model` attribute.

        Args:
            path (str): Path to pretrained local or NLP Models Hub SparkNLP model
        """
        if os.path.exists(path):
            if try_import_lib('johnsnowlabs'):
                loaded_model = nlp.load(path=path)
            else:
                loaded_model = PipelineModel.load(path)
        else:
            if try_import_lib('johnsnowlabs'):
                loaded_model = nlp.load(path)
            else:
                raise ValueError(f'johnsnowlabs is not installed. '
                                 f'In order to use NLP Models Hub, johnsnowlabs should be installed!')

        return loaded_model

    def predict(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """
        Perform predictions with SparkNLP LightPipeline on the input text.

        Args:
            text (str): Input text to perform NER on.
            return_all_scores (bool): Option to return score for all labels.

        Returns:
            SequenceClassificationOutput: Classification output from SparkNLP LightPipeline.
        """
        prediction_metadata = self.model.fullAnnotate(text)[0][self.output_col][0].metadata
        prediction = [{"label": x, "score": y} for x, y in prediction_metadata.items()]

        if not return_all_scores:
            prediction = [max(prediction, key=lambda x: x['score'])]

        return SequenceClassificationOutput(
            text=text,
            predictions=prediction
        )

    def predict_raw(self, text: str) -> List[str]:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform text classification on.

        Returns:
            List[str]: Predictions as a list of strings.
        """
        prediction_metadata = self.model.fullAnnotate(text)[0][self.output_col][0].metadata
        prediction = [{"label": x, "score": y} for x, y in prediction_metadata.items()]
        prediction = [max(prediction, key=lambda x: x['score'])]
        return [x["label"] for x in prediction]

    def __call__(self, text: str) -> SequenceClassificationOutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text)
