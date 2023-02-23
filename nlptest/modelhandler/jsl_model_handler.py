from typing import List, Union

from .modelhandler import _ModelHandler
from ..utils.lib_manager import try_import_lib
from ..utils.custom_types import NEROutput, SequenceClassificationOutput

if try_import_lib('pyspark'):
    from pyspark.ml import PipelineModel

if try_import_lib('johnsnowlabs'):
    from johnsnowlabs import nlp
    from johnsnowlabs.nlp.nlu import NLUPipeline

SUPPORTED_SPARKNLP_NER_MODELS = []
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

if try_import_lib("sparknlp_jsl"):

    from sparknlp_jsl.legal import LegalBertForTokenClassification, LegalNerModel
    from sparknlp_jsl.finance import FinanceBertForTokenClassification, FinanceNerModel
    from sparknlp_jsl.annotator import MedicalBertForTokenClassifier, MedicalNerModel

    SUPPORTED_SPARKNLP_NER_MODELS.extend([
        LegalBertForTokenClassification, LegalNerModel,
        FinanceBertForTokenClassification, FinanceNerModel,
        MedicalBertForTokenClassifier, MedicalNerModel
    ])


class NERJohnSnowLabsPretrainedModel(_ModelHandler):

    def __init__(
            self,
            model: Union[NLUPipeline, PretrainedPipeline, LightPipeline, PipelineModel]
    ):
        """
        Attributes:
            model (LightPipeline):
                Loaded SparkNLP Light Pipeline for inference.
        """

        if isinstance(model, PipelineModel):
            model = model

        elif isinstance(model, LightPipeline):
            model = model.pipeline_model

        elif isinstance(model, PretrainedPipeline):
            model = model.model

        elif isinstance(model, NLUPipeline):
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
            if self.is_instance_supported(annotator):
                ner_model = annotator
                break

        if ner_model is None:
            raise ValueError('Invalid PipelineModel! There should be at least one NER component.')

        #    this line is to set pipeline to add confidence score in predictions
        #    even though they are useful information, not used yet.
        ner_model.setIncludeConfidence(True)
        ner_model.setIncludeAllConfidenceScores(True)

        self.output_col = ner_model.getOutputCol()

        #   in order to overwrite configs, light pipeline should be reinitialized.
        self.model = LightPipeline(model)

    @classmethod
    def load_model(cls, path) -> 'NERJohnSnowLabsPretrainedModel':
        """Load the NER model into the `model` attribute.
        Args:
            path (str): Load PipelineModel from given path.
        """
        if try_import_lib('johnsnowlabs'):
            loaded_model = nlp.load(path)
        else:
            loaded_model = PipelineModel.load(path)

        return cls(
            model=loaded_model
        )

    def predict(self, text: str) -> List[NEROutput]:
        """Perform predictions with SparkNLP LightPipeline on the input text.
        Args:
            text (str): Input text to perform NER on.
        Returns:
            NEROutput: A list of named entities recognized in the input text.
        """
        prediction = self.model.fullAnnotate(text)[0][self.output_col]
        return [NEROutput(
            entity=pred.result,
            word=pred.metadata['word'],
            start=pred.begin,
            end=pred.end)
            for pred in prediction]

    def __call__(self, text: str) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.predict(text=text)

    #   helpers
    @staticmethod
    def is_instance_supported(model_instance) -> bool:
        """Check ner model instance is supported by nlptest"""
        for model in SUPPORTED_SPARKNLP_NER_MODELS:
            if isinstance(model_instance, model):
                return True
        return False


class TextClassificationJohnSnowLabsPretrainedModel(_ModelHandler):

    def __init__(
            self,
            model: NLUPipeline
    ):

        """
        Attributes:
            model (LightPipeline):
                Loaded SparkNLP Light Pipeline for inference.
        """

        if isinstance(model, PretrainedPipeline):
            model = model.model

        elif isinstance(model, LightPipeline):
            model = model.pipeline_model

        elif isinstance(model, NLUPipeline):
            stages = [comp.model for comp in model.components]
            _pipeline = nlp.Pipeline().setStages(stages)
            tmp_df = model.spark.createDataFrame([['']]).toDF('text')
            model = _pipeline.fit(tmp_df)
        else:
            raise ValueError('Invalid model for JSL.')

        #   there can be multiple ner model in the pipeline
        #   but at first I will set first as default one. Later we can adjust Harness to test multiple model
        classifier_model = None
        for annotator in model.stages:
            pass

        if classifier_model is None:
            raise ValueError('Invalid PipelineModel! There should be at least one ClassifierDL component.')

        self.output_col = classifier_model.getOutputCol()

        #   in order to overwrite configs, light pipeline should be reinitialzied.
        self.model = LightPipeline(model)

    @classmethod
    def load_model(cls, path) -> 'TextClassificationJohnSnowLabsPretrainedModel':
        """Load the ClassifierDL Pipeline into the `model` attribute.
        Args:
            path (str): Load PipelineModel from given path.
        """
        nlu_pipeline = nlp.load(path)
        return cls(
            model=nlu_pipeline
        )

    def predict(self, text: str) -> SequenceClassificationOutput:
        """Perform predictions with SparkNLP LightPipeline on the input text.
        Args:
            text (str): Input text to perform Text Classification on.
        Returns:
            NEROutput: A list of named entities recognized in the input text.
        """
        prediction = self.model.fullAnnotate(text)[0][self.output_col]
        return SequenceClassificationOutput(
            text=text,
            labels=prediction[0]['class'][0].result
        )

    def __call__(self, text: str) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.predict(text=text)