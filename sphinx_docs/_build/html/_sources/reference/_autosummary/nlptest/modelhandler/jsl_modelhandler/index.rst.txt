:orphan:

.. INDEX

:py:mod:`nlptest.modelhandler.jsl_modelhandler`
===============================================

.. py:module:: nlptest.modelhandler.jsl_modelhandler


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.modelhandler.jsl_modelhandler.PretrainedModelForNER
   nlptest.modelhandler.jsl_modelhandler.PretrainedModelForTextClassification




.. py:class:: PretrainedModelForNER(model: Union[nlu.NLUPipeline, sparknlp.pretrained.PretrainedPipeline, sparknlp.base.LightPipeline, pyspark.ml.PipelineModel])



   Abstract base class for handling different models.

   Implementations should inherit from this class and override load_model() and predict() methods.

   .. py:method:: load_model(path) -> nlu.NLUPipeline
      :classmethod:

      Load the NER model into the `model` attribute.
      Args:
          path (str): Path to pretrained local or NLP Models Hub SparkNLP model


   .. py:method:: predict(text: str, *args, **kwargs) -> nlptest.utils.custom_types.NEROutput

      Perform predictions with SparkNLP LightPipeline on the input text.
      Args:
          text (str): Input text to perform NER on.
      Returns:
          NEROutput: A list of named entities recognized in the input text.


   .. py:method:: predict_raw(text: str) -> List[str]

      Perform predictions with SparkNLP LightPipeline on the input text.
      Args:
          text (str): Input text to perform NER on.
      Returns:
          List[str]: Predicted labels.


   .. py:method:: is_ner_annotator(model_instance) -> bool
      :staticmethod:

      Check ner model instance is supported by nlptest



.. py:class:: PretrainedModelForTextClassification(model: Union[nlu.NLUPipeline, sparknlp.pretrained.PretrainedPipeline, sparknlp.base.LightPipeline, pyspark.ml.PipelineModel])



   Abstract base class for handling different models.

   Implementations should inherit from this class and override load_model() and predict() methods.

   .. py:method:: load_model(path) -> nlu.NLUPipeline

      Load the NER model into the `model` attribute.
      Args:
          path (str): Path to pretrained local or NLP Models Hub SparkNLP model


   .. py:method:: predict(text: str, return_all_scores: bool = False, *args, **kwargs) -> nlptest.utils.custom_types.SequenceClassificationOutput

      Perform predictions with SparkNLP LightPipeline on the input text.
      Args:
          text (str): Input text to perform NER on.
          return_all_scores (bool): Option to return score for all labels.

      Returns:
          SequenceClassificationOutput: Classification output from SparkNLP LightPipeline.


   .. py:method:: is_classifier(model_instance) -> bool
      :staticmethod:

      Check classifier model instance is supported by nlptest



