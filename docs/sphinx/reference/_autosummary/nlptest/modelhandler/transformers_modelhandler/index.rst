:orphan:

.. INDEX

:py:mod:`nlptest.modelhandler.transformers_modelhandler`
========================================================

.. py:module:: nlptest.modelhandler.transformers_modelhandler


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.modelhandler.transformers_modelhandler.PretrainedModelForNER
   nlptest.modelhandler.transformers_modelhandler.PretrainedModelForTextClassification




.. py:class:: PretrainedModelForNER(model)



   Args:
       model (transformers.pipeline.Pipeline): Pretrained HuggingFace NER pipeline for predictions.

   .. py:method:: load_model(path) -> transformers.Pipeline
      :classmethod:

      Load the NER model into the `model` attribute.


   .. py:method:: predict(text: str, **kwargs) -> nlptest.utils.custom_types.NEROutput

      Perform predictions on the input text.

      Args:
          text (str): Input text to perform NER on.
          kwargs: Additional keyword arguments.

      Keyword Args:
          group_entities (bool): Option to group entities.

      Returns:
          NEROutput: A list of named entities recognized in the input text.


   .. py:method:: predict_raw(text: str) -> List[str]

      Predict a list of labels.
      Args:
          text (str): Input text to perform NER on.
      Returns:
          List[str]: A list of named entities recognized in the input text.



.. py:class:: PretrainedModelForTextClassification(model)



   Attributes:
       model (transformers.pipeline.Pipeline):
           Loaded Text Classification pipeline for predictions.

   .. py:method:: load_model(path) -> None
      :classmethod:

      Load and return text classification transformers pipeline


   .. py:method:: predict(text: str, return_all_scores: bool = False, *args, **kwargs) -> nlptest.utils.custom_types.SequenceClassificationOutput

      Perform predictions on the input text.

      Args:
          text (str): Input text to perform NER on.
          return_all_scores (bool): Option to group entities.
          kwargs: Additional keyword arguments.

      Returns:
          SequenceClassificationOutput: text classification from the input text.


   .. py:method:: predict_raw(text: str) -> List[str]

      Perform predictions on the input text.

      Args:
          text (str): Input text to perform NER on.


      Returns:
          List[str]: Predictions as a list of strings.



