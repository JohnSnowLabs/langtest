:orphan:

.. INDEX

:py:mod:`nlptest.modelhandler.spacy_modelhandler`
=================================================

.. py:module:: nlptest.modelhandler.spacy_modelhandler


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.modelhandler.spacy_modelhandler.PretrainedModelForNER
   nlptest.modelhandler.spacy_modelhandler.PretrainedModelForTextClassification




.. py:class:: PretrainedModelForNER(model)



   Args:
       model: Pretrained SpaCy pipeline.

   .. py:method:: load_model(path)
      :classmethod:

      Load and return SpaCy pipeline


   .. py:method:: predict(text: str, *args, **kwargs) -> nlptest.utils.custom_types.NEROutput

      Perform predictions on the input text.

      Args:
          text (str): Input text to perform NER on.
          kwargs: Additional keyword arguments.

      Keyword Args:
          group_entities (bool): Option to group entities.

      Returns:
          NEROutput: A list of named entities recognized in the input text.


   .. py:method:: predict_raw(text: str) -> List[str]

      Predict a list of labels in form of strings.

      Args:
          text (str): Input text to perform NER on.

      Returns:
          List[str]: A list of named entities recognized in the input text.



.. py:class:: PretrainedModelForTextClassification(model)



   Args:
       model: Pretrained SpaCy pipeline.

   .. py:method:: load_model(path)

      Load and return SpaCy pipeline


   .. py:method:: predict(text: str, return_all_scores: bool = False, *args, **kwargs) -> nlptest.utils.custom_types.SequenceClassificationOutput

      Perform text classification predictions on the input text.

      Args:
          text (str): Input text to classify.
          return_all_scores (bool): Option to return score for all labels.

      Returns:
          SequenceClassificationOutput: Text classification predictions from the input text.


   .. py:method:: predict_raw(text: str) -> List[str]

      Perform classification predictions on input text.

      Args:
          text (str): Input text to classify.

      Returns:
          List[str]: Predictions of the model.



