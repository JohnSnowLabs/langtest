:orphan:

.. INDEX

:py:mod:`nlptest.modelhandler.modelhandler`
===========================================

.. py:module:: nlptest.modelhandler.modelhandler


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.modelhandler.modelhandler.ModelFactory




.. py:class:: ModelFactory(model, task: str)

   A factory class for instantiating models.

   .. py:method:: load_model(task: str, hub: str, path: str) -> ModelFactory
      :classmethod:

      Load the model.

      Args:
          path (str): path to model to use
          task (str): task to perform
          hub (str): model hub to load custom model from the path, either to hub or local disk.


   .. py:method:: predict(text: str, **kwargs) -> Union[nlptest.utils.custom_types.NEROutput, nlptest.utils.custom_types.SequenceClassificationOutput]

      Perform predictions on input text.

      Args:
          text (str): Input text to perform predictions on.

      Returns:
          NEROutput or SequenceClassificationOutput


   .. py:method:: predict_raw(text) -> List[str]

      Perform predictions on input text.

      Args:
          text (str): Input text to perform predictions on.

      Returns:
          List[str]: Predictions.



