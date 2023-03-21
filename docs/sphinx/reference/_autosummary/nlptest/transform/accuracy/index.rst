:orphan:

.. INDEX

:py:mod:`nlptest.transform.accuracy`
====================================

.. py:module:: nlptest.transform.accuracy


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.transform.accuracy.BaseAccuracy
   nlptest.transform.accuracy.MinPrecisionScore
   nlptest.transform.accuracy.MinRecallScore
   nlptest.transform.accuracy.MinF1Score
   nlptest.transform.accuracy.MinMicroF1Score
   nlptest.transform.accuracy.MinMacroF1Score
   nlptest.transform.accuracy.MinWeightedF1Score




.. py:class:: BaseAccuracy



   Abstract base class for implementing accuracy measures.

   Attributes:
       alias_name (str): A name or list of names that identify the accuracy measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented accuracy measure.

   .. py:method:: transform(y_true, y_pred)
      :staticmethod:
      :abstractmethod:

      Abstract method that implements the accuracy measure.

      Args:
          y_true: True values
          y_pred: Predicted values
          model (ModelFactory): Model to be evaluted.

      Returns:
          Any: The transformed data based on the implemented accuracy measure.



.. py:class:: MinPrecisionScore



   Subclass of BaseAccuracy that implements the minimum precision score.

   Attributes:
       alias_name (str): The name "min_precision_score" for config.

   Methods:
       transform(y_true, y_pred) -> Any: Creates accuracy test results.

   .. py:method:: transform(y_true, y_pred, params)
      :staticmethod:

      Computes the minimum F1 score for the given data.

      Args:
          y_true: True values
          y_pred: Predicted values

      Returns:
          List[Sample]: Precision test results.



.. py:class:: MinRecallScore



   Subclass of BaseAccuracy that implements the minimum precision score.

   Attributes:
       alias_name (str): The name "min_precision_score" for config.

   Methods:
       transform(y_true, y_pred) -> Any: Creates accuracy test results.

   .. py:method:: transform(y_true, y_pred, params)
      :staticmethod:

      Computes the minimum recall score for the given data.

      Args:
          y_true: True values
          y_pred: Predicted values

      Returns:
          List[Sample]: Precision recall results.



.. py:class:: MinF1Score



   Subclass of BaseAccuracy that implements the minimum precision score.

   Attributes:
       alias_name (str): The name "min_precision_score" for config.

   Methods:
       transform(y_true, y_pred) -> Any: Creates accuracy test results.

   .. py:method:: transform(y_true, y_pred, params)
      :staticmethod:

      Computes the minimum F1 score for the given data.

      Args:
          y_true: True values
          y_pred: Predicted values

      Returns:
          List[Sample]: F1 score test results.



.. py:class:: MinMicroF1Score



   Subclass of BaseAccuracy that implements the minimum precision score.

   Attributes:
       alias_name (str): The name for config.

   Methods:
       transform(y_true, y_pred) -> Any: Creates accuracy test results.

   .. py:method:: transform(y_true, y_pred, params)
      :staticmethod:

      Computes the minimum F1 score for the given data.

      Args:
          y_true: True values
          y_pred: Predicted values

      Returns:
          Any: The transformed data based on the minimum F1 score.



.. py:class:: MinMacroF1Score



   Subclass of BaseAccuracy that implements the minimum precision score.

   Attributes:
       alias_name (str): The name "min_precision_score" for config.

   Methods:
       transform(y_true, y_pred) -> Any: Creates accuracy test results.

   .. py:method:: transform(y_true, y_pred, params)
      :staticmethod:

      Computes the minimum F1 score for the given data.

      Args:
          y_true: True values
          y_pred: Predicted values

      Returns:
          Any: The transformed data based on the minimum F1 score.



.. py:class:: MinWeightedF1Score



   Subclass of BaseAccuracy that implements the minimum weighted f1 score.

   Attributes:
       alias_name (str): The name for config.

   Methods:
       transform(y_true, y_pred) -> Any: Creates accuracy test results.

   .. py:method:: transform(y_true, y_pred, params)
      :staticmethod:

      Computes the minimum weighted F1 score for the given data.

      Args:
          y_true: True values
          y_pred: Predicted values   

      Returns:
          Any: The transformed data based on the minimum F1 score.



