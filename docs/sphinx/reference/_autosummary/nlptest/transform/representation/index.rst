:orphan:

.. INDEX

:py:mod:`nlptest.transform.representation`
==========================================

.. py:module:: nlptest.transform.representation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.transform.representation.BaseRepresentation
   nlptest.transform.representation.GenderReprestation




.. py:class:: BaseRepresentation



   Abstract base class for implementing representation measures.

   Attributes:
       alias_name (str): A name or list of names that identify the representation measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented representation measure.

   .. py:method:: transform(self)
      :staticmethod:
      :abstractmethod:

      Abstract method that implements the representation measure.

      Args:
          data (List[Sample]): The input data to be transformed.

      Returns:
          Any: The transformed data based on the implemented representation measure.



.. py:class:: GenderReprestation



   Abstract base class for implementing representation measures.

   Attributes:
       alias_name (str): A name or list of names that identify the representation measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented representation measure.

   .. py:method:: transform()

      Abstract method that implements the representation measure.

      Args:
          data (List[Sample]): The input data to be transformed.

      Returns:
          Any: The transformed data based on the implemented representation measure.



