:orphan:

.. INDEX

:py:mod:`nlptest.datahandler.format`
====================================

.. py:module:: nlptest.datahandler.format


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.datahandler.format.BaseFormatter
   nlptest.datahandler.format.Formatter
   nlptest.datahandler.format.SequenceClassificationOutputFormatter
   nlptest.datahandler.format.NEROutputFormatter




.. py:class:: BaseFormatter



   Abstract base class for defining formatter classes.
   Subclasses should implement the static methods `to_csv` and `to_conll`.

   .. py:method:: to_csv(custom_type)
      :staticmethod:
      :abstractmethod:

      Converts a custom type to a CSV string.

      Args:
          custom_type: The custom type to convert.

      Returns:
          The CSV string representation of the custom type.

      Raises:
          NotImplementedError: This method should be implemented by the subclass.


   .. py:method:: to_conll(custom_type)
      :staticmethod:
      :abstractmethod:

      Converts a custom type to a CoNLL string.

      Args:
          custom_type: The custom type to convert.

      Returns:
          The CoNLL string representation of the custom type.

      Raises:
          NotImplementedError: This method should be implemented by the subclass.



.. py:class:: Formatter

   Formatter class for converting between custom types and different output formats.

   This class uses the `to_csv` and `to_conll` methods of subclasses of `BaseFormatter`
   to perform the conversions. The appropriate subclass is selected based on the
   type of the expected results in the `sample` argument.

   Args:
       sample: The input sample to convert.
       format: The output format to convert to, either "csv" or "conll".
       *args: Optional positional arguments to pass to the `to_csv` or `to_conll` methods.
       **kwargs: Optional keyword arguments to pass to the `to_csv` or `to_conll` methods.

   Returns:
       The output string in the specified format.

   Raises:
       NameError: If no formatter subclass is defined for the type of the expected results in the sample.


.. py:class:: SequenceClassificationOutputFormatter



   Formatter class for converting `SequenceClassificationOutput` objects to CSV.

   The `to_csv` method returns a CSV string representing the `SequenceClassificationOutput`
   object in the sample argument.

   Args:
       sample: The input sample containing the `SequenceClassificationOutput` object to convert.
       delimiter: The delimiter character to use in the CSV string.

   Returns:
       The CSV string representation of the `SequenceClassificationOutput` object.

   Raises:
       None.

   .. py:method:: to_csv(delimiter=',')

      Converts a custom type to a CSV string.

      Args:
          custom_type: The custom type to convert.

      Returns:
          The CSV string representation of the custom type.

      Raises:
          NotImplementedError: This method should be implemented by the subclass.



.. py:class:: NEROutputFormatter



   Formatter class for converting `NEROutput` objects to CSV and CoNLL.

   The `to_csv` method returns a CSV string representing the `NEROutput` object in the sample
   argument. The `to_conll` method returns a CoNLL string representing the `NEROutput` object.

   Args:
       sample: The input sample containing the `NEROutput` object to convert.
       delimiter: The delimiter character to use in the CSV string.
       temp_id: A temporary ID to use for grouping entities by document.

   Returns:
       The CSV or CoNLL string representation of the `NEROutput` object.

   Raises:
       None.

   .. py:method:: to_csv(delimiter=',', temp_id=None)

      Converts a custom type to a CSV string.

      Args:
          custom_type: The custom type to convert.

      Returns:
          The CSV string representation of the custom type.

      Raises:
          NotImplementedError: This method should be implemented by the subclass.


   .. py:method:: to_conll(temp_id=None)

      Converts a custom type to a CoNLL string.

      Args:
          custom_type: The custom type to convert.

      Returns:
          The CoNLL string representation of the custom type.

      Raises:
          NotImplementedError: This method should be implemented by the subclass.



