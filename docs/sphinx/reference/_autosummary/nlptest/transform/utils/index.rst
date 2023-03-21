:orphan:

.. INDEX

:py:mod:`nlptest.transform.utils`
=================================

.. py:module:: nlptest.transform.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   nlptest.transform.utils.get_substitution_names
   nlptest.transform.utils.create_terminology



.. py:function:: get_substitution_names(values_list)

   Helper function to get list of substitution names 

   Args:
        values_list : list of substitution lists.

   Returns:
        List of substitution names


.. py:function:: create_terminology(ner_data: pandas.DataFrame) -> Dict[str, List[str]]

   Iterate over the DataFrame to create terminology from the predictions. IOB format converted to the IO.

   Args:
       ner_data: Pandas DataFrame that has 2 column, 'text' as string and 'label' as list of labels

   Returns:
       Dictionary of entities and corresponding list of words.


