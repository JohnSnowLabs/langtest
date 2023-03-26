:orphan:

.. INDEX

:py:mod:`nlptest.datahandler.datasource`
========================================

.. py:module:: nlptest.datahandler.datasource


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.datahandler.datasource.DataFactory
   nlptest.datahandler.datasource.ConllDataset
   nlptest.datahandler.datasource.JSONDataset
   nlptest.datahandler.datasource.CSVDataset




.. py:class:: DataFactory(file_path: str, task: str)

   Data factory for creating Dataset objects.

   The DataFactory class is responsible for creating instances of the
   correct Dataset type based on the file extension.

   .. py:method:: load()

      Loads the data for the correct Dataset type.

      Returns:
          list[str]: Loaded text data.



.. py:class:: ConllDataset(file_path: str, task: str)



   Class to handle Conll files. Subclass of _IDataset.
       

   .. py:method:: load_data() -> List[nlptest.utils.custom_types.Sample]

      Loads data from a CoNLL file.

      Returns:
          list: List of sentences in the dataset.



.. py:class:: JSONDataset(file_path)



   Class to handle JSON dataset files. Subclass of _IDataset.
       

   .. py:method:: load_data()

      Load data from the file_path.



.. py:class:: CSVDataset(file_path: str, task: str)



   Class to handle CSV files dataset. Subclass of _IDataset.
       

   .. py:method:: load_data() -> List[nlptest.utils.custom_types.Sample]

      Loads data from a csv file.

      Returns:




