:orphan:

.. INDEX

:py:mod:`nlptest.nlptest`
=========================

.. py:module:: nlptest.nlptest


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.nlptest.Harness




.. py:class:: Harness(task: Optional[str], model: Union[str], hub: Optional[str] = None, data: Optional[str] = None, config: Optional[Union[str, dict]] = None)

   Harness is a testing class for NLP models.

   Harness class evaluates the performance of a given NLP model. Given test data is
   used to test the model. A report is generated with test results.

   .. py:method:: configure(config: Union[str, dict])

      Configure the Harness with a given configuration.

      Args:
          config (str | dict): Configuration file path or dictionary
              for the tests to be performed.

      Returns:
          dict: Loaded configuration.


   .. py:method:: generate() -> Harness

      Generates the testcases to be used when evaluating the model.

      Returns:
          None: The generated testcases are stored in `_testcases` attribute.


   .. py:method:: run() -> Harness

      Run the tests on the model using the generated testcases.

      Returns:
          None: The evaluations are stored in `generated_results` attribute.


   .. py:method:: report() -> pandas.DataFrame

      Generate a report of the test results.
      Returns:
          pd.DataFrame: DataFrame containing the results of the tests.


   .. py:method:: generated_results() -> pandas.DataFrame

      Generates an overall report with every textcase and labelwise metrics.

      Returns:
          pd.DataFrame: Generated dataframe.


   .. py:method:: augment(input_path, output_path, inplace=False)

      Augments the data in the input file located at `input_path` and saves the result to `output_path`.

      Args:
          input_path (str): Path to the input file.
          output_path (str): Path to save the augmented data.
          inplace (bool, optional): Whether to modify the input file directly. Defaults to False.

      Returns:
          Harness: The instance of the class calling this method.

      Raises:
          ValueError: If the `pass_rate` or `minimum_pass_rate` columns have an unexpected data type.

      Note:
          This method uses an instance of `AugmentRobustness` to perform the augmentation.

      Example:
          >>> harness = Harness(...)
          >>> harness.augment("train.conll", "augmented_train.conll")


   .. py:method:: testcases() -> pandas.DataFrame

      Testcases after .generate() is called


   .. py:method:: save(save_dir: str) -> None

      Save the configuration, generated testcases and the `DataFactory` to be reused later.

      Args:
          save_dir (str): path to folder to save the different files
      Returns:



   .. py:method:: save_testcases(path_to_file: str) -> None

      Save the generated testcases into a pickle file.

      Args:
          path_to_file (str):
              location to save the pickle file to
      Returns:



   .. py:method:: load(save_dir: str, task: str, model: Union[str, nlptest.modelhandler.ModelFactory], hub: str = None) -> Harness
      :classmethod:

      Loads a previously saved `Harness` from a given configuration and dataset

      Args:
          save_dir (str):
              path to folder containing all the needed files to load an saved `Harness`
          task (str):
              task for which the model is to be evaluated.
          model (str | ModelFactory):
              ModelFactory object or path to the model to be evaluated.
          hub (str, optional):
              model hub to load from the path. Required if path is passed as 'model'.
      Returns:
          Harness:
              `Harness` loaded from from a previous configuration along with the new model to evaluate


   .. py:method:: load_testcases(path_to_file: str) -> None

      Loads the testcases from a pickle file

      Args:
          path_to_file (str):
              location to load the test cases from
      Returns:




