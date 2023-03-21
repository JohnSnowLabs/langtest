:orphan:

.. INDEX

:py:mod:`nlptest.augmentation.fix_robustness`
=============================================

.. py:module:: nlptest.augmentation.fix_robustness


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.augmentation.fix_robustness.BaseAugmentaion
   nlptest.augmentation.fix_robustness.AugmentRobustness




.. py:class:: BaseAugmentaion



   Abstract base class for data augmentation techniques.

   Attributes:
       None

   Methods:
       fix: Abstract method that should be implemented by child classes.
            This method should perform the data augmentation operation.

            Returns:
                NotImplementedError: Raised if the method is not implemented by child classes.

   .. py:method:: fix()
      :abstractmethod:

      Abstract method that should be implemented by child classes.
      This method should perform the data augmentation operation.

      Returns:
          NotImplementedError: Raised if the method is not implemented by child classes.



.. py:class:: AugmentRobustness(task, h_report, config, max_prop=0.5)



   A class for performing a specified task with historical results.

   Attributes:

       task (str): A string indicating the task being performed.
       config (dict): A dictionary containing configuration parameters for the task.
       h_report (pandas.DataFrame): A DataFrame containing a report of historical results for the task.
       max_prop (float): The maximum proportion of improvement that can be suggested by the class methods.
                       Defaults to 0.5.

   Methods:

       __init__(self, task, h_report, config, max_prop=0.5) -> None:
           Initializes an instance of MyClass with the specified parameters.

       fix(self) -> List[Sample]:
           .

       suggestions(self, prop) -> pandas.DataFrame:
           Calculates suggestions for improving test performance based on a given report.

       



   .. py:method:: fix(input_path: str, output_path, inplace: bool = False)

      Applies perturbations to the input data based on the recommendations from harness reports.

      Args:
          input_path (str): The path to the input data file.
          output_path (str): The path to save the augmented data file.
          inplace (bool, optional): If True, the list of samples is modified in place.
                                    Otherwise, a new samples are add to input data. Defaults to False.

      Returns:
          List[Dict[str, Any]]: A list of augmented data samples.


   .. py:method:: suggestions(report)

      Calculates suggestions for improving test performance based on a given report.

      Args:
          report (pandas.DataFrame): A DataFrame containing test results by category and test type,
                                      including pass rates and minimum pass rates.

      Returns:
          pandas.DataFrame: A DataFrame containing the following columns for each suggestion:
                              - category: the test category
                              - test_type: the type of test
                              - ratio: the pass rate divided by the minimum pass rate for the test
                              - proportion_increase: a proportion indicating how much the pass rate
                                                  should increase to reach the minimum pass rate




