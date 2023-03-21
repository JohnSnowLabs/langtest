:orphan:

.. INDEX

:py:mod:`nlptest.testrunner`
============================

.. py:module:: nlptest.testrunner


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.testrunner.TestRunner
   nlptest.testrunner.RobustnessTestRunner




.. py:class:: TestRunner(load_testcases: List[nlptest.utils.custom_types.Sample], model_handler: nlptest.modelhandler.ModelFactory, data: List[nlptest.utils.custom_types.Sample])

   Base class for running tests on models.

   .. py:method:: evaluate() -> Tuple[List[nlptest.utils.custom_types.Sample], pandas.DataFrame]

      Abstract method to evaluate the testcases.

      Returns:
          Tuple[List[Sample], pd.DataFrame]



.. py:class:: RobustnessTestRunner(load_testcases: List[nlptest.utils.custom_types.Sample], model_handler: nlptest.modelhandler.ModelFactory, data: List[nlptest.utils.custom_types.Sample])



   Class for running robustness tests on models.
   Subclass of TestRunner.

   .. py:method:: evaluate()

      Evaluate the testcases and return the evaluation results.

      Returns:
          List[Sample]:
              all containing the predictions for both the original text and the pertubed one



