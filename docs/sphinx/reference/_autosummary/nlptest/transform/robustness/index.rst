:orphan:

.. INDEX

:py:mod:`nlptest.transform.robustness`
======================================

.. py:module:: nlptest.transform.robustness


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.transform.robustness.BaseRobustness
   nlptest.transform.robustness.UpperCase
   nlptest.transform.robustness.LowerCase
   nlptest.transform.robustness.TitleCase
   nlptest.transform.robustness.AddPunctuation
   nlptest.transform.robustness.StripPunctuation
   nlptest.transform.robustness.AddTypo
   nlptest.transform.robustness.SwapEntities
   nlptest.transform.robustness.SwapCohyponyms
   nlptest.transform.robustness.ConvertAccent
   nlptest.transform.robustness.AddContext
   nlptest.transform.robustness.AddContraction



Functions
~~~~~~~~~

.. autoapisummary::

   nlptest.transform.robustness.get_cohyponyms_wordnet



.. py:class:: BaseRobustness



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:
      :abstractmethod:

      Abstract method that implements the robustness measure.

      Args:
          data (List[Sample]): The input data to be transformed.

      Returns:
          Any: The transformed data based on the implemented robustness measure.



.. py:class:: UpperCase



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Transform a list of strings with uppercase robustness
      Args:
          sample_list: List of sentences to apply robustness.
      Returns:
          List of sentences that uppercase robustness is applied.



.. py:class:: LowerCase



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Transform a list of strings with lowercase robustness
      Args:
          sample_list: List of sentences to apply robustness.
      Returns:
          List of sentences that lowercase robustness is applied.



.. py:class:: TitleCase



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Transform a list of strings with titlecase robustness
      Args:
          sample_list: List of sentences to apply robustness.
      Returns:
          List of sentences that titlecase robustness is applied.



.. py:class:: AddPunctuation



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], whitelist: Optional[List[str]] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Add punctuation at the end of the string, if there is punctuation at the end skip it
      Args:
          sample_list: List of sentences to apply robustness.
          whitelist: Whitelist for punctuations to add to sentences.
      Returns:
          List of sentences that have punctuation at the end.



.. py:class:: StripPunctuation



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], whitelist: Optional[List[str]] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Add punctuation from the string, if there isn't punctuation at the end skip it

      Args:
          sample_list: List of sentences to apply robustness.
          whitelist: Whitelist for punctuations to strip from sentences.
      Returns:
          List of sentences that punctuation is stripped.



.. py:class:: AddTypo



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Add typo to the sentences using keyboard typo and swap typo.
      Args:
          sample_list: List of sentences to apply robustness.
      Returns:
          List of sentences that typo introduced.



.. py:class:: SwapEntities



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], labels: List[List[str]] = None, terminology: Dict[str, List[str]] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Swaps named entities with the new one from the terminology extracted from passed data.

      Args:
          sample_list: List of sentences to process.
          labels: Corresponding labels to make changes according to sentences.
          terminology: Dictionary of entities and corresponding list of words.
      Returns:
          List of sentences that entities swapped with the terminology.



.. py:function:: get_cohyponyms_wordnet(word: str) -> str

   Retrieve co-hyponym of the input string using WordNet when a hit is found.

   Args:
       word: input string to retrieve co-hyponym
   Returns:
       Cohyponym of the input word if exists, else original word.


.. py:class:: SwapCohyponyms



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], labels: List[List[str]] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Swaps named entities with the new one from the terminology extracted from passed data.

      Args:
          sample_list: List of sentences to process.
          labels: Corresponding labels to make changes according to sentences.

      Returns:
          List sample indexes and corresponding augmented sentences, tags and labels if provided.



.. py:class:: ConvertAccent



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], accent_map: Dict[str, str] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Converts input sentences using a conversion dictionary
      Args:
          sample_list: List of sentences to process.
          accent_map: Dictionary with conversion terms.
      Returns:
          List of sentences that perturbed with accent conversion.



.. py:class:: AddContext



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], starting_context: Optional[List[str]] = None, ending_context: Optional[List[str]] = None, strategy: List[str] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Converts input sentences using a conversion dictionary
      Args:
          sample_list: List of sentences to process.
          strategy: Config method to adjust where will context tokens added. start, end or combined.
          starting_context: list of terms (context) to input at start of sentences.
          ending_context: list of terms (context) to input at end of sentences.
      Returns:
          List of sentences that context added at to begging, end or both, randomly.



.. py:class:: AddContraction



   Abstract base class for implementing robustness measures.

   Attributes:
       alias_name (str): A name or list of names that identify the robustness measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[str]
      :staticmethod:

      Converts input sentences using a conversion dictionary
      Args:
          sample_list: List of sentences to process.



