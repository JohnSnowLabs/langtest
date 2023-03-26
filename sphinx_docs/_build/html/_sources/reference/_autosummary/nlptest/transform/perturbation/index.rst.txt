:orphan:

.. INDEX

:py:mod:`nlptest.transform.perturbation`
========================================

.. py:module:: nlptest.transform.perturbation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.transform.perturbation.BasePerturbation
   nlptest.transform.perturbation.UpperCase
   nlptest.transform.perturbation.LowerCase
   nlptest.transform.perturbation.TitleCase
   nlptest.transform.perturbation.AddPunctuation
   nlptest.transform.perturbation.StripPunctuation
   nlptest.transform.perturbation.AddTypo
   nlptest.transform.perturbation.SwapEntities
   nlptest.transform.perturbation.GenderPronounBias
   nlptest.transform.perturbation.SwapCohyponyms
   nlptest.transform.perturbation.ConvertAccent
   nlptest.transform.perturbation.AddContext
   nlptest.transform.perturbation.AddContraction



Functions
~~~~~~~~~

.. autoapisummary::

   nlptest.transform.perturbation.get_cohyponyms_wordnet



.. py:class:: BasePerturbation



   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: UpperCase



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Transform a list of strings with uppercase perturbation
      Args:
          sample_list: List of sentences to apply perturbation.
      Returns:
          List of sentences that uppercase perturbation is applied.



.. py:class:: LowerCase



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Transform a list of strings with lowercase perturbation
      Args:
          sample_list: List of sentences to apply perturbation.
      Returns:
          List of sentences that lowercase perturbation is applied.



.. py:class:: TitleCase



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Transform a list of strings with titlecase perturbation
      Args:
          sample_list: List of sentences to apply perturbation.
      Returns:
          List of sentences that titlecase perturbation is applied.



.. py:class:: AddPunctuation



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], whitelist: Optional[List[str]] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Add punctuation at the end of the string, if there is punctuation at the end skip it
      Args:
          sample_list: List of sentences to apply perturbation.
          whitelist: Whitelist for punctuations to add to sentences.
      Returns:
          List of sentences that have punctuation at the end.



.. py:class:: StripPunctuation



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], whitelist: Optional[List[str]] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Add punctuation from the string, if there isn't punctuation at the end skip it

      Args:
          sample_list: List of sentences to apply perturbation.
          whitelist: Whitelist for punctuations to strip from sentences.
      Returns:
          List of sentences that punctuation is stripped.



.. py:class:: AddTypo



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Add typo to the sentences using keyboard typo and swap typo.
      Args:
          sample_list: List of sentences to apply perturbation.
      Returns:
          List of sentences that typo introduced.



.. py:class:: SwapEntities



   Helper class that provides a standard way to create an ABC using
   inheritance.

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


.. py:class:: GenderPronounBias



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], pronouns_to_substitute: List[str], pronoun_type: str) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Replace pronouns to check the gender bias

      Args:
          sample_list: List of sentences to apply perturbation.
          pronouns_to_substitute: list of pronouns that need to be substituted.
          pronoun_type: replacing pronoun type string ('male', 'female' or 'neutral')

      Returns:
          List of sentences with replaced pronouns



.. py:class:: SwapCohyponyms



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], labels: List[List[str]] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Swaps named entities with the new one from the terminology extracted from passed data.

      Args:
          sample_list: List of sentences to process.
          labels: Corresponding labels to make changes according to sentences.

      Returns:
          List sample indexes and corresponding augmented sentences, tags and labels if provided.



.. py:class:: ConvertAccent



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], accent_map: Dict[str, str] = None) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Converts input sentences using a conversion dictionary
      Args:
          sample_list: List of sentences to process.
          accent_map: Dictionary with conversion terms.
      Returns:
          List of sentences that perturbed with accent conversion.



.. py:class:: AddContext



   Helper class that provides a standard way to create an ABC using
   inheritance.

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



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample]) -> List[str]
      :staticmethod:

      Converts input sentences using a conversion dictionary
      Args:
          sample_list: List of sentences to process.



