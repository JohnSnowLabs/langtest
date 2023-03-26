:orphan:

.. INDEX

:py:mod:`nlptest.transform.bias`
================================

.. py:module:: nlptest.transform.bias


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nlptest.transform.bias.BaseBias
   nlptest.transform.bias.GenderPronounBias
   nlptest.transform.bias.CountryEconomicBias
   nlptest.transform.bias.EthnicityNameBias
   nlptest.transform.bias.ReligionBias




.. py:class:: BaseBias



   Abstract base class for implementing bias measures.

   Attributes:
       alias_name (str): A name or list of names that identify the bias measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.

   .. py:method:: transform()
      :abstractmethod:

      Abstract method that implements the bias measure.

      Args:
          data (List[Sample]): The input data to be transformed.

      Returns:
          Any: The transformed data based on the implemented bias measure.



.. py:class:: GenderPronounBias



   Abstract base class for implementing bias measures.

   Attributes:
       alias_name (str): A name or list of names that identify the bias measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], pronouns_to_substitute: List[str], pronoun_type: str) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Replace pronouns to check the gender bias

      Args:
          sample_list: List of sentences to apply perturbation.
          pronouns_to_substitute: list of pronouns that need to be substituted.
          pronoun_type: replacing pronoun type string ('male', 'female' or 'neutral')

      Returns:
          List of sentences with replaced pronouns



.. py:class:: CountryEconomicBias



   Abstract base class for implementing bias measures.

   Attributes:
       alias_name (str): A name or list of names that identify the bias measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], country_names_to_substitute: List[str], chosen_country_names: List[str]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Replace country names to check the ethnicity bias


      Args:
          sample_list: List of sentences to apply perturbation.
          country_names_to_substitute: list of country names that need to be substituted.
          chosen_country_names: list of country names to replace with.

      Returns:
          List of sentences with replaced names



.. py:class:: EthnicityNameBias



   Abstract base class for implementing bias measures.

   Attributes:
       alias_name (str): A name or list of names that identify the bias measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], names_to_substitute: List[str], chosen_ethnicity_names: List[str]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Replace names to check the ethnicity bias
      Ethnicity Dataset Curated from the United States Census Bureau surveys

      Args:
          sample_list: List of sentences to apply perturbation.
          names_to_substitute: list of ethnicity names that need to be substituted.
          chosen_ethnicity_names: list of ethnicity names to replace with.

      Returns:
          List of sentences with replaced names



.. py:class:: ReligionBias



   Abstract base class for implementing bias measures.

   Attributes:
       alias_name (str): A name or list of names that identify the bias measure.

   Methods:
       transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.

   .. py:method:: transform(sample_list: List[nlptest.utils.custom_types.Sample], names_to_substitute: List[str], chosen_names: List[str]) -> List[nlptest.utils.custom_types.Sample]
      :staticmethod:

      Replace  names to check the religion bias


      Args:
          sample_list: List of sentences to apply perturbation.
          names_to_substitute: list of names that need to be substituted.
          chosen_names: list of names to replace with.

      Returns:
          List of sentences with replaced names



