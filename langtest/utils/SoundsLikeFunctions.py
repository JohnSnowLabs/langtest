import re
from typing import List
from ..transform.constants import CMU_dict


class G2p(object):
    """Grapheme-to-Phoneme conversion class.

    Converts a given text to its corresponding phonetic representation using the CMU dictionary.
    """

    def __init__(self):
        """Constructor method"""
        super().__init__()
        self.cmu = CMU_dict

    def __call__(self, text: str) -> List[str]:
        """Converts the given text to its corresponding phonetic representation.

        Args:
            text (str): The input text to convert.

        Returns:
            list: A list of phonetic representations of the input text.
        """
        prons = []
        text = text.lower()
        text = re.sub("[^ a-z'.,?!-]", "", text)

        if text in self.cmu:
            return self.cmu[text][0]
        elif text in self.CMU_dict:
            return self.CMU_dict[text]
        else:
            return prons


g2p = G2p()


class WordFunctions:
    """Class containing functions related to words and their pronunciations."""

    @staticmethod
    def pronunciation(
        term: str, generate: bool = False, dictionary: dict = CMU_dict
    ) -> List[str]:
        """Takes a term and returns its pronunciation in the CMU dictionary.

        If no pronunciation is found, it throws an error by default.
        If generate is set to True, it attempts to generate a pronunciation.
        Args:
            term (str): The term for which the pronunciation is to be retrieved.
            generate (bool): Whether to attempt pronunciation generation if the term is not found (default: False).
            dictionary (dict): The dictionary containing the word-pronunciation mappings (default: CMU_dict).

        Returns:
            List[str]: A list of phonetic representations of the term's pronunciation.

        Raises:
            ValueError: If the search term or search token is not found in the dictionary.
        """
        search_pron = []
        for w in term.lower().split():
            if w in dictionary:
                w_pron = dictionary[w]
                search_pron.append(w_pron)
            elif generate:
                return PronunciationFunctions.generate_pronunciation(term)
            else:
                raise ValueError(
                    "Dictionary Error: Search term or search token not found in dictionary. "
                    "Contact administrator to update dictionary if necessary."
                )

        pron = [
            p for sublist in search_pron for p in sublist
        ]  # flatten list of lists into one list
        return pron


class PronunciationFunctions:
    """Class containing functions related to phonetic representations and pronunciations."""

    @staticmethod
    def generate_pronunciation(text: str) -> List[str]:
        """Generates the phonetic representation of a given text using the CMU dictionary.

        If the text is not found in the dictionary, it returns a list of phonetic representations
        obtained from the G2p class.

        Args:
            text (str): The input text for which the pronunciation is to be generated.

        Returns:
            List[str]: A list of phonetic representations of the input text.
        """
        if text not in CMU_dict:
            return [phone for phone in g2p(text) if phone != " "]
        else:
            return CMU_dict[text]


class PhoneFunctions:
    """Class containing functions related to phones and phonetic representations."""

    @staticmethod
    def unstressed_phone(phone: str) -> str:
        """Removes the stress marker from a given phone if it exists.

        Args:
            phone (str): The phone from which the stress marker is to be removed.

        Returns:
            str: The phone without the stress marker.
        """
        if not phone[-1].isdigit():
            return phone
        else:
            return phone[:-1]


class Search:
    """Class containing functions for searching words and their pronunciations."""

    @staticmethod
    def perfectHomophones(
        search_term: str, generate: bool = False, dictionary: dict = CMU_dict
    ) -> List[str]:
        """Searches for words with the exact same pronunciation as the given search term in the CMU dictionary.

        Args:
            search_term (str): The search term for which to find perfect homophones.
            generate (bool): Whether to attempt pronunciation generation for the search term (default: False).
            dictionary (dict): The dictionary containing the word-pronunciation mappings (default: CMU_dict).

        Returns:
            list: A list of words with the exact same pronunciation as the search term.
        """
        Search_Pron = WordFunctions.pronunciation(search_term, generate)
        PerfectHomophones = [
            word.title() for word in dictionary if dictionary[word] == Search_Pron
        ]
        return PerfectHomophones
