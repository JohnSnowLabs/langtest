import re
from ..transform.utils import CMU_dict


class G2p(object):
    def __init__(self):
        super().__init__()
        self.cmu = CMU_dict
        
    def __call__(self, text):
        prons = []
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)

        if text in self.cmu:
            return self.cmu[text][0]
        elif text in self.CMU_dict:
            return self.CMU_dict[text]
        else:

            return prons
g2p = G2p()
        
class Word_Functions:


    def pronunciation(term, generate=False, dictionary=CMU_dict):
        """Takes a term and returns its pronunciation in CMU dict.
        Default behavior is to throw an error if no pronunciation is found.
        if optional argument generate=True, it will attempt to generate a pronunciation."""
        search_pron = []
        for w in term.lower().split():
            if w in dictionary:
                w_pron = dictionary[w]
                search_pron.append(w_pron)
            elif generate:
                 return Pronunciation_Functions.generate_pronunciation(term)
            else:
                raise ValueError(
                    "Dictionary Error: Search term or search token not found in dictionary. "
                    "Contact administrator to update dictionary if necessary."
                )

        pron = [p for sublist in search_pron for p in sublist]  # flatten list of lists into one list
        return pron

class Pronunciation_Functions:
    def generate_pronunciation(text):
        
        if text not in CMU_dict:
            return [phone for phone in g2p(text) if phone != ' ']
        else:
            return CMU_dict[text]


class Phone_Functions:

    def unstressed_phone(phone):
        """Takes a phone and removes the stress marker if it exists"""
        if not phone[-1].isdigit():
            return phone
        else:
            return phone[:-1]


class Search:

    def perfectHomophones(Search_Term, generate=False, dictionary=CMU_dict):
        """
        Takes a search term, searches its pronunciation,
        and returns a list of words with the exact same pronunciation in CMU Dict.
        """
        Search_Pron = Word_Functions.pronunciation(Search_Term, generate)
        PerfectHomophones = [word.title() for word in dictionary if dictionary[word] == Search_Pron]

        return PerfectHomophones