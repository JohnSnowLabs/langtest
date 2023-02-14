import pandas as pd
from sparknlp.base import LightPipeline
from typing import List

class TestRunner:
    """Base class for running tests on models.
    """

    def __init__(
            self,
            load_testcases: pd.DataFrame,
            model_handler,
    ) -> None:
        """Initialize the TestRunner class.

        Args:
            load_testcases (pd.DataFrame): DataFrame containing the testcases to be evaluated.
            model_handler (spark or spacy model): Object representing the model handler, either spaCy or SparkNLP.
        """
        self._load_testcases = load_testcases.copy()
        self._model_handler = model_handler

        if self._model_handler.backend in ["huggingface", "spacy"]:
            self._model_handler.load_model()

    # @abc.abstractmethod
    def evaluate(self):
        """Abstract method to evaluate the testcases.

        Returns:
            DataFrame: DataFrame containing the evaluation results.
        """
        runner = RobustnessTestRunner(self._load_testcases, self._model_handler)
        return runner.evaluate()


class RobustnessTestRunner(TestRunner):
    """Class for running robustness tests on models.
    Subclass of TestRunner.
    """

    def evaluate(self):
        """Evaluate the testcases and return the evaluation results.

        Returns:
            pd.DataFrame: DataFrame containing the evaluation results.
        """
        expected_result = []
        actual_result = []

        for i, r in self._load_testcases.iterrows():
            if self._model_handler.task == "text-classification":
                expected_result.append(self._model_handler(r["Original"]))
                actual_result.append(self._model_handler(r["Test_Case"]))
            else:
                if "spacy" in self._model_handler.backend:
                    doc1 = self._model_handler(r['Original'])
                    doc2 = self._model_handler(r['Test_Case'])

                    expected_result.append([pred.entity for pred in doc1])
                    actual_result.append([pred.entity for pred in doc2])

                elif "sparknlp.pretrained" in self._model_handler.backend:
                    expected_result.append((self._model_handler).annotate(r['Original'])['ner'])
                    actual_result.append((self._model_handler).annotate(r['Test_Case'])['ner'])

                elif "spark" in self._model_handler.backend:
                    expected_result.append(LightPipeline(self._model_handler).annotate(r['Original'])['ner'])
                    actual_result.append(LightPipeline(self._model_handler).annotate(r['Test_Case'])['ner'])

                elif "huggingface" in self._model_handler.backend:
                    doc1 = self._model_handler(r['Original'])
                    doc2 = self._model_handler(r['Test_Case'])

                    expected_result.append([pred.entity for pred in doc1])
                    actual_result.append([pred.entity for pred in doc2])

        self._load_testcases['expected_result'] = expected_result
        self._load_testcases['actual_result'] = actual_result
        # Checking for any token mismatches

        final_perturbed_labels = []
        for i, r in self._load_testcases.iterrows():
            main_list = r['Test_Case'].split(' ')
            sub_list = r['Original'].split(' ')

            org_sentence_labels = list(r['expected_result'])
            perturbed_sentence_labels = list(r['actual_result'])

            if len(org_sentence_labels) == len(perturbed_sentence_labels):
                final_perturbed_labels.append(perturbed_sentence_labels)
            else:
                # HACK: it might happen that no sublist are actually found which leads to some mismatch
                # size in the dataframe. Adding an empty list to overcome this until we fix the
                # token mismatch problem
                is_added = False
                for i, _ in enumerate(main_list):
                    if main_list[i:i + len(sub_list)] == sub_list:
                        sub_list_start = i
                        sub_list_end = i + len(sub_list)
                        final_perturbed_labels.append(perturbed_sentence_labels[sub_list_start:sub_list_end])
                        is_added = True
                if not is_added:
                    final_perturbed_labels.append([])

        self._load_testcases['actual_result'] = final_perturbed_labels
        self._load_testcases = self._load_testcases.assign(
            is_pass=self._load_testcases.apply(lambda row: row['expected_result'] == row['actual_result'], axis=1))
        return self._load_testcases

        
    def _eval_add_contraction(self):
        e_words = [(x.word, x.label) for x in self.expected_result if x.label != "O"]
        a_words = [(x.word, x.label) for x in self.actual_result if x.label != "O"]
        return e_words == a_words

    def _eval_add_punctuation(self):
        a_words = list(map(lambda x: (x.word, x.label), self.actual_result))
        e_words = list(map(lambda x: (x.word, x.label), self.expected_result))

        if not e_words[-1][0].isalnum():
            return a_words == e_words[:-1]
        return a_words == e_words


    def _eval_convert_spelling(self):
        e_words = [(x.word, x.label) for x in self.expected_result if x.label != "O"]
        a_words = [(x.word, x.label) for x in self.actual_result if x.label != "O"]
        return e_words == a_words


    def _IOB_to_IO(self, iob_tokens):
        return filter(lambda x: x.label[0]!="I", iob_tokens)

    def _eval_title_case(self):
        e_words = [(x.word, x.label) for x in self.expected_result]
        a_words = [(x.word, x.label) for x in self.actual_result]
        return len(e_words) == len(a_words) and e_words == a_words

    def _eval_strip_punc(self):
        e_words = [(x.word, x.label) for x in self.expected_result if x.label != "O"]
        a_words = [(x.word, x.label) for x in self.actual_result if x.label != "O"]
        if a_words[-1][0].isalnum():
            a_words = a_words[:-1]

        return len(e_words) == len(a_words) and e_words == a_words

    def _eval_swap_cohyphonyms(self):
        a_words = list(map(lambda x: (x.word, x.label), self.actual_result))
        e_words = list(map(lambda x: (x.word, x.label), self.expected_result))

        a_words = self._IOB_to_IO(a_words)
        e_words = self._IOB_to_IO(e_words)
        
        
        return len(e_words) == len(a_words) and a_words == e_words

