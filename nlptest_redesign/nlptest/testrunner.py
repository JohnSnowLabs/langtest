
import pandas as pd
from sparknlp.base import LightPipeline


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
        expected_result=[]
        actual_result=[]
        for i, r in self._load_testcases.iterrows():

            if "spacy" in str(type(self._model_handler)):
                expected_result_dict={}
                actual_result_dict={}
                doc1 = self._model_handler(r['Orginal'])
                doc2 = self._model_handler(r['Test_Case'])
                
                for token in doc1:
                    if not token.ent_type_:
                        expected_result_dict[token]='O'
                    
                    else:
                        expected_result_dict[token]=token.ent_type_
                
                for token in doc2:
                    if not token.ent_type_:
                        actual_result_dict[token]='O'
                    
                    else:
                        actual_result_dict[token]=token.ent_type_
                
                # print("Processed row : ",i)
                

                expected_result.append(list(expected_result_dict.values()))

                actual_result.append(list(actual_result_dict.values())) 
                
            elif "spark" in str(type(self._model_handler)):
                expected_result.append(LightPipeline(self._model_handler).annotate(r['Orginal'])['ner'])
                actual_result.append(LightPipeline(self._model_handler).annotate(r['Test_Case'])['ner'])
                # print("Processed row : ",i)


        self._load_testcases['expected_result'] =expected_result
        self._load_testcases['actual_result'] =actual_result
        #Checking for any token mismatches

        final_perturbed_labels=[]
        for i,r in self._load_testcases.iterrows():
            main_list=(r['Test_Case']).split(' ')
            sub_list=(r['Orginal']).split(' ')
            org_sentence_labels=list(r['expected_result'])
            perturbed_sentence_labels=list(r['actual_result'])
            if len(org_sentence_labels)==len(perturbed_sentence_labels):
                final_perturbed_labels.append(perturbed_sentence_labels)  
            else:
                for i, _ in enumerate(main_list): 
                    if main_list[i:i+len(sub_list)] == sub_list:
                        sub_list_start=i
                        sub_list_end=i+len(sub_list)
                        final_perturbed_labels.append(perturbed_sentence_labels[sub_list_start:sub_list_end])
            
        self._load_testcases['actual_result']=final_perturbed_labels
        self._load_testcases = self._load_testcases.assign(is_pass=self._load_testcases.apply(lambda row: row['expected_result'] == row['actual_result'], axis=1))
        return self._load_testcases


