import unittest
import sys
import pandas as pd
sys.path.insert(0,"..")

from nlptest.nlptest import Harness
from nlptest.datahandler.datasource import DataFactory
from nlptest.modelhandler.modelhandler import ModelFactory


class HarnessTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = Harness(
            task='ner',
            model='dslim/bert-base-NER',
            data="./nlptest_redesign/demo/test.conll",
            config="./nlptest_redesign/demo/config.yml"
        )


    def test_Harness(self):
        self.assertIsInstance(self.harness, Harness)


    def test_attributes(self):
        """
        Testing Attributes of Harness Class
        """
        self.assertIsInstance(self.harness.task, str) 
        self.assertIsInstance(self.harness.model, (str, ModelFactory)) 
        # self.assertIsInstance(self.harness.data, (str, DataFactory)) 
        self.assertIsInstance(self.harness._config, (str, dict)) 

    
    def test_generate_testcases(self):
        df = self.harness.generate() #._load_testcases
        self.assertIsInstance(df, pd.DataFrame)
    
    def test_run_testcases(self):
        self.harness.generate()
        df = self.harness.run() #._load_testcases
        self.assertIsInstance(df, pd.DataFrame)
        
    def test_report(self):
        self.harness.generate()
        df = self.harness.run() #._load_testcases
        self.assertIsInstance(df, pd.DataFrame)
        #Checking Columns
        self.assertEqual(df.columns, ['Test_type', 'fail_count',\
             'pass_count',	'minimum_pass_rate', 'pass_rate', 'pass'])