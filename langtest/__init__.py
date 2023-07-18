from .datahandler.datasource import DataFactory
from .modelhandler.modelhandler import ModelFactory
from .langtest import Harness

"""
langtest is python library package which can useful for testing of nlp models
Like Spacy, HuggingFace, SparkNLP ...etc

To testing of NLP Models by import this library as follows

Harness is a class, which can be instaniting and do testing in flow like
workflow: generate --> run --> report --> save
>>> from langtest import Harness

ModelFactory is for handling of strings to models
like access or download resources from cloud.
>>> from langtest import ModelFactory

DataFactory is for handling of like csv, json, conll ...
>>> from langtest import DataFactory


"""
__all__ = [Harness, ModelFactory, DataFactory]
