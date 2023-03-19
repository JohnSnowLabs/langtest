from .datahandler.datasource import DataFactory
from .modelhandler.modelhandler import ModelFactory
from .nlptest import Harness
from .transform.perturbation import PerturbationFactory


"""
nlptest is python library package which can useful for testing of nlp models
Like Spacy, HuggingFace, SparkNLP ...etc

To testing of NLP Models by import this library as follows

Harness is a class, which can be instaniting and do testing in flow like
workflow: generate --> run --> report --> save 
>>> from nlptest import Harness 

ModelFactory is for handling of strings to models 
like access or download resources from cloud.
>>> from nlptest import ModelFactory

DataFactory is for handling of like csv, json, conll ...
>>> from nlptest import DataFactory

Augumentation of NLP datasets like Perturbations(
uppercase, lowercase, add_context, add_contractions) 
>>> from nlptest import PerturbationFactory

"""
__all__ = [
    Harness,
    ModelFactory,
    DataFactory,
    PerturbationFactory
]
