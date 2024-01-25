import click
from .datahandler.datasource import DataFactory
from .modelhandler.modelhandler import ModelAPI
from .langtest import Harness
from .utils.lib_manager import try_import_lib

"""
langtest is python library package which can useful for testing of nlp models
Like Spacy, HuggingFace, SparkNLP ...etc

To testing of NLP Models by import this library as follows

Harness is a class, which can be instaniting and do testing in flow like
workflow: generate --> run --> report --> save
>>> from langtest import Harness

ModelAPI is for handling of strings to models
like access or download resources from cloud.
>>> from langtest import ModelAPI

DataFactory is for handling of like csv, json, conll ...
>>> from langtest import DataFactory


"""

__all__ = [Harness, ModelAPI, DataFactory]

if try_import_lib("transformers"):
    from .callback import LangTestCallback

    __all__.append(LangTestCallback)


@click.group()
def cli():
    pass
