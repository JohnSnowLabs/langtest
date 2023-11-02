import importlib
from .modelhandler import ModelAPI


RENAME_HUBS = {
    "azureopenai": "azure-openai",
    "huggingfacehub": "huggingface-inference-api",
}

INSTALLED_HUBS = []

if importlib.util.find_spec("johnsnowlabs"):
    import langtest.modelhandler.jsl_modelhandler

    INSTALLED_HUBS.append("johnsnowlabs")

if importlib.util.find_spec("transformers"):
    import langtest.modelhandler.transformers_modelhandler

    INSTALLED_HUBS.append("transformers")

if importlib.util.find_spec("spacy"):
    import langtest.modelhandler.spacy_modelhandler

    INSTALLED_HUBS.append("spacy")

if importlib.util.find_spec("langchain"):
    import langchain
    import langtest.modelhandler.llm_modelhandler

    LANGCHAIN_HUBS = {
        RENAME_HUBS.get(hub.lower(), hub.lower())
        if hub.lower() in RENAME_HUBS
        else hub.lower(): hub
        for hub in langchain.llms.__all__
    }
    INSTALLED_HUBS + list(LANGCHAIN_HUBS.keys())
else:
    LANGCHAIN_HUBS = {}

import langtest.modelhandler.custom_modelhandler
