import importlib
from .modelhandler import ModelAPI

RENAME_HUBS = {
    "azureopenai": "azure-openai",
    "huggingfacehub": "huggingface-inference-api",
}


if importlib.util.find_spec("johnsnowlabs"):
    import langtest.modelhandler.jsl_modelhandler

if importlib.util.find_spec("transformers"):
    import langtest.modelhandler.transformers_modelhandler

if importlib.util.find_spec("spacy"):
    import langtest.modelhandler.spacy_modelhandler

if importlib.util.find_spec("langchain"):
    import langchain
    import langtest.modelhandler.llm_modelhandler

    LANGCHAIN_HUBS = {
        RENAME_HUBS.get(hub.lower(), hub.lower())
        if hub.lower() in RENAME_HUBS
        else hub.lower(): hub
        for hub in langchain.llms.__all__
    }
else:
    LANGCHAIN_HUBS = {}
