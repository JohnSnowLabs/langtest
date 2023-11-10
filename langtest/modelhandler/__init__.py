import importlib
from .modelhandler import ModelAPI
import langtest.modelhandler.custom_modelhandler

RENAME_HUBS = {
    "azureopenai": "azure-openai",
    "huggingfacehub": "huggingface-inference-api",
}

INSTALLED_HUBS = ["custom"]

libraries = [
    ("johnsnowlabs", "langtest.modelhandler.jsl_modelhandler"),
    ("transformers", "langtest.modelhandler.transformers_modelhandler"),
    ("spacy", "langtest.modelhandler.spacy_modelhandler"),
    ("langchain", "langtest.modelhandler.llm_modelhandler"),
]

for library_name, import_statement in libraries:
    if importlib.util.find_spec(library_name):
        importlib.import_module(import_statement)
        if library_name in ("transformers"):
            INSTALLED_HUBS.append("huggingface")
        else:
            INSTALLED_HUBS.append(library_name)

if "langchain" in INSTALLED_HUBS:
    import langchain

    LANGCHAIN_HUBS = {
        RENAME_HUBS.get(hub.lower(), hub.lower())
        if hub.lower() in RENAME_HUBS
        else hub.lower(): hub
        for hub in langchain.llms.__all__
    }
    INSTALLED_HUBS += list(LANGCHAIN_HUBS.keys())
else:
    LANGCHAIN_HUBS = {}
