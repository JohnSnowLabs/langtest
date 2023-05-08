import inspect
from typing import Union
import langchain.llms as lc
from langchain import LLMChain, PromptTemplate
from ..modelhandler.modelhandler import _ModelHandler

DEFAULT_LLM_HUB = {hub.lower(): hub for hub in lc.__all__}

class PretrainedModelForQA(_ModelHandler):

    def __init__(self,  hub: str, model: str, *args, **kwargs):
        self.model = model
        self.hub = DEFAULT_LLM_HUB[hub]
        self.kwargs = kwargs

    @classmethod
    def load_model(cls, hub: str, path: str, *args, **kwargs):
        """"""

        try:
            model = getattr(lc, DEFAULT_LLM_HUB[hub])
            default_args = inspect.getfullargspec(model).kwonlyargs
            # warning if model parameters are not passed to specfic model
            if 'model' in default_args:
                cls.model = model(model=path, *args, **kwargs)
            else:
                cls.model = model(model_name=path, *args, **kwargs)
            return cls.model
        except ImportError:
            raise ValueError(
                f'''Model "{path}" is not found online or local.
                Please install langchain by pip install langchain''')

    def predict(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        prompt_template = PromptTemplate(**prompt)
        llmchain = LLMChain(prompt=prompt_template, llm=self.model)
        return llmchain.run(**text)

    def __call__(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        """Alias of the 'predict' method"""
        return self.predict(text, prompt, *args, **kwargs)
    
    def default_args(self, parameter: str, hub: str):

        defaults_args = {
            'ai21': {
                'model_name': 'model',
                'max_tokens': 'maxToken'
            },
            'anthropic': {
                'model_name': 'model',
                'max_tokens': 'max_tokens_to_sample'
            },
            'cohere': {
                'model_name': 'model',
                'max_tokens': 'max_tokens_to_sample'
            }

        }
        return defaults_args[hub][parameter]

