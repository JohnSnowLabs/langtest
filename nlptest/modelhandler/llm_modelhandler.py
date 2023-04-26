import langchain.llms as llms

from ..modelhandler.modelhandler import _ModelHandler

all_models_hub = llms.__all__



class PertrainedLlmModel(_ModelHandler):

    def __init__(self, model: str, hub: str):
        self.model = model
        self.hub = hub
    
    @classmethod
    def load_model(cls, model: str, hub: str, *args, **kwargs):
        """Load and return SpaCy pipeline"""

        try:
            return getattr(llms, hub)(model_name=model, *args, **kwargs)
        except:
            raise ValueError(
                f'''Model "{model}" is not found online or local.
                Please install langchain by pip install langchain''')

    def predict(self, text: str, *args, **kwargs):
        pass

    def __call__(self, text: str, *args, **kwargs):
        """Alias of the 'predict' method"""
        return self.predict(text=text)