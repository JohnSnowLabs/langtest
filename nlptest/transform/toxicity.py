

class BaseToxicity:

    def transform(self):
        pass

    async def async_run(self):
        pass

    async def run(self):
        pass


class PromptToxicity(BaseToxicity):

    def __init__(self, text):
        self.text = text

    def transform(self):
        return self.text

    async def async_run(self):
        return self.transform()

    def run(self):
        return self.transform()