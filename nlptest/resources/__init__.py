import os
import pickle


class Resource:

    resoucres_dir = os.path.dirname(__file__)

    def __init__(self) -> None:
        self.files = {i[:-4]: f'{Resource.resoucres_dir}/{i}' 
                      for i in os.listdir(Resource.resoucres_dir) 
                      if i.endswith('.pkl')}

    def __getitem__(self, resource_name: str):
        return pickle.load(open(self.files[resource_name.upper()], 'rb'))
    
    def get_names(self):
        return list(self.files.keys())
