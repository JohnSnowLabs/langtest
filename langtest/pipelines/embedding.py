import os
import pickle
from langtest import Harness
from llama_index.readers import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.evaluation import generate_question_context_pairs
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices import VectorStoreIndex
from llama_index.service_context import ServiceContext, set_global_service_context

package_path = os.path.abspath(__package__)


class BasePipeline:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError


class EmbeddingPipeline(BasePipeline):
    def __init__(self, embed_model, dataset=None, config=None, hub="huggingface"):
        super().__init__()
        self.embed_model = embed_model
        self.dataset = dataset or f"{package_path}/data/Retrieval_Datasets/qa_dataset.pkl"
        self.config = config
        self.documents = None
        self.nodes = None
        self.llm = None
        self.hub = hub
        self.node_parser = None
        self.qa_dataset = None

    def load_data(self):
        if isinstance(self.dataset, str):
            if self.dataset.endswith(".pkl"):
                with open(self.dataset, "rb") as file:
                    out = pickle.load(file)
                    self.documents = out["documents"]
                    self.nodes = out["nodes"]
                    self.llm = out["llm"]
                    self.node_parser = out["node_parser"]
                    self.qa_dataset = out["qa_dataset"]
            else:
                documents = SimpleDirectoryReader(self.dataset).load_data()

                try:
                    # Define an LLM
                    llm = OpenAI(model="gpt-3.5-turbo")
    
                except Exception as e:
                    print(f"An error occurred: {e}. Kindly add your key to environment using python -m langtest config set 'OPENAI_API_KEY'= KEY ")


                # Build index with a chunk_size of 512
                node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
                nodes = node_parser.get_nodes_from_documents(documents)

                qa_dataset = generate_question_context_pairs(
                    nodes, llm=llm, num_questions_per_chunk=2
                )

                self.documents = documents
                self.nodes = nodes
                self.llm = llm
                self.node_parser = node_parser
                self.qa_dataset = qa_dataset

                with open(self.dataset, "wb") as file:
                    pickle.dump({
                        "documents": self.documents,
                        "nodes": self.nodes,
                        "llm": self.llm,
                        "node_parser": self.node_parser,
                        "qa_dataset": self.qa_dataset,
                    }, file)


    def load_model(self):
        if self.hub=="huggingface":
            em = HuggingFaceEmbedding(self.embed_model, max_length=512)
            servicecontext = ServiceContext.from_defaults(embed_model=em)
            set_global_service_context(servicecontext)
            vector_index = VectorStoreIndex(self.nodes, servicecontext = servicecontext)
            self.query_engine = vector_index.as_query_engine()
        else:
            vector_index = VectorStoreIndex(self.nodes)
            self.query_engine = vector_index.as_query_engine()

          
            

   

