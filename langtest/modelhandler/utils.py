# This file contains the model classes that are used in the model handler.
# from langchain
from typing import Dict, TypedDict, Union


class ModuleInfo(TypedDict):
    module: str  # module path
    chat: str  # class name for chat model
    completion: str  # class name for completion model


CHAT_MODEL_CLASSES: Dict[str, Union[ModuleInfo, str]] = {
    "anthropic": {
        "module": "langchain_anthropic.chat_models",
        "chat": "ChatAnthropic",
        "completion": "Anthropic",
    },
    "anyscale": "ChatAnyscale",
    "azure_openai": {
        "module": "langchain_openai.chat_models",
        "chat": "AzureChatOpenAI",
        "completion": "AzureOpenAI",
    },
    "baichuan": "ChatBaichuan",
    "baidu_qianfan_endpoint": "QianfanChatEndpoint",
    "bedrock": "BedrockChat",
    "cohere": {
        "module": "langchain_cohere.chat_models",
        "chat": "ChatCohere",
        "completion": "Cohere",
    },
    "databricks": {
        "module": "langchain_databricks.chat_models",
        "chat": "ChatDatabricks",
        "completion": "Databricks",
    },
    "deepinfra": "ChatDeepInfra",
    "ernie": "ErnieBotChat",
    "everlyai": "ChatEverlyAI",
    "fake": "FakeListChatModel",
    "fireworks": {
        "module": "langchain_fireworks.chat_models",
        "chat": "ChatFireworks",
        "completion": "Fireworks",
    },
    "gigachat": "GigaChat",
    "google_palm": "ChatGooglePalm",
    "gpt_router": "GPTRouter",
    "huggingface": "ChatHuggingFace",
    "human": "HumanInputChatModel",
    "hunyuan": "ChatHunyuan",
    "javelin_ai_gateway": "ChatJavelinAIGateway",
    "jinachat": "JinaChat",
    "kinetica": "ChatKinetica",
    "konko": "ChatKonko",
    "litellm": "ChatLiteLLM",
    "litellm_router": "ChatLiteLLMRouter",
    "llama_edge": "LlamaEdgeChatService",
    "maritalk": "ChatMaritalk",
    "minimax": "MiniMaxChat",
    "mlflow": "ChatMlflow",
    "mlflow_ai_gateway": "ChatMLflowAIGateway",
    "ollama": {
        "module": "langchain_ollama.chat_models",
        "chat": "ChatOllama",
        "completion": "Ollama",
    },
    "openai": {
        "module": "langchain_openai.chat_models",
        "chat": "ChatOpenAI",
        "completion": "OpenAI",
    },
    "pai_eas_endpoint": "PaiEasChatEndpoint",
    "perplexity": "ChatPerplexity",
    "promptlayer_openai": "PromptLayerChatOpenAI",
    "sparkllm": "ChatSparkLLM",
    "tongyi": "ChatTongyi",
    "vertexai": "ChatVertexAI",
    "volcengine_maas": "VolcEngineMaasChat",
    "yandex": "ChatYandexGPT",
    "yuan2": "ChatYuan2",
    "zhipuai": "ChatZhipuAI",
}
