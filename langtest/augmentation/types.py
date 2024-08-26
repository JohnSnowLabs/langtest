from typing import TypedDict, Union
import os


class OpenAIConfig(TypedDict):
    api_key: str = os.environ.get("OPENAI_API_KEY")
    base_url: Union[str, None] = None
    organization: Union[str, None] = (None,)
    project: Union[str, None] = (None,)
    provider: str = "openai"


class AzureOpenAIConfig(TypedDict):
    from openai.lib.azure import AzureADTokenProvider

    azure_endpoint: str
    api_version: str
    api_key: str
    provider: str
    azure_deployment: Union[str, None] = None
    azure_ad_token: Union[str, None] = (None,)
    azure_ad_token_provider: Union[AzureADTokenProvider, None] = (None,)
    organization: Union[str, None] = (None,)
