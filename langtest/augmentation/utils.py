import re
from typing import List, TypedDict, Union
import os

from pydantic import BaseModel, field_validator
from langtest.logger import logger


class OpenAIConfig(TypedDict):
    """OpenAI Configuration for API Key and Provider."""

    api_key: str = os.environ.get("OPENAI_API_KEY")
    base_url: Union[str, None] = None
    organization: Union[str, None] = (None,)
    project: Union[str, None] = (None,)
    provider: str = "openai"
    model: str = "gpt-4o-mini"


class AzureOpenAIConfig(TypedDict):
    """Azure OpenAI Configuration for API Key and Provider."""

    azure_endpoint: str
    api_version: str
    api_key: str
    provider: str
    azure_deployment: Union[str, None] = None
    azure_ad_token: Union[str, None] = (None,)
    azure_ad_token_provider = (None,)
    organization: Union[str, None] = (None,)
    model: str = "gpt-4o-mini"


class Templates(BaseModel):
    """Model to validate generated templates."""

    templates: List[str]

    def __post_init__(self):
        """Post init method to remove quotes from templates."""
        self.templates = [i.strip('"') for i in self.templates]
        logger.info(f"Generated templates: {self.templates}")

    @field_validator("templates", mode="before")
    def check_templates(cls, v: str):
        """Validator to check if templates are generated."""
        if not v:
            raise ValueError("No templates generated.")
        if isinstance(v, list):
            return [i.strip('"') for i in v]
        return v.strip('"')

    def remove_invalid_templates(self, original_template):
        """Remove invalid templates based on the original template."""
        # extract variable names using regex
        regexs = r"{([^{}]*)}"
        original_vars = re.findall(regexs, original_template)
        original_vars = set([var.strip() for var in original_vars])

        # remove invalid templates
        valid_templates = []
        for template in self.templates:
            template_vars: List[str] = re.findall(regexs, template)
            template_vars = set([var.strip() for var in template_vars])
            if template_vars == original_vars:
                valid_templates.append(template)
                logger.info(f"Valid template: {template}")
            else:
                logger.warning(
                    f"Invalid Variables in template: {template} - {template_vars}"
                )

        self.templates = valid_templates
        logger.info(f"Valid templates: {self.templates}")


def generate_templates_azoi(
    template: str, num_extra_templates: int, model_config: AzureOpenAIConfig
):
    """Generate new templates based on the provided template using Azure OpenAI API."""

    import openai

    model_name = model_config.get("model", "gpt-4o-mini")

    if "provider" in model_config:
        del model_config["provider"]

    if "model" in model_config:
        del model_config["model"]

    client = openai.AzureOpenAI(**model_config)

    prompt = (
        "Based on the provided template, create {num_extra_templates} new and unique templates that are "
        "variations on this theme. Present these as a list, with each template as a quoted string. The list should "
        "contain only the templates, without any additional text or explanation. Ensure that the structure of "
        "these variables remains consistent in each generated template. Note: don't add any extra variables and ignore typo errors.\n\n"
        "Template:\n"
        "{template}\n"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"Generate up to {num_extra_templates} templates based on the provided template.\n\n JSON Output Schema: {Templates.schema()}\n",
            },
            {
                "role": "user",
                "content": prompt.format(
                    template="The {ORG} company is located in {LOC}",
                    num_extra_templates=2,
                ),
            },
            {
                "role": "assistant",
                "content": '["The {ORG} corporation is based out of {LOC}",\n "The {ORG} organization operates in {LOC}"]',
            },
            {
                "role": "user",
                "content": prompt.format(
                    template=template, num_extra_templates=num_extra_templates
                ),
            },
        ],
        temperature=0.1,
    )

    import json

    try:
        clean_response = response.choices[0].message.content.replace("'", '"')
        gen_templates = Templates(templates=json.loads(clean_response))
        gen_templates.remove_invalid_templates(template)

        return gen_templates.templates[:num_extra_templates]

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding response: {e}")
        raise ValueError(f"Error decoding response: {e}")


def generate_templates_openai(
    template: str, num_extra_templates: int, model_config: OpenAIConfig = OpenAIConfig()
):
    """Generate new templates based on the provided template using OpenAI API."""

    import openai

    model_name = model_config.get("model", "gpt-4o-mini")

    if "provider" in model_config:
        del model_config["provider"]

    if "model" in model_config:
        del model_config["model"]

    client = openai.OpenAI(**model_config)

    prompt = (
        f"Based on the provided template, create {num_extra_templates} new and unique templates that are "
        "variations on this theme. Present these as a list, with each template as a quoted string. The list should "
        "contain only the templates, without any additional text or explanation. Ensure that the structure of "
        "these variables remains consistent in each generated template. Note: don't add any extra variables and ignore typo errors.\n\n"
        "Template:\n"
        f"{template}\n"
    )
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"Action: Generate up to {num_extra_templates} templates and ensure that the structure of the variables within the templates remains unchanged and don't add any extra variables.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format=Templates,
    )

    generated_response = response.choices[0].message.parsed
    generated_response.remove_invalid_templates(template)

    return generated_response.templates[:num_extra_templates]


def generate_templates_ollama(
    template: str, num_extra_templates: int, model_config: OpenAIConfig = OpenAIConfig()
):
    """Generate new templates based on the provided template using OpenAI API."""
    import ollama

    # model_name
    model_name = model_config.get("model", "llama3.1")
    try:

        if "provider" in model_config:
            del model_config["provider"]

        if "model" in model_config:
            del model_config["model"]

        client = ollama.Client()

        prompt = (
            f"Based on the provided template, create {num_extra_templates} new and unique templates that are "
            "variations on this theme. Present these as a list, with each template as a quoted string. The list should "
            "contain only the templates, without any additional text or explanation. Ensure that the structure of "
            "these variables remains consistent in each generated template. Note: don't add any extra variables and ignore typo errors.\n\n"
            "Template:\n"
            f"{template}\n"
        )
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"Action: Generate up to {num_extra_templates} templates and ensure that the structure of the variables within the templates remains unchanged and don't add any extra variables.",
                },
                {"role": "user", "content": prompt},
            ],
            format=Templates.model_json_schema(),
        )

        generated_response = Templates.model_validate_json(response.message.content)
        generated_response.remove_invalid_templates(template)
        return generated_response.templates[:num_extra_templates]
    except ollama.ResponseError as e:
        if any("model" in arg for arg in e.args):
            raise ValueError(
                f"Model not found: {e}, please pull model using `ollama pull {model_name}`"
            )
        raise ValueError(f"Error in response: {e}")
