from pkg_resources import resource_filename


LLM_DEFAULTS_CONFIG = {
    "azure-openai": resource_filename(
        "langtest", "data/config/QA_summarization_azure_config.yml"
    ),
    "huggingface": resource_filename(
        "langtest", "data/config/QA_summarization_huggingface_config.yml"
    ),
    "default": resource_filename("langtest", "data/config/QA_summarization_config.yml"),
}

DEFAULTS_CONFIG = {
    "question-answering": LLM_DEFAULTS_CONFIG,
    "summarization": LLM_DEFAULTS_CONFIG,
    "ideology": resource_filename("langtest", "data/config/political_config.yml"),
    "toxicity": resource_filename("langtest", "data/config/toxicity_config.yml"),
    "clinical": resource_filename("langtest", "data/config/clinical_config.yml"),
    "legal": resource_filename("langtest", "data/config/legal_config.yml"),
    "crows-pairs": resource_filename("langtest", "data/config/crows_pairs_config.yml"),
    "stereoset": resource_filename("langtest", "data/config/stereoset_config.yml"),
    "security": resource_filename("langtest", "data/config/security_config.yml"),
    "disinformation": resource_filename(
        "langtest", "data/config/disinformation_config.yml"
    ),
    "factuality": resource_filename("langtest", "data/config/factuality_config.yml"),
    "sycophancy": resource_filename("langtest", "data/config/sycophancy_config.yml"),
    "sensitivity": {
        "huggingface": resource_filename(
            "langtest", "data/config/sensitivity_huggingface_config.yml"
        ),
        "default": resource_filename("langtest", "data/config/sensitivity_config.yml"),
    },
    "translation": {
        "default": resource_filename(
            "langtest", "data/config/translation_transformers_config.yml"
        ),
        "johnsnowlabs": resource_filename(
            "langtest", "data/config/translation_johnsnowlabs_config.yml"
        ),
    },
    "wino-bias": {
        "huggingface": resource_filename(
            "langtest", "data/config/wino_huggingface_config.yml"
        ),
        "default": resource_filename("langtest", "data/config/wino_llm_config.yml"),
    },
}
