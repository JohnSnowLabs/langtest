from pkg_resources import resource_filename

LLM_DEFAULTS_CONFIG = {
    "azure-openai": resource_filename("langtest", "data/config/azure_config.yml"),
    "openai": resource_filename("langtest", "data/config/openai_config.yml"),
    "cohere": resource_filename("langtest", "data/config/cohere_config.yml"),
    "ai21": resource_filename("langtest", "data/config/ai21_config.yml"),
    "huggingface-inference-api": resource_filename(
        "langtest", "data/config/huggingface_config.yml"
    ),
}

DEFAULTS_CONFIG = {
    "question-answering": LLM_DEFAULTS_CONFIG,
    "summarization": LLM_DEFAULTS_CONFIG,
    "ideology": resource_filename("langtest", "data/config/political_config.yml"),
    "toxicity": resource_filename("langtest", "data/config/toxicity_config.yml"),
    "clinical-tests": resource_filename("langtest", "data/config/clinical_config.yml"),
    "legal-tests": resource_filename("langtest", "data/config/legal_config.yml"),
    "crows-pairs": resource_filename("langtest", "data/config/crows_pairs_config.yml"),
    "stereoset": resource_filename("langtest", "data/config/stereoset_config.yml"),
    "security": resource_filename("langtest", "data/config/security_config.yml"),
    "sensitivity-test": resource_filename(
        "langtest", "data/config/sensitivity_config.yml"
    ),
    "disinformation-test": {
        "huggingface-inference-api": resource_filename(
            "langtest", "data/config/disinformation_huggingface_config.yml"
        ),
        "openai": resource_filename(
            "langtest", "data/config/disinformation_openai_config.yml"
        ),
        "ai21": resource_filename(
            "langtest", "data/config/disinformation_openai_config.yml"
        ),
    },
    "factuality-test": {
        "huggingface-inference-api": resource_filename(
            "langtest", "data/config/factuality_huggingface_config.yml"
        ),
        "openai": resource_filename(
            "langtest", "data/config/factuality_openai_config.yml"
        ),
        "ai21": resource_filename("langtest", "data/config/factuality_openai_config.yml"),
    },
    "translation": {
        "huggingface": resource_filename(
            "langtest", "data/config/translation_transformers_config.yml"
        ),
        "johnsnowlabs": resource_filename(
            "langtest", "data/config/translation_johnsnowlabs_config.yml"
        ),
    },
    "sycophancy-test": {
        "huggingface-inference-api": resource_filename(
            "langtest", "data/config/sycophancy_huggingface_config.yml"
        ),
        "openai": resource_filename(
            "langtest", "data/config/sycophancy_openai_config.yml"
        ),
        "ai21": resource_filename("langtest", "data/config/sycophancy_openai_config.yml"),
    },
    "wino-bias": {
        "huggingface": resource_filename(
            "langtest", "data/config/wino_huggingface_config.yml"
        ),
        "openai": resource_filename("langtest", "data/config/wino_openai_config.yml"),
        "ai21": resource_filename("langtest", "data/config/wino_ai21_config.yml"),
    },
}
