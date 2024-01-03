DEFAULT_CONFIG = {
    "evaluation": {
        "metric": "llm_eval",
        "model": "gpt-3.5-turbo-instruct",
        "hub": "openai",
    },
    "tests": {
        "defaults": {"min_pass_rate": 0.65},
        "robustness": {
            "uppercase": {"min_pass_rate": 0.75},
            "lowercase": {"min_pass_rate": 0.75},
            "titlecase": {"min_pass_rate": 0.75},
            "add_typo": {"min_pass_rate": 0.75},
            "dyslexia_word_swap": {"min_pass_rate": 0.75},
            "add_abbreviation": {"min_pass_rate": 0.75},
            "add_slangs": {"min_pass_rate": 0.75},
            "add_speech_to_text_typo": {"min_pass_rate": 0.75},
            "add_ocr_typo": {"min_pass_rate": 0.75},
            "adjective_synonym_swap": {"min_pass_rate": 0.75},
        },
    },
}

BENCHMARK_DATASETS = [
    "BoolQ",
    "NQ-open",
    "LogiQA",
    "OpenBookQA",
]
