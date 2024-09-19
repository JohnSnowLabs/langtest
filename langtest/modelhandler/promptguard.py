class PromptGuard:
    _instance = None

    def __new__(cls, model_name: str = "meta-llama/Prompt-Guard-86M", device="cpu"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.device = device
            (
                cls._instance.model,
                cls._instance.tokenizer,
            ) = cls._instance._load_model_and_tokenizer()
        return cls._instance

    def __init__(
        self, model_name: str = "meta-llama/Prompt-Guard-86M", device="cpu"
    ) -> None:
        self.model_name = "meta-llama/Prompt-Guard-86M"
        self.device = "cpu"
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer from Hugging Face.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(
            self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def _preprocess_text(self, text):
        """
        Preprocess the input text by removing spaces to mitigate prompt injection tactics.
        """
        cleaned_text = "".join([char for char in text if not char.isspace()])
        tokens = self.tokenizer.tokenize(cleaned_text)
        result = " ".join(
            [self.tokenizer.convert_tokens_to_string([token]) for token in tokens]
        )
        return result or text

    def _get_class_probabilities(self, texts, temperature=1.0, preprocess=True):
        """
        Internal method to get class probabilities for a single or batch of texts.
        """
        import torch
        from torch.nn.functional import softmax

        if preprocess:
            texts = [self._preprocess_text(text) for text in texts]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = softmax(logits / temperature, dim=-1)
        return probabilities

    def get_jailbreak_score(self, text, temperature=1.0, preprocess=True):
        """
        Get jailbreak score for a single input text.
        """
        probabilities = self._get_class_probabilities([text], temperature, preprocess)
        return probabilities[0, 2].item()

    def get_indirect_injection_score(self, text, temperature=1.0, preprocess=True):
        """
        Get indirect injection score for a single input text.
        """
        probabilities = self._get_class_probabilities([text], temperature, preprocess)
        return (probabilities[0, 1] + probabilities[0, 2]).item()

    def _process_text_batch(
        self, texts, score_indices, temperature=1.0, max_batch_size=16, preprocess=True
    ):
        """
        Internal method to process texts in batches and return scores.
        """
        import torch

        num_texts = len(texts)
        all_scores = torch.zeros(num_texts)

        for i in range(0, num_texts, max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            probabilities = self._get_class_probabilities(
                batch_texts, temperature, preprocess
            )
            batch_scores = probabilities[:, score_indices].sum(dim=1).cpu()

            all_scores[i : i + max_batch_size] = batch_scores

        return all_scores.tolist()

    def get_jailbreak_scores_for_texts(
        self, texts, temperature=1.0, max_batch_size=16, preprocess=True
    ):
        """
        Get jailbreak scores for a batch of texts.
        """
        return self._process_text_batch(
            texts,
            score_indices=[2],
            temperature=temperature,
            max_batch_size=max_batch_size,
            preprocess=preprocess,
        )

    def get_indirect_injection_scores_for_texts(
        self, texts, temperature=1.0, max_batch_size=16, preprocess=True
    ):
        """
        Get indirect injection scores for a batch of texts.
        """
        return self._process_text_batch(
            texts,
            score_indices=[1, 2],
            temperature=temperature,
            max_batch_size=max_batch_size,
            preprocess=preprocess,
        )
