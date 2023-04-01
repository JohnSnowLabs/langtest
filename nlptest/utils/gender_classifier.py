import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.utils import logging

class GenderClassifier(object):
    """Helper model to predict the 'gender' of a piece of text."""
    LABELS = {"LABEL_0": "female", "LABEL_1": "male", "LABEL_2": "unknown"}

    def __init__(self) -> None:
        logging.set_verbosity_error()

        tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/xtremedistil-l6-h256-uncased",
            num_labels=3
        )

        curr_dir = os.path.dirname(__file__)
        ckpt_path = os.path.join(curr_dir, 'checkpoints.ckpt')
        ckpts = torch.load(ckpt_path)
        model.load_state_dict(ckpts)
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(self, text: str) -> str:
        """"""
        prediction = self.pipe(text, truncation=True, max_length=512)[0]["label"]
        return self.LABELS[prediction]
