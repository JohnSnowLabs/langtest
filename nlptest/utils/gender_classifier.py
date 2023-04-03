import logging
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class GenderClassifier(object):
    """
    Helper model to predict the 'gender' of a piece of text.
    """

    LABELS = {"LABEL_0": "female", "LABEL_1": "male", "LABEL_2": "unknown"}
    PRETRAINED_MODEL = "microsoft/xtremedistil-l6-h256-uncased"

    def __init__(self):
        """"""
        logging.getLogger("transformers").setLevel(logging.ERROR)

        tokenizer = AutoTokenizer.from_pretrained(self.PRETRAINED_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.PRETRAINED_MODEL,
            num_labels=3
        )

        curr_dir = os.path.dirname(__file__)
        ckpt_path = os.path.join(curr_dir, 'checkpoints.ckpt')
        ckpts = torch.load(ckpt_path)
        model.load_state_dict(ckpts)
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(self, text: str) -> str:
        """
        Args:
            text (str):
                piece of text to run through the gender classifier

        Returns:
            str: predicted label
        """
        prediction = self.pipe(text, truncation=True, max_length=512)[0]["label"]
        return self.LABELS[prediction]
