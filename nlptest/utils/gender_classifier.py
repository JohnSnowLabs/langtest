import logging
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class GenderClassifier(object):
    def __init__(self) -> None:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/xtremedistil-l6-h256-uncased",
                                                                   num_labels=3)

        curr_dir = os.path.dirname(__file__)
        ckpt_path = os.path.join(curr_dir, 'checkpoints.ckpt')
        ckpts = torch.load(ckpt_path)
        model.load_state_dict(ckpts)
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(self, text: str):
        """"""
        pred = self.pipe(text, truncation=True, max_length=512)[0]["label"]
        if pred == "LABEL_0":
            return "female"
        if pred == "LABEL_1":
            return "male"

        return "unknown"
