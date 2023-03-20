import torch
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class GenderClassifier():
    def __init__(self) -> None:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/xtremedistil-l6-h256-uncased", num_labels=3)
        ckpts = torch.load("checkpoints.ckpt")
        model.load_state_dict(ckpts)
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(self, text: str):
        pred = self.pipe(text)[0]["label"]
        if pred == "LABEL_0":
            return "female"
        if pred == "LABEL_1":
            return "male"

        return "unknown" 