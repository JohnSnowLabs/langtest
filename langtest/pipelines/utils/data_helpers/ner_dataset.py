from typing import List, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class NERDataset(Dataset):
    """Dataset for NER task"""

    def __init__(
        self,
        tokens: List[List[str]],
        labels: List[List[str]],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 128,
    ):
        """Constructor method

        Args:
            tokens (List[List[str]]): list of tokens per sample
            labels (List[List[str]]): list of labels per sample
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): tokenizer to use
            max_length (int): number of maximum tokens to use per sample
        """
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_map = {
            label: i
            for i, label in enumerate(
                sorted(set([lbl for doc_labels in labels for lbl in doc_labels]))
            )
        }
        self.id2label = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token_list = self.tokens[idx]
        label_list = self.labels[idx]

        encoded = self.tokenizer(
            token_list,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        token_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)

        label_ids = [self.label_map[label] for label in label_list]
        label_ids = [-100] + label_ids + [-100]  # Account for [CLS] and [SEP] tokens
        label_ids += [-100] * (self.max_length - len(label_ids))  # Pad labels

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
