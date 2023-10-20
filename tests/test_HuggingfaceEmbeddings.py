import unittest
import torch
import numpy as np
from langtest.embeddings.huggingface import HuggingfaceEmbeddings


class TestHuggingfaceEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embeddings = HuggingfaceEmbeddings("bert-base-uncased")
        self.single_sentence = "This is a test sentence."
        self.multiple_sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        self.long_sentence = (
            "This is a very long sentence that should be truncated as soon as possible."
        )
        self.empty_sentence = ""

    def test_single_sentence_embedding(self):
        """Test getting embedding for a single sentence."""
        embedding = self.embeddings.get_embedding(
            self.single_sentence, convert_to_tensor=True
        )
        self.assertTrue(isinstance(embedding, torch.Tensor))
        self.assertEqual(embedding.shape, (1, 768))

    def test_multiple_sentences_embeddings(self):
        """Test getting embeddings for multiple sentences."""
        embeddings = self.embeddings.get_embedding(
            self.multiple_sentences, convert_to_tensor=False
        )
        self.assertTrue(isinstance(embeddings, np.ndarray))
        self.assertEqual(len(embeddings), len(self.multiple_sentences))

    def test_max_length_truncation(self):
        """Test truncation of sentences longer than max_length."""
        max_length = 2
        embedding = self.embeddings.get_embedding(
            self.long_sentence, convert_to_tensor=True, max_length=max_length
        )
        self.assertTrue(isinstance(embedding, torch.Tensor))
        self.assertEqual(embedding.shape, (1, 768))

    def test_empty_input(self):
        """Test getting embedding for an empty input."""
        embedding = self.embeddings.get_embedding(
            self.empty_sentence, convert_to_tensor=True
        )
        self.assertTrue(isinstance(embedding, torch.Tensor))
        self.assertEqual(embedding.shape, (1, 768))
