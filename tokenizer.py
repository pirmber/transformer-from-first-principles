"""
tokenizer.py
------------
A minimal character-level tokenizer implemented from scratch (no external
tokenizer libraries).

Vocabulary is built directly from the training corpus, so every character
that appears in the text gets a unique integer id.
"""

from typing import List


class CharTokenizer:
    def __init__(self, corpus: str):
        chars = sorted(set(corpus))
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        """text -> list of token ids"""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """list of token ids -> text"""
        return "".join(self.itos[i] for i in ids)
