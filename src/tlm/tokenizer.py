"""Word-level tokenizer (Chapter 2).

Real models use sub-word BPE tokenizers; a word tokenizer keeps the ids readable.
"""
from __future__ import annotations

import json
from typing import Iterable, List, Sequence


class WordTokenizer:
    def __init__(self, vocab: Sequence[str]):
        self.vocab: List[str] = list(vocab)
        self.ids = {w: i for i, w in enumerate(self.vocab)}
        if len(self.ids) != len(self.vocab):
            raise ValueError("duplicate tokens in vocabulary")

    @classmethod
    def build(cls, texts: Iterable[str], specials: Sequence[str] = ()) -> "WordTokenizer":
        """Vocabulary = special tokens, then every distinct word in order of first appearance."""
        words: List[str] = list(specials)
        for t in texts:
            words.extend(t.split())
        return cls(list(dict.fromkeys(words)))

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        try:
            return [self.ids[w] for w in text.split()]
        except KeyError as e:
            raise KeyError(f"unknown word {e.args[0]!r}; vocabulary is {self.vocab}") from None

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path) as f:
            return cls(json.load(f))
