"""Word-level tokenizer (Chapter 2).

Real models use sub-word BPE tokenizers; a word tokenizer keeps the ids readable.

Guide mapping (Chapter 2):
  * The tokenizer is not learned and holds no weights; it is built once and never changes.
  * Ids are the row numbers of the embedding table (Ch 4).
  * Guide vocabulary: I=0 love=1 tea=2 coffee=3 hate=4 rain=5 (Llama 3.1: 128,256 sub-word tokens,
    about 4 characters per token on English). Prices, limits and speed are counted in tokens.
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
        """Vocabulary = special tokens, then every distinct word in order of first appearance.

        Same as the guide's  {w: i for i, w in enumerate(dict.fromkeys(words))}.
        Specials are the chat markers of Ch 22 (harness.SPECIALS); the guide's toy has none.
        """
        words: List[str] = list(specials)
        for t in texts:
            words.extend(t.split())
        return cls(list(dict.fromkeys(words)))

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Text -> pieces -> ids:  "I love" -> ["I", "love"] -> [0, 1]  (Ch 2 table, steps 1-3)."""
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
