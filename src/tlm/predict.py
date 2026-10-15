"""Next-token prediction demo: python -m tlm predict "I love"

Chapters 1-14 end to end: tokenize (2), random birth (4), the BEFORE bars (Stage 2, first run),
train (10-12), the AFTER bars (13: tea ~66%, coffee ~33%), then greedy vs sampled picks (14).
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

from .model import TLM, Config
from .tokenizer import WordTokenizer
from .train import SGD, train

DEFAULT_TEXT = ["I love tea", "I love tea", "I love coffee", "I hate rain"]   # Ch 1's training text


def bars(model: TLM, tok: WordTokenizer, prompt: str, title: str, temperature: float = 1.0) -> str:
    p = model.probs(tok.encode(prompt), temperature=temperature)
    lines = [f"\n{title}: '{prompt}' -> next token"]
    for w, x in sorted(zip(tok.vocab, p), key=lambda t: -t[1]):
        lines.append(f"  {w:8s} {x * 100:5.1f}%  {'#' * int(x * 40)}")
    return "\n".join(lines)


def run(prompt: str = "I love", text: Optional[List[str]] = None, steps: int = 500, lr: float = 0.1,
        samples: int = 12, temperature: float = 1.0, seed: int = 1) -> str:
    text = text or DEFAULT_TEXT
    tok = WordTokenizer.build(text)
    model = TLM(Config(vocab_size=len(tok)), seed=seed)
    out = [f"training text: {text}", f"vocabulary: {tok.vocab}   weights: {model.num_params()}"]
    out.append(bars(model, tok, prompt, "BEFORE training (random weights)"))
    train(model, [tok.encode(s) for s in text], steps=steps, optimizer=SGD(lr))
    out.append(bars(model, tok, prompt, f"AFTER {steps} training steps"))
    if temperature != 1.0:
        out.append(bars(model, tok, prompt, f"AFTER training, temperature {temperature}", temperature))
    ids = tok.encode(prompt)
    greedy = tok.decode(model.generate(ids, max_new=1, temperature=0))
    rng = np.random.default_rng(0)
    picks = [tok.decode(model.generate(ids, max_new=1, temperature=temperature, rng=rng)) for _ in range(samples)]
    out.append(f"\nGreedy (always the top token): {greedy}")
    out.append(f"Sampled {samples} times (temperature {temperature}): {' '.join(picks)}")
    cont = tok.decode(model.generate(ids, max_new=3, temperature=0))
    out.append(f"Greedy continuation, 3 tokens: {prompt} {cont}")
    out.append("  (tokens past where the training sentences end are confident guesses from situations\n"
               "   the model never trained on: the 'rain 95%' effect from Chapter 14)")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tlm predict",
                                 description="Train the tiny model and show its next-token prediction.")
    ap.add_argument("prompt", nargs="?", default="I love", help="words from the training text")
    ap.add_argument("--text", help="file with one training sentence per line (default: the guide's 4 sentences)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    text = None
    if a.text:
        with open(a.text) as f:
            text = [l.strip() for l in f if l.strip()]
    try:
        print(run(a.prompt, text, a.steps, a.lr, a.samples, a.temperature, a.seed))
    except KeyError as e:
        print(f"error: {e.args[0]}")


if __name__ == "__main__":
    main()
