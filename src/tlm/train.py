"""Loss, gradients and the training loop (Chapters 10-12).

Guide mapping:
  Ch 10 Loss ............... loss_and_grads (the -log line)
  Ch 11 Backprop ........... loss_and_grads (share - 1), TLM.backward, numeric_grad (the check)
  Ch 12 Gradient descent ... SGD (the guide's: lr 0.1, 500 steps), Adam (what Llama used), train
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import TLM, Params, softmax

Example = Tuple[List[int], Optional[List[bool]]]   # token ids, optional per-target mask


def as_examples(seqs: Sequence) -> List[Example]:
    out = []
    for s in seqs:
        if isinstance(s, tuple):
            out.append((list(s[0]), None if s[1] is None else list(s[1])))
        else:
            out.append((list(s), None))
    return out


def loss_and_grads(model: TLM, data: Sequence, params: Optional[Params] = None,
                   need_grads: bool = True) -> Tuple[float, Optional[Params]]:
    """Cross-entropy averaged over every next-token prediction in the data.

    Each example is a list of token ids, or (ids, mask) where mask[t] says whether the
    prediction of token t+1 counts (used to train only on assistant replies in chat).

    The guide's 4 sentences give 8 prediction tasks (Ch 1): each word predicts the next.
    """
    P = params if params is not None else model.params
    data = as_examples(data)
    n_targets = sum((len(ids) - 1) if m is None else sum(m) for ids, m in data)
    total = 0.0
    G = {k: np.zeros_like(v) for k, v in P.items()} if need_grads else None
    for ids, mask in data:
        inp, tgt = ids[:-1], ids[1:]                    # "I love tea": I -> love, love -> tea
        logits, tr = model.forward(inp, P, keep=True)
        p = softmax(logits)
        w = np.ones(len(tgt)) if mask is None else np.array(mask, dtype=float)
        rows = np.arange(len(tgt))
        # Ch 10: loss = -ln(share given to the word that actually came next). 0 if it gave 100%;
        # uniform guessing over 6 words gives ln 6 = 1.79 (over Llama's 128,256: ~11.8).
        # The best possible for our text is not 0: "I love" is tea 67% / coffee 33% (Ch 1).
        total += float(-(np.log(p[rows, tgt]) * w).sum())
        if need_grads:
            # Ch 11: blame on each word's score = its share, minus 1 for the actual next word.
            # (That is d loss / d score, for softmax followed by -ln.)
            d = p.copy()
            d[rows, tgt] -= 1.0
            d *= (w / n_targets)[:, None]
            g = model.backward(tr, d, P)
            for k in G:
                G[k] += g[k]
    return total / n_targets, G


def numeric_grad(f: Callable[[], float], arr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Nudge each number up and down and measure (how the guide verifies backprop)."""
    g = np.zeros_like(arr)
    for i in np.ndindex(arr.shape):
        old = arr[i]
        arr[i] = old + eps; up = f()
        arr[i] = old - eps; down = f()
        arr[i] = old
        g[i] = (up - down) / (2 * eps)
    return g


class SGD:
    """Plain gradient descent: w <- w - lr * g (what the guide uses).

    Ch 12: new weight = old weight - learning rate x gradient. Too big a rate and the loss blows up
    (train() stops on a non-finite loss); too small and 500 steps isn't enough.
    """
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, params: Params, grads: Params, keys=None) -> None:
        for k in (keys or params):
            params[k] -= self.lr * grads[k]


class Adam:
    """Adam: keeps two running averages per weight (why training needs ~16 bytes/weight).

    Ch 12 / Ch 17: Llama 3.1 used AdamW at lr 3e-4. m and v are the two extra numbers per weight;
    with an fp32 master copy that is 2 + 2 + 12 = 16 bytes/weight (scale.train_memory_bytes).
    weight_decay is the "W" in AdamW.
    """
    def __init__(self, lr: float = 0.01, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, weight_decay
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: Params, grads: Params, keys=None) -> None:
        self.t += 1
        for k in (keys or params):
            g = grads[k]
            m = self.m.setdefault(k, np.zeros_like(g))
            v = self.v.setdefault(k, np.zeros_like(g))
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            params[k] -= self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * params[k])


def train(model: TLM, data: Sequence, steps: int = 500, optimizer=None,
          log_every: int = 0, callback: Optional[Callable[[int, float, TLM], None]] = None,
          keys: Optional[Sequence[str]] = None) -> List[float]:
    """Full-batch training loop: measure -> blame -> nudge, repeated (Chapter 12).

    callback(step, loss, model) is called before each update and once after the last one.
    keys restricts which weights move (everything else stays frozen).
    """
    opt = optimizer or SGD(0.1)                         # the guide's run: SGD, lr 0.1, 500 steps
    losses = []
    for step in range(steps + 1):
        L, G = loss_and_grads(model, data, need_grads=step < steps)   # measure (Ch 10) + blame (Ch 11)
        losses.append(L)
        if callback:
            callback(step, L, model)
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  loss {L:.4f}")
        if not np.isfinite(L):
            print(f"step {step}: loss is not finite; training blew up (learning rate too high?)")
            break
        if step < steps:
            opt.step(model.params, G, keys)              # nudge (Ch 12)
    return losses
