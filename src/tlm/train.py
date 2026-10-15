"""Loss, gradients and the training loop (Chapters 10-12)."""
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
    """
    P = params if params is not None else model.params
    data = as_examples(data)
    n_targets = sum((len(ids) - 1) if m is None else sum(m) for ids, m in data)
    total = 0.0
    G = {k: np.zeros_like(v) for k, v in P.items()} if need_grads else None
    for ids, mask in data:
        inp, tgt = ids[:-1], ids[1:]
        logits, tr = model.forward(inp, P, keep=True)
        p = softmax(logits)
        w = np.ones(len(tgt)) if mask is None else np.array(mask, dtype=float)
        rows = np.arange(len(tgt))
        total += float(-(np.log(p[rows, tgt]) * w).sum())
        if need_grads:
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
    """Plain gradient descent: w <- w - lr * g (what the guide uses)."""
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, params: Params, grads: Params, keys=None) -> None:
        for k in (keys or params):
            params[k] -= self.lr * grads[k]


class Adam:
    """Adam: keeps two running averages per weight (why training needs ~16 bytes/weight)."""
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
    opt = optimizer or SGD(0.1)
    losses = []
    for step in range(steps + 1):
        L, G = loss_and_grads(model, data, need_grads=step < steps)
        losses.append(L)
        if callback:
            callback(step, L, model)
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  loss {L:.4f}")
        if not np.isfinite(L):
            print(f"step {step}: loss is not finite; training blew up (learning rate too high?)")
            break
        if step < steps:
            opt.step(model.params, G, keys)
    return losses
