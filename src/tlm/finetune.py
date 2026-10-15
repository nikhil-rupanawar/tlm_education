"""Fine-tuning and LoRA (Chapter 19)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .model import TLM, Params
from .train import SGD, loss_and_grads, train


def finetune(model: TLM, data: Sequence, steps: int = 40, optimizer=None, **kw) -> List[float]:
    """Full fine-tuning: the same training loop, starting from trained weights."""
    return train(model, data, steps=steps, optimizer=optimizer or SGD(0.1), **kw)


class LoRA:
    """Low-rank adaptation: W_used = W_frozen + A @ B, train only A and B.

    B starts at zero, so the adapted model equals the base model at step 0 (as in Hu et al. 2021).
    """

    def __init__(self, base: TLM, targets: Sequence[str] = ("W2",), rank: int = 1,
                 seed: int = 0, init_std: float = 0.5, round_to: Optional[int] = 1):
        self.base = base
        self.targets = list(targets)
        rng = np.random.default_rng(seed)
        self.A: Dict[str, np.ndarray] = {}
        self.B: Dict[str, np.ndarray] = {}
        for t in self.targets:
            rows, cols = base.params[t].shape
            a = rng.normal(0.0, init_std, (rows, rank))
            self.A[t] = np.round(a, round_to) if round_to is not None else a
            self.B[t] = np.zeros((rank, cols))

    def num_trainable(self) -> int:
        return int(sum(self.A[t].size + self.B[t].size for t in self.targets))

    def merged_params(self) -> Params:
        P = {k: v for k, v in self.base.params.items()}
        for t in self.targets:
            P[t] = self.base.params[t] + self.A[t] @ self.B[t]
        return P

    def merged_model(self) -> TLM:
        return TLM(self.base.cfg, {k: v.copy() for k, v in self.merged_params().items()})

    def train(self, data: Sequence, steps: int = 40, lr: float = 0.1, callback=None) -> List[float]:
        losses = []
        for step in range(steps + 1):
            P = self.merged_params()
            L, G = loss_and_grads(self.base, data, P, need_grads=step < steps)
            losses.append(L)
            if callback:
                callback(step, L, self)
            if step == steps:
                break
            for t in self.targets:                       # chain rule through W = W0 + A B
                dA = G[t] @ self.B[t].T
                dB = self.A[t].T @ G[t]
                self.A[t] -= lr * dA
                self.B[t] -= lr * dB
        return losses
