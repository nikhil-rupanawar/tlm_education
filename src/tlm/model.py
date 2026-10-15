"""The tiny language model (Chapters 3-9, 13-15).

Decoder-only transformer in plain NumPy:
  embedding lookup -> [causal single-head attention -> ReLU feed-forward] x n_layers -> tied output scoring.
Residual shortcuts, no normalisation; positions are optional (the guide's model has none).
Forward, analytic backward, and KV-cached generation are all here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

Params = Dict[str, np.ndarray]


@dataclass
class Config:
    vocab_size: int = 6
    d_model: int = 4        # slots per token
    d_head: int = 2         # size of Q, K, V rows
    d_ff: int = 3           # feed-forward detectors
    n_layers: int = 1
    max_len: int = 0        # >0 adds learned position embeddings of this length
    init_std: float = 0.5
    init_round: Optional[int] = 1   # the guide rounds starting weights to 1 decimal


def param_shapes(cfg: Config) -> Dict[str, tuple]:
    shapes = {"E": (cfg.vocab_size, cfg.d_model)}
    for l in range(cfg.n_layers):
        p = "" if cfg.n_layers == 1 else f"{l}."
        shapes.update({
            p + "Wq": (cfg.d_model, cfg.d_head), p + "Wk": (cfg.d_model, cfg.d_head),
            p + "Wv": (cfg.d_model, cfg.d_head), p + "Wo": (cfg.d_head, cfg.d_model),
            p + "W1": (cfg.d_ff, cfg.d_model), p + "W2": (cfg.d_ff, cfg.d_model),
        })
    if cfg.max_len:
        shapes["P"] = (cfg.max_len, cfg.d_model)
    return shapes


def init_params(cfg: Config, seed: int = 1) -> Params:
    """Random birth (Chapter 4). Seed 1 with the default config reproduces the guide exactly."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, s in param_shapes(cfg).items():
        w = rng.normal(0.0, cfg.init_std, s)
        out[k] = np.round(w, cfg.init_round) if cfg.init_round is not None else w
    return out


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class TLM:
    def __init__(self, cfg: Config, params: Optional[Params] = None, seed: int = 1):
        self.cfg = cfg
        self.params: Params = params if params is not None else init_params(cfg, seed)

    # ------------------------------------------------------------------ helpers
    def _layer_names(self, l: int) -> Dict[str, str]:
        p = "" if self.cfg.n_layers == 1 else f"{l}."
        return {k: p + k for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2")}

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def copy(self) -> "TLM":
        return TLM(self.cfg, {k: v.copy() for k, v in self.params.items()})

    # ------------------------------------------------------------------ forward
    def forward(self, ids: Sequence[int], params: Optional[Params] = None, keep: bool = False):
        """Full forward pass over a sequence. Returns logits (n, vocab) and, if keep, a trace."""
        P = params if params is not None else self.params
        ids = list(ids)
        n = len(ids)
        if self.cfg.max_len and n > self.cfg.max_len:
            raise ValueError(f"sequence of {n} tokens exceeds max_len {self.cfg.max_len}")
        X = P["E"][ids].copy()
        if self.cfg.max_len:
            X = X + P["P"][:n]
        scale = 1.0 / np.sqrt(self.cfg.d_head)
        mask = np.triu(np.full((n, n), -np.inf), 1)   # causal mask: look backwards only
        layers = []
        H = X
        for l in range(self.cfg.n_layers):
            nm = self._layer_names(l)
            Xin = H
            Q, K, V = Xin @ P[nm["Wq"]], Xin @ P[nm["Wk"]], Xin @ P[nm["Wv"]]
            A = softmax(Q @ K.T * scale + mask)
            Z = A @ V
            Hmid = Xin + Z @ P[nm["Wo"]]                # shortcut 1
            pre = Hmid @ P[nm["W1"]].T                  # detect
            r = np.maximum(pre, 0.0)                    # switch (ReLU)
            H = Hmid + r @ P[nm["W2"]]                  # write + shortcut 2
            layers.append(dict(Xin=Xin, Q=Q, K=K, V=V, A=A, Z=Z, Hmid=Hmid, pre=pre, r=r))
        logits = H @ P["E"].T                           # tied output scoring (Chapter 9)
        if keep:
            return logits, dict(ids=ids, X=X, layers=layers, Hout=H)
        return logits

    def probs(self, ids: Sequence[int], temperature: float = 1.0) -> np.ndarray:
        """Next-token shares after the last token."""
        return softmax(self.forward(ids)[-1] / temperature)

    # ------------------------------------------------------------------ backward
    def backward(self, trace, dlogits: np.ndarray, params: Optional[Params] = None) -> Params:
        """Analytic gradients (Chapter 11) given dLoss/dlogits for every position."""
        P = params if params is not None else self.params
        G = {k: np.zeros_like(v) for k, v in P.items()}
        ids = trace["ids"]
        H = trace["Hout"]
        scale = 1.0 / np.sqrt(self.cfg.d_head)
        G["E"] += dlogits.T @ H                         # logits = H E^T
        dH = dlogits @ P["E"]
        for l in reversed(range(self.cfg.n_layers)):
            nm = self._layer_names(l)
            c = trace["layers"][l]
            # feed-forward:  H = Hmid + relu(Hmid W1^T) W2
            G[nm["W2"]] += c["r"].T @ dH
            dpre = (dH @ P[nm["W2"]].T) * (c["pre"] > 0)
            G[nm["W1"]] += dpre.T @ c["Hmid"]
            dHmid = dH + dpre @ P[nm["W1"]]
            # attention:  Hmid = Xin + (softmax(Q K^T s) V) Wo
            G[nm["Wo"]] += c["Z"].T @ dHmid
            dZ = dHmid @ P[nm["Wo"]].T
            dA = dZ @ c["V"].T
            dV = c["A"].T @ dZ
            dS = c["A"] * (dA - (dA * c["A"]).sum(axis=1, keepdims=True)) * scale
            dQ = dS @ c["K"]
            dK = dS.T @ c["Q"]
            G[nm["Wq"]] += c["Xin"].T @ dQ
            G[nm["Wk"]] += c["Xin"].T @ dK
            G[nm["Wv"]] += c["Xin"].T @ dV
            dH = dHmid + dQ @ P[nm["Wq"]].T + dK @ P[nm["Wk"]].T + dV @ P[nm["Wv"]].T
        np.add.at(G["E"], ids, dH)                     # embedding lookup
        if self.cfg.max_len:
            G["P"][: len(ids)] += dH
        return G

    # ------------------------------------------------------------------ KV-cached generation
    def new_cache(self) -> List[Dict[str, list]]:
        return [dict(K=[], V=[]) for _ in range(self.cfg.n_layers)]

    def step(self, token: int, cache: List[Dict[str, list]], pos: int) -> np.ndarray:
        """Process ONE new token, reusing cached K and V (Chapter 15). Returns its logits."""
        P = self.params
        x = P["E"][token].copy()
        if self.cfg.max_len:
            if pos >= self.cfg.max_len:
                raise ValueError("context full: max_len reached")
            x = x + P["P"][pos]
        scale = 1.0 / np.sqrt(self.cfg.d_head)
        h = x
        for l in range(self.cfg.n_layers):
            nm = self._layer_names(l)
            q, k, v = h @ P[nm["Wq"]], h @ P[nm["Wk"]], h @ P[nm["Wv"]]
            cache[l]["K"].append(k)
            cache[l]["V"].append(v)
            K, V = np.array(cache[l]["K"]), np.array(cache[l]["V"])
            a = softmax(K @ q * scale)
            hmid = h + (a @ V) @ P[nm["Wo"]]
            h = hmid + np.maximum(hmid @ P[nm["W1"]].T, 0.0) @ P[nm["W2"]]
        return h @ P["E"].T

    def generate_stream(self, prompt: Sequence[int], max_new: int = 20, temperature: float = 1.0,
                        top_p: float = 1.0, stop: Optional[Sequence[int]] = None,
                        rng: Optional[np.random.Generator] = None) -> Iterator[int]:
        """Yield tokens one at a time (Chapter 14): prefill the prompt, then sample + append."""
        rng = rng or np.random.default_rng()
        cache = self.new_cache()
        logits = None
        for pos, t in enumerate(prompt):                 # prefill
            logits = self.step(t, cache, pos)
        pos = len(prompt)
        for _ in range(max_new):
            tok = sample(logits, temperature, top_p, rng)
            yield tok
            if stop and tok in stop:
                return
            if self.cfg.max_len and pos >= self.cfg.max_len:
                return
            logits = self.step(tok, cache, pos)
            pos += 1

    def generate(self, prompt: Sequence[int], **kw) -> List[int]:
        return list(self.generate_stream(prompt, **kw))

    # ------------------------------------------------------------------ save / load
    def save(self, path: str) -> None:
        np.savez(path, __cfg__=np.array(list(asdict(self.cfg).items()), dtype=object), **self.params)

    @classmethod
    def load(cls, path: str) -> "TLM":
        z = np.load(path, allow_pickle=True)
        cfg = Config(**dict(z["__cfg__"].tolist()))
        return cls(cfg, {k: z[k] for k in z.files if k != "__cfg__"})


def sample(logits: np.ndarray, temperature: float = 1.0, top_p: float = 1.0,
           rng: Optional[np.random.Generator] = None) -> int:
    """Greedy if temperature == 0, else temperature + top-p sampling (Chapter 14)."""
    if temperature == 0:
        return int(np.argmax(logits))
    rng = rng or np.random.default_rng()
    p = softmax(logits / temperature)
    if top_p < 1.0:
        order = np.argsort(-p)
        keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p) + 1)]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(rng.choice(len(p), p=p))
