"""The tiny language model (Chapters 3-9, 13-15).

Decoder-only transformer in plain NumPy:
  embedding lookup -> [causal single-head attention -> ReLU feed-forward] x n_layers -> tied output scoring.
Residual shortcuts, no normalisation; positions are optional (the guide's model has none).
Forward, analytic backward, and KV-cached generation are all here.

Where each guide chapter lives in this file:
  Ch 3  The shape: 80 weights ............ Config, param_shapes
  Ch 4  Random birth, embedding table .... init_params, forward (X = E[ids])
  Ch 5  Dot product ...................... every `@` below
  Ch 6  Attention ........................ forward (Q, K, V, A, Z, Wo)
  Ch 7  Feed-forward ..................... forward (detect -> switch -> write)
  Ch 8  One layer: add, don't replace .... forward (the two shortcuts)
  Ch 9  Scores to percentages ............ forward (logits = H E^T), softmax, probs
  Ch 11 Backprop ......................... backward
  Ch 13 The trained model answers ........ same code, trained numbers (see guide.py)
  Ch 14 Picking the word, going again .... sample, generate_stream
  Ch 15 KV cache ......................... new_cache, step
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

Params = Dict[str, np.ndarray]


@dataclass
class Config:
    """The shape (Chapter 3): fixed before training; training changes numbers, never the shape.

    Defaults are the guide's toy. Llama 3.1 8B for comparison is in the comments
    (and in scale.LlamaShape).
    """
    vocab_size: int = 6     # Ch 2: I love tea coffee hate rain            (Llama 128,256)
    d_model: int = 4        # slots per token                              (Llama 4,096)
    d_head: int = 2         # size of Q, K, V rows                         (Llama 128, 32 heads)
    d_ff: int = 3           # feed-forward detectors                       (Llama 14,336)
    n_layers: int = 1       #                                              (Llama 32)
    max_len: int = 0        # >0 adds learned position embeddings of this length
    init_std: float = 0.5
    init_round: Optional[int] = 1   # the guide rounds starting weights to 1 decimal


def param_shapes(cfg: Config) -> Dict[str, tuple]:
    """Where the 80 weights live (Chapter 3 table), for the default config:

      E           embedding table         6 x 4      = 24
      Wq, Wk, Wv  query / key / value     3 x 4 x 2  = 24
      Wo          attention output        2 x 4      =  8
      W1, W2      detectors + write rows  3x4 + 3x4  = 24
                                                       --
                                                       80   (Llama 3.1 8B: 8.03 B)
    """
    shapes = {"E": (cfg.vocab_size, cfg.d_model)}           # one row per token (Ch 4)
    for l in range(cfg.n_layers):
        p = "" if cfg.n_layers == 1 else f"{l}."
        shapes.update({
            p + "Wq": (cfg.d_model, cfg.d_head), p + "Wk": (cfg.d_model, cfg.d_head),   # Ch 6: 4 slots -> 2
            p + "Wv": (cfg.d_model, cfg.d_head), p + "Wo": (cfg.d_head, cfg.d_model),   # Ch 6: 2 back -> 4
            p + "W1": (cfg.d_ff, cfg.d_model), p + "W2": (cfg.d_ff, cfg.d_model),       # Ch 7: detect, write
        })
    if cfg.max_len:
        shapes["P"] = (cfg.max_len, cfg.d_model)
    return shapes


def init_params(cfg: Config, seed: int = 1) -> Params:
    """Random birth (Chapter 4). Seed 1 with the default config reproduces the guide exactly.

    The guide's line is  rng.normal(0, 0.5, (6, 4)).round(1)  for E; the same draw fills every
    other matrix in param_shapes order. Rounding to 1 decimal only keeps the printed numbers short.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for k, s in param_shapes(cfg).items():
        w = rng.normal(0.0, cfg.init_std, s)
        out[k] = np.round(w, cfg.init_round) if cfg.init_round is not None else w
    return out


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scores to percentages (Chapter 9): raise e (~2.72) to each score, divide by the total.

    Used twice: on attention scores (Ch 6, giving shares like I 60% / love 40%) and on word
    scores (Ch 9). Subtracting the max first changes nothing but avoids overflow.
    """
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
        # Ch 4: the embedding table is a learned lookup; the token id is the row number.
        # "I love" -> ids [0, 1] -> rows 0 and 1 of E, one row of 4 slots per token.
        X = P["E"][ids].copy()
        if self.cfg.max_len:
            X = X + P["P"][:n]                          # optional positions (not in the guide's model)
        scale = 1.0 / np.sqrt(self.cfg.d_head)          # Ch 6: divide by sqrt(2) for our d_head = 2
        mask = np.triu(np.full((n, n), -np.inf), 1)   # causal mask: look backwards only
        layers = []
        H = X
        for l in range(self.cfg.n_layers):              # 1 layer here; Llama stacks 32 of the same
            nm = self._layer_names(l)
            Xin = H
            # Ch 6 step 1: each row asks (Q, query), advertises (K, key) and offers (V, value).
            # These are dot products (Ch 5): 4 slots in, 2 numbers out.
            Q, K, V = Xin @ P[nm["Wq"]], Xin @ P[nm["Wk"]], Xin @ P[nm["Wv"]]
            # Ch 6 step 2: Q of each token . K of every earlier token, / sqrt(d_head), then
            # softmax into shares. Row "love" gives I 60%, love 40% with the starting weights.
            # Attention is the ONLY step where tokens read each other.
            A = softmax(Q @ K.T * scale + mask)
            # Ch 6 step 3: blend the V rows by those shares.
            Z = A @ V
            # Ch 6 step 4: the output matrix Wo turns 2 numbers back into 4 slots.
            # Ch 8: then ADD it to the row (shortcut 1) - a layer adds changes, never replaces.
            Hmid = Xin + Z @ P[nm["Wo"]]                # shortcut 1
            # Ch 7 feed-forward, per token on its own: detect -> switch -> write.
            pre = Hmid @ P[nm["W1"]].T                  # detect: one dot product per detector (3)
            r = np.maximum(pre, 0.0)                    # switch (ReLU): negative -> off (0)
            H = Hmid + r @ P[nm["W2"]]                  # write: scaled sum of write rows + shortcut 2 (Ch 8)
            layers.append(dict(Xin=Xin, Q=Q, K=K, V=V, A=A, Z=Z, Hmid=Hmid, pre=pre, r=r))
        # Ch 9: score each word = dot product of the final row with that word's embedding row.
        # Tied: the same E reads words in and scores them out (Llama 3.1 8B uses a separate table).
        logits = H @ P["E"].T                           # tied output scoring (Chapter 9)
        if keep:
            return logits, dict(ids=ids, X=X, layers=layers, Hout=H)
        return logits

    def probs(self, ids: Sequence[int], temperature: float = 1.0) -> np.ndarray:
        """Next-token shares after the last token (Ch 9; temperature from Ch 14).

        Only the last row matters for "what comes next"; trained, "I love" gives tea 0.66,
        coffee 0.33 (Ch 13), close to the text's 67% / 33% (Ch 1).
        """
        return softmax(self.forward(ids)[-1] / temperature)

    # ------------------------------------------------------------------ backward
    def backward(self, trace, dlogits: np.ndarray, params: Optional[Params] = None) -> Params:
        """Analytic gradients (Chapter 11) given dLoss/dlogits for every position.

        "Blame flows back": dlogits is the blame on each word's score (share - 1 for the actual
        next word, share for the rest; see train.loss_and_grads). Each line below is the forward
        step read backwards, so every weight learns how much it pushed the loss up.
        train.numeric_grad checks these against the nudge-and-measure method the guide uses.
        """
        P = params if params is not None else self.params
        G = {k: np.zeros_like(v) for k, v in P.items()}
        ids = trace["ids"]
        H = trace["Hout"]
        scale = 1.0 / np.sqrt(self.cfg.d_head)
        # Ch 9 backwards: E gets blame twice (as the scoring table here, as the lookup at the end).
        G["E"] += dlogits.T @ H                         # logits = H E^T
        dH = dlogits @ P["E"]
        for l in reversed(range(self.cfg.n_layers)):
            nm = self._layer_names(l)
            c = trace["layers"][l]
            # feed-forward (Ch 7):  H = Hmid + relu(Hmid W1^T) W2
            G[nm["W2"]] += c["r"].T @ dH
            dpre = (dH @ P[nm["W2"]].T) * (c["pre"] > 0)   # a switched-off detector gets no blame
            G[nm["W1"]] += dpre.T @ c["Hmid"]
            dHmid = dH + dpre @ P[nm["W1"]]             # "dH +": the shortcut passes blame straight through (Ch 8)
            # attention (Ch 6):  Hmid = Xin + (softmax(Q K^T s) V) Wo
            G[nm["Wo"]] += c["Z"].T @ dHmid
            dZ = dHmid @ P[nm["Wo"]].T
            dA = dZ @ c["V"].T
            dV = c["A"].T @ dZ
            dS = c["A"] * (dA - (dA * c["A"]).sum(axis=1, keepdims=True)) * scale   # back through softmax
            dQ = dS @ c["K"]
            dK = dS.T @ c["Q"]
            G[nm["Wq"]] += c["Xin"].T @ dQ
            G[nm["Wk"]] += c["Xin"].T @ dK
            G[nm["Wv"]] += c["Xin"].T @ dV
            dH = dHmid + dQ @ P[nm["Wq"]].T + dK @ P[nm["Wk"]].T + dV @ P[nm["Wv"]].T
        np.add.at(G["E"], ids, dH)                     # embedding lookup (Ch 4): only rows used get blame
        if self.cfg.max_len:
            G["P"][: len(ids)] += dH
        return G

    # ------------------------------------------------------------------ KV-cached generation
    def new_cache(self) -> List[Dict[str, list]]:
        return [dict(K=[], V=[]) for _ in range(self.cfg.n_layers)]

    def step(self, token: int, cache: List[Dict[str, list]], pos: int) -> np.ndarray:
        """Process ONE new token, reusing cached K and V (Chapter 15). Returns its logits.

        Earlier tokens' K and V rows never change (the causal mask means they never look ahead),
        so they are kept instead of recomputed. Same maths as forward(), one row at a time, and
        the shares come out identical. The cache saves computing, not reading or memory:
        it grows by one K and one V row per token per layer (Ch 17: ~128 KiB/token for Llama 8B).
        """
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
            a = softmax(K @ q * scale)                   # Ch 6, no mask needed: the cache holds only the past
            hmid = h + (a @ V) @ P[nm["Wo"]]             # Ch 6 + shortcut (Ch 8)
            h = hmid + np.maximum(hmid @ P[nm["W1"]].T, 0.0) @ P[nm["W2"]]   # Ch 7 + shortcut
        return h @ P["E"].T                              # Ch 9 tied scoring

    def generate_stream(self, prompt: Sequence[int], max_new: int = 20, temperature: float = 1.0,
                        top_p: float = 1.0, stop: Optional[Sequence[int]] = None,
                        rng: Optional[np.random.Generator] = None) -> Iterator[int]:
        """Yield tokens one at a time (Chapter 14): prefill the prompt, then sample + append."""
        rng = rng or np.random.default_rng()
        cache = self.new_cache()
        logits = None
        for pos, t in enumerate(prompt):                 # prefill: read the prompt, fill the cache
            logits = self.step(t, cache, pos)
        pos = len(prompt)
        for _ in range(max_new):                         # decode: "pick the word, then go again"
            tok = sample(logits, temperature, top_p, rng)
            yield tok                                    # streaming: each word shows as it is picked
            if stop and tok in stop:
                return
            if self.cfg.max_len and pos >= self.cfg.max_len:
                return
            logits = self.step(tok, cache, pos)          # append the picked word, run again (one new row)
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
        return int(np.argmax(logits))                # greedy: always the top word ("tea")
    rng = rng or np.random.default_rng()
    # Temperature divides the scores before softmax: below 1 sharpens, above 1 flattens.
    p = softmax(logits / temperature)
    if top_p < 1.0:
        # Top-p: keep the smallest set of top words whose shares reach p, renormalise, drop the tail.
        order = np.argsort(-p)
        keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p) + 1)]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(rng.choice(len(p), p=p))              # sample: tea about 2 times in 3, coffee 1 in 3
