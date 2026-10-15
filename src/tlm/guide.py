"""Reproduce every tiny-model number in the guide 'Arithmetic Intelligence'."""
from __future__ import annotations

import numpy as np

from .finetune import LoRA, finetune
from .model import TLM, Config
from .rl import kl_optimum, rl_next_token_exact
from .tokenizer import WordTokenizer
from .train import SGD, loss_and_grads, numeric_grad, train

CORPUS = ["I love tea", "I love tea", "I love coffee", "I hate rain"]
FINETUNE_TEXT = ["I love coffee"] * 3


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def setup():
    tok = WordTokenizer.build(CORPUS)
    data = [tok.encode(s) for s in CORPUS]
    model = TLM(Config(vocab_size=len(tok)), seed=1)
    return tok, data, model


def pct(x):
    return f"{100 * x:5.1f}%"


def run(verbose: bool = True) -> dict:
    out = {}
    say = print if verbose else (lambda *a, **k: None)
    tok, data, model = setup()
    I, love, tea, coffee, hate, rain = (tok.ids[w] for w in ["I", "love", "tea", "coffee", "hate", "rain"])

    say("== Stage 1: design ==")
    say(f"vocabulary {tok.vocab}   weights {model.num_params()}")

    say("\n== Stage 2: first run (random weights) ==")
    p0 = model.probs([I, love])
    say("I love -> " + "  ".join(f"{w} {pct(p)}" for w, p in zip(tok.vocab, p0)))
    out["newborn_tea"] = float(p0[tea])

    say("\n== Stage 3: train (SGD, lr 0.1, 500 steps) ==")
    L0, G = loss_and_grads(model, data)
    k = model._layer_names(0)["W2"]
    num = numeric_grad(lambda: loss_and_grads(model, data, need_grads=False)[0], model.params[k])
    say(f"gradient check on {k}: max |analytic - numeric| = {np.abs(G[k] - num).max():.2e}")

    def cb(step, L, m):
        if step in (0, 10, 25, 50, 100, 150, 200, 300, 500):
            p = m.probs([I, love]); q = m.probs([I])
            say(f"step {step:3d}  loss {L:.3f}  tea {pct(p[tea])} coffee {pct(p[coffee])} rain {pct(p[rain])}"
                f"   I->love {pct(q[love])} hate {pct(q[hate])}   cos(tea,coffee) {cos(m.params['E'][tea], m.params['E'][coffee]):.2f}")
    losses = train(model, data, steps=500, optimizer=SGD(0.1), callback=cb)
    out["loss_final"] = losses[-1]
    p = model.probs([I, love])
    out["trained_tea"], out["trained_coffee"] = float(p[tea]), float(p[coffee])

    say("\n== Stage 4: use ==")
    for T in (0.5, 1.0, 2.0):
        pt = model.probs([I, love], temperature=T)
        say(f"temperature {T}: tea {pct(pt[tea])} coffee {pct(pt[coffee])} rain {pct(pt[rain])}")
    p3 = model.probs([I, love, tea])
    say(f"after 'I love tea' (never trained): rain {pct(p3[rain])}")
    out["after_tea_rain"] = float(p3[rain])
    cache = model.new_cache()
    for pos, t in enumerate([I, love, tea]):
        cached = model.step(t, cache, pos)
    same = np.allclose(cached, model.forward([I, love, tea])[-1])
    say(f"KV cache gives identical logits to full recompute: {same}")
    out["kv_same"] = bool(same)
    trained = model.copy()

    say("\n== Stage 6a: full fine-tune on 'I love coffee' x3 ==")
    ft_data = [tok.encode(s) for s in FINETUNE_TEXT]
    m = trained.copy()
    def cbf(step, L, mm):
        if step in (0, 5, 10, 20, 40):
            q = mm.probs([I]); pp = mm.probs([I, love])
            old = loss_and_grads(mm, data, need_grads=False)[0]
            say(f"step {step:2d}  coffee {pct(pp[coffee])}  I->hate {pct(q[hate])}  loss on original text {old:.3f}")
    finetune(m, ft_data, steps=40, callback=cbf)

    say("\n== Stage 6b: LoRA rank 1 on W2 (7 trainable numbers) ==")
    lora = LoRA(trained.copy(), targets=[k], rank=1, seed=0)
    def cbl(step, L, lo):
        if step in (0, 5, 10, 20, 40):
            mm = lo.merged_model(); q = mm.probs([I]); pp = mm.probs([I, love])
            old = loss_and_grads(mm, data, need_grads=False)[0]
            say(f"step {step:2d}  coffee {pct(pp[coffee])}  I->hate {pct(q[hate])}  loss on original text {old:.3f}")
    lora.train(ft_data, steps=40, lr=0.1, callback=cbl)

    say("\n== Stage 6c: RL, reward coffee = 1 ==")
    r = np.zeros(len(tok)); r[coffee] = 1.0
    for beta in (0.0, 0.5):
        mm = trained.copy()
        h = rl_next_token_exact(mm, [I, love], r, beta=beta, steps=300, lr=0.1)
        old = loss_and_grads(mm, data, need_grads=False)[0]
        theory = "" if beta == 0 else f"   theory {pct(kl_optimum(trained.probs([I, love]), r, beta)[coffee])}"
        say(f"beta {beta}: coffee {pct(h[-1]['p'][coffee])}  KL {h[-1]['kl']:.3f}  loss on original text {old:.3f}{theory}")
        out[f"rl_beta_{beta}"] = float(h[-1]["p"][coffee])
    return out


if __name__ == "__main__":
    run()
