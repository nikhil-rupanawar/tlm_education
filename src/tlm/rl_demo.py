"""RL demo (Chapter 21): python -m tlm rl

Use case: a café owns the model trained on the guide's four sentences (Ch 1). It wants the model
to recommend coffee more, and never to say "hate" - without writing any new training text.
RL replaces "compare with the text" by "try answers, score them with a rule, push up the good ones".

Part 1, exact (the guide's experiment): one decision after "I love", reward coffee = 1.
  Shows the beta trade-off: beta 0 maximises the reward and wrecks the original text;
  beta > 0 stops where theory says, new share ∝ old share x e^(reward / beta).
Part 2, sampled (how real RL runs, GRPO-style): from "I", sample a group of 2-word answers,
  score each with the rule, compare to the group average, push up / down, with a KL penalty.

  python -m tlm rl                                  # coffee +1, hate -1, beta 0 and 0.5 (the guide's)
  python -m tlm rl --reward tea --beta 0.2 1         # any word from the vocabulary
  python -m tlm rl --penalize ""  --steps 60        # no penalty word; longer sampled run
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

from .model import TLM, Config
from .rl import group_advantages, kl, kl_optimum, rl_next_token_exact, rl_sampled
from .tokenizer import WordTokenizer
from .train import SGD, loss_and_grads, train

CORPUS = ["I love tea", "I love tea", "I love coffee", "I hate rain"]


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def shares(model: TLM, tok: WordTokenizer, prompt: List[int], words: List[str]) -> str:
    p = model.probs(prompt)
    return "  ".join(f"{w} {pct(p[tok.ids[w]])}" for w in words)


def run(reward_word: str = "coffee", penalize: Optional[str] = "hate", betas=(0.0, 0.5),
        steps: int = 30, group: int = 8, seed: int = 1) -> str:
    tok = WordTokenizer.build(CORPUS)
    for w in filter(None, (reward_word, penalize)):
        if w not in tok.ids:
            raise KeyError(f"unknown word {w!r}; vocabulary is {tok.vocab}")
    data = [tok.encode(s) for s in CORPUS]
    base = TLM(Config(vocab_size=len(tok)), seed=seed)
    train(base, data, steps=500, optimizer=SGD(0.1))           # the pre-trained model (Ch 12-13)
    I, love = tok.ids["I"], tok.ids["love"]
    show = ["tea", "coffee", "rain"] if reward_word in ("tea", "coffee", "rain") else [reward_word, "tea", "coffee"]
    out = [f"vocabulary {tok.vocab}",
           f"reward rule: '{reward_word}' = +1" + (f", '{penalize}' = -1" if penalize else "") + ", else 0",
           "",
           "== Part 1: exact, one decision after 'I love' (guide Stage 6c) ==",
           f"before RL      {shares(base, tok, [I, love], show)}   loss on original text "
           f"{loss_and_grads(base, data, need_grads=False)[0]:.3f}"]

    r = np.zeros(len(tok)); r[tok.ids[reward_word]] = 1.0
    if penalize:
        r[tok.ids[penalize]] = -1.0
    p_ref = base.probs([I, love])
    for beta in betas:
        m = base.copy()
        h = rl_next_token_exact(m, [I, love], r, beta=beta, steps=300, lr=0.1)
        old = loss_and_grads(m, data, need_grads=False)[0]
        theory = "" if beta == 0 else f"   theory {reward_word} {pct(kl_optimum(p_ref, r, beta)[tok.ids[reward_word]])}"
        out.append(f"beta {beta:<4}      {shares(m, tok, [I, love], show)}   loss on original text {old:.3f}"
                   f"   KL {h[-1]['kl']:.3f}{theory}")
    out.append("  beta 0: all reward, and the original text is forgotten (loss jumps).")
    out.append("  bigger beta: drifting costs more, so it stays closer to where it started.")
    out.append("  The KL only watches this one decision; other predictions (I -> love) can still drift,")
    out.append("  so the loss on the original text is not guaranteed to fall as beta grows.")

    out += ["", "== Part 2: sampled, GRPO-style, answers from 'I' ==", ]

    def reward_fn(prompt: List[int], answer: List[int]) -> float:
        score = 0.0
        if tok.ids[reward_word] in answer:
            score += 1.0
        if penalize and tok.ids[penalize] in answer:
            score -= 1.0
        return score

    rng = np.random.default_rng(0)
    answers = [base.generate([I], max_new=2, temperature=1.0, rng=rng) for _ in range(group)]
    rewards = [reward_fn([I], a) for a in answers]
    adv = group_advantages(rewards)
    out.append(f"one group of {group} sampled answers, before RL (steps 1-3 of Ch 21):")
    for a, rw, ad in zip(answers, rewards, adv):
        out.append(f"  I {tok.decode(a):12s} reward {rw:+.0f}   advantage {ad:+.2f}"
                   f"  -> {'pushed up' if ad > 0 else 'pushed down' if ad < 0 else 'unchanged'}")

    m = base.copy()
    ref = base.copy()
    out.append(f"training {steps} steps, group {group}, lr 0.02, beta 0.1:")
    means = rl_sampled(m, [[I]], reward_fn, steps=steps, group=group, max_new=2, lr=0.02, beta=0.1,
                       ref=ref, seed=0)
    marks = sorted({0, steps // 4, steps // 2, 3 * steps // 4, steps - 1})
    out.append("  mean reward  " + "  ".join(f"step {s}: {means[s]:+.2f}" for s in marks))
    q0, q1 = base.probs([I]), m.probs([I])
    out.append(f"  after 'I':      love {pct(q0[love])} -> {pct(q1[love])}   "
               f"hate {pct(q0[tok.ids['hate']])} -> {pct(q1[tok.ids['hate']])}   KL {kl(q1, q0):.3f}")
    out.append(f"  after 'I love': {shares(base, tok, [I, love], show)}  ->  {shares(m, tok, [I, love], show)}")
    out.append(f"  loss on original text {loss_and_grads(base, data, need_grads=False)[0]:.3f} -> "
               f"{loss_and_grads(m, data, need_grads=False)[0]:.3f}")
    out.append("  The group's average is the baseline, so no reward model or value network is needed.")
    out.append("  Group-normalised advantages are a different objective from Part 1, so this does not")
    out.append("  land on the closed-form number. Try a bigger lr (it overshoots) or smaller beta (collapse).")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tlm rl",
                                 description="Reinforcement learning on the tiny model (guide Chapter 21).")
    ap.add_argument("--reward", default="coffee", help="word that scores +1 (default: coffee)")
    ap.add_argument("--penalize", default="hate", help='word that scores -1 (default: hate; "" for none)')
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.5], help="KL penalty weights for part 1")
    ap.add_argument("--steps", type=int, default=30, help="sampled RL steps for part 2")
    ap.add_argument("--group", type=int, default=8, help="answers sampled per step for part 2")
    a = ap.parse_args(argv)
    try:
        print(run(a.reward, a.penalize or None, a.beta, a.steps, a.group))
    except KeyError as e:
        print(f"error: {e.args[0]}")


if __name__ == "__main__":
    main()
