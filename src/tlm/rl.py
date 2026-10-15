"""Reinforcement learning with a KL penalty (Chapter 20).

Two tools:
  * rl_next_token_exact  - the guide's experiment: maximise  E[reward] - beta * KL(pi || pi_ref)
                           over one next-token decision, with exact gradients.
  * rl_sampled           - the general recipe: sample a group of answers, score them, use
                           group-relative advantages (GRPO-style) plus a KL penalty.

Guide mapping (Chapter 20): the loop changes in one place - instead of comparing with a text,
the model tries answers, a reward rule scores them, and good answers get their share pushed up.
  1 Try answers ............. rl_sampled: model.generate(...) x group
  2 Reward each one ......... the cafe rule: coffee = 1, anything else = 0
  3 Compare to the group .... group_advantages: [0,1,0,0] -> [-0.58, 1.73, -0.58, -0.58]
  4 Push up / down + penalty  objective = reward - beta x KL (kl, _dlogits_objective)
Results to expect (python -m tlm guide, Stage 6c; python -m tlm rl):
  beta 0   -> coffee 100%, loss on the original text 0.523 -> ~3.8 (it gave up everything else)
  beta 0.5 -> coffee 78.5%, exactly kl_optimum's  new share ∝ old share x e^(reward / beta)
  (The guide prints 78.7% and 3.908; this code's trained start is a hair different, and the
   training still lands on its own theory number.)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .model import TLM, softmax


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence (Ch 20): grows as the new percentages p drift from the original model's q."""
    return float(np.sum(p * (np.log(p) - np.log(q))))


def kl_optimum(p_ref: np.ndarray, reward: np.ndarray, beta: float) -> np.ndarray:
    """Closed-form best policy for E[r] - beta*KL:  p* ∝ p_ref * exp(r / beta).

    The guide's theory check: p_ref (tea 0.66, coffee 0.33), r = coffee 1, beta 0.5 -> coffee ~78.5%
    (guide: 78.7%); beta 1 -> ~57.4% (guide: 57.6%). The training in rl_next_token_exact lands on this number.
    """
    w = p_ref * np.exp(reward / beta)
    return w / w.sum()


def _dlogits_objective(p: np.ndarray, p_ref: np.ndarray, reward: np.ndarray, beta: float) -> np.ndarray:
    """d(-objective)/dlogits for objective J = sum p r - beta * KL(p || p_ref)."""
    g_reward = p * (reward - p @ reward)         # above-average reward -> score goes up
    lr = np.log(p) - np.log(p_ref)
    g_kl = p * (lr - p @ lr)                     # drifted further than average -> pulled back
    return -(g_reward - beta * g_kl)


def rl_next_token_exact(model: TLM, prompt: Sequence[int], reward: np.ndarray, beta: float = 0.5,
                        steps: int = 300, lr: float = 0.1, ref: Optional[TLM] = None,
                        callback: Optional[Callable[[int, dict, TLM], None]] = None) -> List[dict]:
    """The guide's experiment: one decision after "I love", exact gradients (no sampling noise)."""
    ref = ref or model.copy()                    # the frozen "where it started" model the KL is measured to
    p_ref = ref.probs(prompt)
    hist = []
    for step in range(steps + 1):
        logits, tr = model.forward(prompt, keep=True)
        p = softmax(logits[-1])
        info = dict(step=step, p=p, reward=float(p @ reward), kl=kl(p, p_ref))
        hist.append(info)
        if callback:
            callback(step, info, model)
        if step == steps:
            break
        d = np.zeros_like(logits)
        d[-1] = _dlogits_objective(p, p_ref, reward, beta)
        G = model.backward(tr, d)
        for k in model.params:
            model.params[k] -= lr * G[k]
    return hist


def group_advantages(rewards: Sequence[float]) -> np.ndarray:
    """Ch 20 step 3: score each answer against its group (GRPO, as used for DeepSeek-R1).
    No reward model or value network needed: the group's own average is the baseline."""
    r = np.asarray(rewards, dtype=float)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def rl_sampled(model: TLM, prompts: Sequence[Sequence[int]], reward_fn: Callable[[List[int], List[int]], float],
               steps: int = 100, group: int = 8, max_new: int = 4, lr: float = 0.05, beta: float = 0.1,
               temperature: float = 1.0, stop: Optional[Sequence[int]] = None, ref: Optional[TLM] = None,
               seed: int = 0, log_every: int = 0) -> List[float]:
    """Policy-gradient RL: sample `group` answers per prompt, reward them, push up the
    log-probability of above-average answers, with a per-token KL penalty to `ref`."""
    rng = np.random.default_rng(seed)
    ref = ref or model.copy()
    mean_rewards = []
    for step in range(steps):
        G = {k: np.zeros_like(v) for k, v in model.params.items()}
        all_r = []
        n_samples = len(prompts) * group
        for prompt in prompts:
            answers = [model.generate(list(prompt), max_new=max_new, temperature=temperature,
                                      stop=stop, rng=rng) for _ in range(group)]
            rewards = [reward_fn(list(prompt), a) for a in answers]
            all_r += rewards
            adv = group_advantages(rewards)
            for ans, a in zip(answers, adv):
                seq = list(prompt) + ans
                logits, tr = model.forward(seq[:-1], keep=True)
                ref_logits = ref.forward(seq[:-1])
                d = np.zeros_like(logits)
                for j, tok in enumerate(ans):
                    pos = len(prompt) - 1 + j
                    p = softmax(logits[pos])
                    q = softmax(ref_logits[pos])
                    onehot = np.zeros_like(p); onehot[tok] = 1.0
                    lrat = np.log(p) - np.log(q)
                    # a * (p - onehot): like Ch 11's "share - 1" blame, scaled by the advantage, so
                    # above-average answers are pushed up and below-average pushed down.
                    # beta * ...: the KL penalty, per token, pulling back toward ref.
                    d[pos] = (a * (p - onehot) + beta * p * (lrat - p @ lrat)) / n_samples
                g = model.backward(tr, d)
                for k in G:
                    G[k] += g[k]
        for k in model.params:
            model.params[k] -= lr * G[k]
        mean_rewards.append(float(np.mean(all_r)))
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  mean reward {mean_rewards[-1]:.3f}")
    return mean_rewards
