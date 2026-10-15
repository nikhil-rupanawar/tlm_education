"""Reinforcement learning with a KL penalty (Chapter 20).

Two tools:
  * rl_next_token_exact  - the guide's experiment: maximise  E[reward] - beta * KL(pi || pi_ref)
                           over one next-token decision, with exact gradients.
  * rl_sampled           - the general recipe: sample a group of answers, score them, use
                           group-relative advantages (GRPO-style) plus a KL penalty.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .model import TLM, softmax


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(p) - np.log(q))))


def kl_optimum(p_ref: np.ndarray, reward: np.ndarray, beta: float) -> np.ndarray:
    """Closed-form best policy for E[r] - beta*KL:  p* ∝ p_ref * exp(r / beta)."""
    w = p_ref * np.exp(reward / beta)
    return w / w.sum()


def _dlogits_objective(p: np.ndarray, p_ref: np.ndarray, reward: np.ndarray, beta: float) -> np.ndarray:
    """d(-objective)/dlogits for objective J = sum p r - beta * KL(p || p_ref)."""
    g_reward = p * (reward - p @ reward)
    lr = np.log(p) - np.log(p_ref)
    g_kl = p * (lr - p @ lr)
    return -(g_reward - beta * g_kl)


def rl_next_token_exact(model: TLM, prompt: Sequence[int], reward: np.ndarray, beta: float = 0.5,
                        steps: int = 300, lr: float = 0.1, ref: Optional[TLM] = None,
                        callback: Optional[Callable[[int, dict, TLM], None]] = None) -> List[dict]:
    ref = ref or model.copy()
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
