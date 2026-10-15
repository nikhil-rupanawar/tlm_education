"""The harness: plain code around the model (Chapter 21).

ChatTemplate  - joins messages into ONE token stream with role markers.
ChatSession   - keeps history, re-sends it every turn, trims to the context budget, streams.
ToolLoop      - think -> act -> observe loop around any model function.
Arithmetic    - multi-step reliability, majority vote, best-of-n, loop token cost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import comb
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np

from .model import TLM
from .tokenizer import WordTokenizer

SPECIALS = ["<bos>", "<system>", "<user>", "<assistant>", "<end>"]


class ChatTemplate:
    """<bos> <system> ... <end> <user> ... <end> <assistant> ... <end> ...

    Same idea as Llama 3.1's <|start_header_id|>user<|end_header_id|> ... <|eot_id|>.
    """

    def __init__(self, tok: WordTokenizer):
        self.tok = tok
        missing = [s for s in SPECIALS if s not in tok.ids]
        if missing:
            raise ValueError(f"tokenizer lacks chat special tokens {missing}")
        self.bos, self.end = tok.ids["<bos>"], tok.ids["<end>"]
        self.role = {r: tok.ids[f"<{r}>"] for r in ("system", "user", "assistant")}

    def render(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True):
        """Returns (ids, mask). mask[t] is True when token t+1 is part of an assistant reply,
        so training can learn replies without learning to imitate the user."""
        ids, mask = [self.bos], []
        for m in messages:
            body = self.tok.encode(m["content"]) + [self.end]
            ids.append(self.role[m["role"]]); mask.append(False)
            for t in body:
                ids.append(t)
                mask.append(m["role"] == "assistant")
        if add_generation_prompt:
            ids.append(self.role["assistant"]); mask.append(False)
        return ids, mask


@dataclass
class ChatSession:
    """What a chat app does. The model keeps nothing between turns; the session re-sends history."""
    model: TLM
    tok: WordTokenizer
    system: str = ""
    max_context: Optional[int] = None      # token budget for the prompt
    reserve: int = 8                       # room left for the reply
    max_turns: Optional[int] = None        # keep at most this many past exchanges
    temperature: float = 0.0
    top_p: float = 1.0
    history: List[Dict[str, str]] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        self.tpl = ChatTemplate(self.tok)
        self.rng = np.random.default_rng(self.seed)
        if self.max_context is None and self.model.cfg.max_len:
            self.max_context = self.model.cfg.max_len

    def _messages(self) -> List[Dict[str, str]]:
        head = [{"role": "system", "content": self.system}] if self.system else []
        return head + self.history

    def build_prompt(self) -> List[int]:
        """Render; if over budget, drop the oldest turns (the 'forgets the start' effect)."""
        if self.max_turns is not None:
            keep = 2 * self.max_turns + 1           # past exchanges + the new user message
            self.history = self.history[-keep:]
        while True:
            ids, _ = self.tpl.render(self._messages())
            if self.max_context is None or len(ids) + self.reserve <= self.max_context or len(self.history) <= 1:
                return ids
            self.history = self.history[2:] if len(self.history) > 2 else self.history[1:]

    def stream(self, user_text: str, max_new: int = 12) -> Iterator[str]:
        self.history.append({"role": "user", "content": user_text})
        prompt = self.build_prompt()
        out = []
        for t in self.model.generate_stream(prompt, max_new=max_new, temperature=self.temperature,
                                            top_p=self.top_p, stop=[self.tpl.end], rng=self.rng):
            if t == self.tpl.end:
                break
            out.append(t)
            yield self.tok.vocab[t]
        self.history.append({"role": "assistant", "content": self.tok.decode(out)})

    def send(self, user_text: str, **kw) -> str:
        return " ".join(self.stream(user_text, **kw))

    def prompt_tokens(self) -> int:
        return len(self.tpl.render(self._messages())[0])


class ToolLoop:
    """Generic agent loop. model_fn(messages) returns {"content": str} or
    {"tool": name, "args": {...}}; the harness runs the tool and calls the model again."""

    def __init__(self, model_fn: Callable[[List[dict]], dict], tools: Dict[str, Callable[..., object]],
                 max_turns: int = 8):
        self.model_fn, self.tools, self.max_turns = model_fn, tools, max_turns

    def run(self, messages: List[dict]) -> List[dict]:
        msgs = list(messages)
        for _ in range(self.max_turns):
            reply = self.model_fn(msgs)
            msgs.append({"role": "assistant", **reply})
            if "tool" not in reply:
                return msgs
            fn = self.tools.get(reply["tool"])
            result = fn(**reply.get("args", {})) if fn else f"error: unknown tool {reply['tool']}"
            msgs.append({"role": "tool", "content": json.dumps(result) if not isinstance(result, str) else result})
        msgs.append({"role": "assistant", "content": "stopped: turn limit reached"})
        return msgs


# ------------------------------------------------------------------ harness arithmetic
def chain_success(p_step: float, steps: int) -> float:
    """All steps must succeed: p^n (assumes independent steps)."""
    return p_step ** steps


def majority_vote(p: float, n: int) -> float:
    """P(more than half of n independent binary answers are right)."""
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


def best_of_n(p: float, n: int) -> float:
    """P(at least one of n tries is right), given a perfect checker."""
    return 1 - (1 - p) ** n


def loop_input_tokens(tokens_per_turn: int, turns: int) -> int:
    """Each turn re-sends everything so far."""
    return tokens_per_turn * turns * (turns + 1) // 2
