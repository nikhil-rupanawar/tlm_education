"""Train a small chat model on a toy café dialogue set, then talk to it through the harness."""
from __future__ import annotations

from typing import List

from .harness import SPECIALS, ChatSession, ChatTemplate
from .model import TLM, Config
from .tokenizer import WordTokenizer
from .train import Adam, train

SYSTEM = "you are a cafe helper"
PAIRS = [
    ("hi", "hello what would you like"),
    ("what goes with tea", "biscuits go with tea"),
    ("what goes with coffee", "cake goes with coffee"),
    ("do you have tea", "yes we have tea"),
    ("do you have coffee", "yes we have coffee"),
    ("do you have juice", "sorry no juice today"),
    ("how much is tea", "tea is two coins"),
    ("how much is coffee", "coffee is three coins"),
    ("thanks", "you are welcome"),
    ("bye", "bye see you soon"),
]


def build_dialogues(n_mixed: int = 200, max_turns: int = 3, seed: int = 0) -> List[List[dict]]:
    """Every pair as a single turn, plus random 2..max_turns conversations, so the model
    learns to answer the LAST question whatever history comes before it."""
    import random
    rnd = random.Random(seed)
    sys = [{"role": "system", "content": SYSTEM}]
    def turn(q, a):
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    dl = [sys + turn(q, a) for q, a in PAIRS]
    for _ in range(n_mixed):
        conv = sys[:]
        for q, a in rnd.choices(PAIRS, k=rnd.randint(2, max_turns)):
            conv += turn(q, a)
        dl.append(conv)
    return dl


def train_chat_model(steps: int = 300, seed: int = 0, verbose: bool = True):
    texts = [SYSTEM] + [q for q, _ in PAIRS] + [a for _, a in PAIRS]
    tok = WordTokenizer.build(texts, specials=SPECIALS)
    tpl = ChatTemplate(tok)
    data = []
    for d in build_dialogues():
        ids, mask = tpl.render(d, add_generation_prompt=False)
        data.append((ids, mask))                 # learn assistant replies only
    max_len = max(len(ids) for ids, _ in data) + 4
    cfg = Config(vocab_size=len(tok), d_model=24, d_head=12, d_ff=48, n_layers=2,
                 max_len=max_len, init_std=0.2, init_round=None)
    model = TLM(cfg, seed=seed)
    if verbose:
        print(f"chat model: {model.num_params():,} weights, vocab {len(tok)}, context {max_len} tokens")
    train(model, data, steps=steps, optimizer=Adam(lr=0.01), log_every=100 if verbose else 0)
    return model, tok


def repl(path: str = "tlm_chat"):
    """Loads tlm_chat.npz / tlm_chat.vocab.json if present; otherwise trains (~1 min) and saves."""
    import os
    if os.path.exists(path + ".npz") and os.path.exists(path + ".vocab.json"):
        model, tok = TLM.load(path + ".npz"), WordTokenizer.load(path + ".vocab.json")
        print(f"loaded {path}.npz")
    else:
        print("training the chat model (about a minute)...")
        model, tok = train_chat_model(steps=300)
        model.save(path + ".npz"); tok.save(path + ".vocab.json")
    s = ChatSession(model, tok, system=SYSTEM, temperature=0.0, max_turns=2)
    print("\nType a message (words from the café set), or 'quit'. Try: what goes with tea")
    print("Known words:", " ".join(w for w in tok.vocab if not w.startswith("<")))
    while True:
        try:
            msg = input("you > ").strip()
        except EOFError:
            break
        if msg in ("quit", "exit"):
            break
        try:
            print("tlm > ", end="", flush=True)
            for w in s.stream(msg):
                print(w, end=" ", flush=True)
            print(f"   [prompt was {s.prompt_tokens()} tokens: whole history re-sent]")
        except KeyError as e:
            print(f"\n(unknown word) {e}")
            s.history.pop()


if __name__ == "__main__":
    repl()
