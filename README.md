# tlm: your Tiny Language Model

The code behind the guide **Arithmetic Intelligence: User-friendly AI maths**. It's a complete, working language model in plain NumPy that is small enough to follow every number.

```
pip install -e .            # needs Python 3.9+ and NumPy
python -m tlm guide         # reproduce every tiny-model number in the guide (a few seconds)
python -m tlm predict "I love"   # next-token demo: before vs after training, greedy vs sampled
python -m tlm rl            # reinforcement learning: reward a word, see the KL trade-off
python -m tlm chat          # train a small café chat model (~1 min) and talk to it
python -m tlm scale         # Llama 3.1 8B memory / compute / speed arithmetic
python -m tlm harness       # reliability, voting and loop-cost arithmetic
pytest                      # 29 tests (add -m "not slow" to skip the 1-min chat test)
```

## What's inside

| Module | Guide chapters | What it does |
|---|---|---|
| `tokenizer.py` | 2 | Word tokenizer: vocabulary in order of first appearance, special tokens for chat |
| `model.py` | 3–9, 13–15 | Embedding → causal attention → ReLU feed-forward (× n layers) → tied output scoring. Forward pass, **analytic backprop**, **KV-cached** streaming generation, temperature / top-p sampling, save/load |
| `train.py` | 10–12 | Cross-entropy loss (with optional per-token masks), numeric gradient check, SGD and Adam, training loop |
| `finetune.py` | 20 | Full fine-tuning, and LoRA (`W = W_frozen + A @ B`, B starts at zero) |
| `rl.py` | 21 | RL with a KL penalty: exact next-token objective (checks against the closed form `p ∝ p_ref·e^(r/β)`), plus sampled policy gradient with group-relative advantages |
| `harness.py` | 22 | Chat template, `ChatSession` (history re-sent every turn, trimming, streaming), `ToolLoop`, and harness arithmetic |
| `scale.py` | 15–19, 23 | Llama-style parameter, memory, KV cache, FLOPs and decode-ceiling calculator |
| `guide.py` | all | Runs the whole guide story end to end |
| `predict.py` | 9, 13–14 | Next-token demo: probability bars before/after training, greedy vs sampled (`--text`, `--temperature`, `--steps`) |
| `rl_demo.py` | 21 | RL demo: exact β sweep against theory, then sampled GRPO-style RL with a reward rule (`--reward`, `--penalize`, `--beta`) |
| `chat_demo.py` | 22 | Trains a ~9K-weight chat model on café dialogues and opens a REPL |

## Next-token demo

```
python -m tlm predict "I love"                    # tea 66.6%, coffee 33.1% after training
python -m tlm predict "I hate"                    # rain 99.6%
python -m tlm predict "I love" --temperature 2    # flatter: more coffee when sampling
python -m tlm predict "the cat" --text my.txt     # your own sentences, one per line
```

## RL demo

The use case: a café owns the model trained on the guide's four sentences. It wants coffee recommended more and
"hate" never said, without writing new training text. RL swaps "compare with the text" for "try answers, score
them with a rule, push up the good ones" (Chapter 21).

```
python -m tlm rl                                  # reward coffee +1, hate -1; beta 0 and 0.5
python -m tlm rl --reward tea --beta 0.2 1        # any vocabulary word, any betas
python -m tlm rl --penalize "" --steps 60         # no penalty word, longer sampled run
```

- **Part 1, exact** (the guide's experiment, one decision after "I love"). With β 0 coffee reaches 100% and the
  loss on the original text jumps from 0.523 to 3.796. With β 0.5 coffee settles at 78.5%, which is what
  `p ∝ p_ref·e^(r/β)` predicts, as in the guide.
- **Part 2, sampled.** From "I", sample 8 two-word answers, score them, and compare each with the group average:
  above average is pushed up, below is pushed down, with a KL penalty to the starting model. Mean reward goes
  from 0 to 1 in about 15 steps.

To use your own rule, pass any `reward_fn(prompt_ids, answer_ids) -> float` to `rl.rl_sampled`.

## Use it in your own code

```python
from tlm import TLM, Config, WordTokenizer, train, SGD

text = ["I love tea", "I love tea", "I love coffee", "I hate rain"]
tok = WordTokenizer.build(text)
model = TLM(Config(vocab_size=len(tok)), seed=1)        # 80 random weights
train(model, [tok.encode(s) for s in text], steps=500, optimizer=SGD(0.1))
p = model.probs(tok.encode("I love"))                   # tea ≈ 0.67, coffee ≈ 0.33
print(tok.decode(model.generate(tok.encode("I love"), max_new=1, temperature=0)))
```

```python
from tlm import ChatSession
from tlm.chat_demo import train_chat_model, SYSTEM
model, tok = train_chat_model()
chat = ChatSession(model, tok, system=SYSTEM, max_turns=2)
print(chat.send("what goes with tea"))                  # biscuits go with tea
print(chat.prompt_tokens())                             # the whole history, re-sent
```

## Honest limits

- **Architecture.** Same building blocks as real models (embedding lookup, causal attention, residual shortcuts, softmax, cross-entropy). It omits RMSNorm, RoPE (positions are optional learned embeddings), multi-head and grouped-query attention, and SwiGLU (it uses ReLU).
- **Tokenizer.** Whole words, not sub-word BPE. Unknown words raise an error.
- **Chat demo.** It learns ten café question/answer pairs and answers most of them correctly with a short history. It is a demonstration of the harness, not a general assistant: it can't answer anything outside its tiny training set.
- **`rl_sampled`.** Uses normalised group advantages (GRPO-style). That changes the objective, so it doesn't land on the closed-form KL optimum and can collapse onto the rewarded answer. `rl_next_token_exact` is the one that matches the theory.
- **Sensitivity.** Training is full-batch on four sentences. Late in training the "I → love/hate" split oscillates a little, so tiny floating-point differences across machines can shift those percentages slightly. The loss and the tea/coffee split are stable.
- **Speed.** NumPy on one CPU core. Fine for thousands of weights, not millions.

## Facts used by `scale.py`

Llama 3.1 8B config (Hugging Face `meta-llama/Llama-3.1-8B`): width 4096, 32 layers, 32 query / 8 key-value heads, feed-forward 14,336, vocabulary 128,256, untied output table. H100 SXM: ≈ 3.35 TB/s, ≈ 989 dense BF16 TFLOPS. 16 bytes per weight for mixed-precision Adam (ZeRO, Rajbhandari et al. 2020). The 6 × parameters × tokens training estimate is from Kaplan et al. 2020.

License: MIT.
