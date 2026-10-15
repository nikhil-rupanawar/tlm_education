import numpy as np
import pytest

from tlm import TLM, Config, WordTokenizer, sample, rl_sampled, finetune, train, Adam, SGD
from tlm import guide
from tlm.harness import SPECIALS, ChatSession
from tlm.train import loss_and_grads


# ---------------- tokenizer
def test_tokenizer_order_roundtrip_and_errors(tmp_path):
    tok = WordTokenizer.build(["b a", "a c"], specials=["<s>"])
    assert tok.vocab == ["<s>", "b", "a", "c"]
    assert tok.decode(tok.encode("c a b")) == "c a b"
    with pytest.raises(KeyError):
        tok.encode("zebra")
    tok.save(str(tmp_path / "v.json"))
    assert WordTokenizer.load(str(tmp_path / "v.json")).vocab == tok.vocab


# ---------------- sampling
def test_sampling_greedy_temperature_top_p():
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    assert sample(logits, temperature=0) == 0
    rng = np.random.default_rng(0)
    draws = [sample(logits, 1.0, top_p=0.85, rng=rng) for _ in range(2000)]
    assert 2 not in draws                                  # outside the top-p set
    frac = np.mean(np.array(draws) == 0)
    assert abs(frac - 0.6 / 0.9) < 0.04                    # renormalised share


def test_generate_respects_stop_and_max_new():
    tok, data, m = guide.setup()
    train(m, data, steps=500, optimizer=SGD(0.1))
    out = m.generate([0, 1], max_new=5, temperature=0)
    assert len(out) == 5
    assert out[0] == tok.ids["tea"]                        # greedy picks the top share
    out = m.generate([0, 1], max_new=5, temperature=0, stop=[tok.ids["tea"]])
    assert out == [tok.ids["tea"]]


# ---------------- training
def test_loss_starts_near_uniform_and_falls():
    tok, data, m = guide.setup()
    losses = train(m, data, steps=200, optimizer=SGD(0.1))
    assert losses[-1] < losses[0] - 1.0


def test_adam_trains_too():
    tok, data, m = guide.setup()
    losses = train(m, data, steps=200, optimizer=Adam(lr=0.05))
    assert losses[-1] < 0.6


def test_learning_rate_too_high_blows_up():
    tok, data, m = guide.setup()
    with np.errstate(all="ignore"):
        losses = train(m, data, steps=60, optimizer=SGD(1.0))
    assert not np.isfinite(losses[-1]) or losses[-1] > losses[0]


def test_frozen_keys_do_not_move():
    tok, data, m = guide.setup()
    before = {k: v.copy() for k, v in m.params.items()}
    train(m, data, steps=5, optimizer=SGD(0.1), keys=["W2"])
    for k in m.params:
        assert np.array_equal(m.params[k], before[k]) == (k != "W2")


def test_masked_loss_ignores_unmasked_targets():
    tok, data, m = guide.setup()
    ids = data[0]
    full, _ = loss_and_grads(m, [ids], need_grads=False)
    last_only, _ = loss_and_grads(m, [(ids, [False, True])], need_grads=False)
    p = m.probs(ids[:2])
    assert abs(last_only - (-np.log(p[ids[2]]))) < 1e-12
    assert full != last_only


# ---------------- fine-tuning and RL
def test_finetune_shifts_behaviour_and_causes_forgetting():
    tok, data, m = guide.setup()
    train(m, data, steps=500, optimizer=SGD(0.1))
    hate_before = m.probs([0])[tok.ids["hate"]]
    finetune(m, [tok.encode("I love coffee")] * 3, steps=10)
    assert m.probs([0, 1])[tok.ids["coffee"]] > 0.95
    assert m.probs([0])[tok.ids["hate"]] < hate_before / 5     # forgetting


def test_rl_sampled_raises_reward():
    tok, data, m = guide.setup()
    train(m, data, steps=500, optimizer=SGD(0.1))
    c = tok.ids["coffee"]
    before = m.probs([0, 1])[c]
    rl_sampled(m, [[0, 1]], lambda p, a: float(a == [c]), steps=20, group=8, max_new=1, lr=0.2, beta=0.5)
    assert m.probs([0, 1])[c] > before + 0.2


# ---------------- harness
def _chat_model():
    tok = WordTokenizer.build(["hi", "hello"], specials=SPECIALS)
    cfg = Config(vocab_size=len(tok), d_model=8, d_head=4, d_ff=8, max_len=30, init_round=None)
    return TLM(cfg, seed=0), tok


def test_chat_session_resends_history_and_trims():
    m, tok = _chat_model()
    s = ChatSession(m, tok, system="hi", max_context=30, reserve=4)
    sizes = []
    for _ in range(6):
        s.send("hi", max_new=3)
        sizes.append(s.prompt_tokens())
    assert sizes[1] > sizes[0]                              # history grows...
    assert max(sizes) + 4 <= 30 + 6                         # ...but is trimmed to the budget
    s2 = ChatSession(m, tok, max_turns=1)
    for _ in range(4):
        s2.send("hi", max_new=2)
    assert len(s2.history) <= 4


@pytest.mark.slow
def test_chat_demo_end_to_end():
    from tlm.chat_demo import PAIRS, SYSTEM, train_chat_model
    model, tok = train_chat_model(steps=300, verbose=False)
    s = ChatSession(model, tok, system=SYSTEM, max_turns=2)
    answers = dict(PAIRS)
    qs = ["hi", "what goes with tea", "how much is coffee", "thanks", "what goes with coffee", "bye"]
    correct = sum(s.send(q) == answers[q] for q in qs)
    assert correct >= 5


# ---------------- predict demo / CLI
def test_predict_demo_output():
    from tlm.predict import run
    out = run("I love")
    assert "BEFORE training" in out and "AFTER 500 training steps" in out
    assert "tea       66.6%" in out
    assert "Greedy (always the top token): tea" in out


def test_predict_custom_text(tmp_path):
    from tlm.predict import main
    f = tmp_path / "t.txt"
    f.write_text("the cat sat\nthe cat sat\nthe dog ran\n")
    from tlm.predict import run
    out = run("the", text=f.read_text().splitlines(), steps=300)
    assert "cat" in out.split("AFTER")[1].splitlines()[1]       # top prediction after 'the'


def test_cli_entrypoint(capsys):
    from tlm.__main__ import main
    main(["predict", "I hate", "--samples", "3"])
    assert "rain" in capsys.readouterr().out
