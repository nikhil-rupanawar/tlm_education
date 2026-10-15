import numpy as np
import pytest

from tlm import (TLM, Config, LoRA, WordTokenizer, best_of_n, chain_success, kl_optimum,
                 loss_and_grads, majority_vote, numeric_grad, rl_next_token_exact)
from tlm.harness import SPECIALS, ChatTemplate, ToolLoop, loop_input_tokens
from tlm.scale import LlamaShape
from tlm import guide


def small_model(max_len=0, layers=2):
    cfg = Config(vocab_size=7, d_model=5, d_head=3, d_ff=4, n_layers=layers, max_len=max_len,
                 init_std=0.5, init_round=None)
    return TLM(cfg, seed=3)


@pytest.mark.parametrize("max_len,layers", [(0, 1), (0, 2), (8, 2)])
def test_analytic_gradients_match_numeric(max_len, layers):
    m = small_model(max_len, layers)
    data = [[0, 1, 2, 3], [4, 1, 5], ([6, 2, 0, 1], [False, True, True])]
    _, G = loss_and_grads(m, data)
    for k, v in m.params.items():
        num = numeric_grad(lambda: loss_and_grads(m, data, need_grads=False)[0], v)
        assert np.abs(G[k] - num).max() < 1e-6, k


@pytest.mark.parametrize("max_len", [0, 8])
def test_kv_cache_matches_full_forward(max_len):
    m = small_model(max_len)
    ids = [3, 1, 4, 1, 5]
    cache = m.new_cache()
    for pos, t in enumerate(ids):
        step_logits = m.step(t, cache, pos)
        assert np.allclose(step_logits, m.forward(ids[: pos + 1])[-1])


def test_guide_initial_state_exact():
    tok, data, model = guide.setup()
    assert tok.vocab == ["I", "love", "tea", "coffee", "hate", "rain"]
    assert model.num_params() == 80
    assert model.params["E"][1].tolist() == [0.5, 0.2, -0.3, 0.3]
    assert abs(loss_and_grads(model, data, need_grads=False)[0] - 1.8642) < 1e-3


def test_guide_reproduction():
    out = guide.run(verbose=False)
    assert abs(out["newborn_tea"] - 0.180) < 1e-3
    assert abs(out["loss_final"] - 0.523) < 2e-3          # best possible is 0.520
    assert abs(out["trained_tea"] - 2 / 3) < 0.02
    assert abs(out["trained_coffee"] - 1 / 3) < 0.02
    assert out["after_tea_rain"] > 0.9
    assert out["kv_same"]
    assert out["rl_beta_0.0"] > 0.99
    assert abs(out["rl_beta_0.5"] - 0.785) < 0.01


def test_rl_matches_closed_form():
    tok, data, model = guide.setup()
    from tlm.train import train, SGD
    train(model, data, steps=500, optimizer=SGD(0.1))
    r = np.zeros(6); r[3] = 1.0
    p_ref = model.probs([0, 1])
    h = rl_next_token_exact(model, [0, 1], r, beta=0.5, steps=300)
    assert np.allclose(h[-1]["p"], kl_optimum(p_ref, r, 0.5), atol=5e-3)


def test_lora_starts_identical_and_counts():
    _, _, model = guide.setup()
    lo = LoRA(model, targets=["W2"], rank=1)
    assert lo.num_trainable() == 7
    for k, v in lo.merged_params().items():
        assert np.allclose(v, model.params[k])


def test_chat_template_and_mask():
    tok = WordTokenizer.build(["hi", "hello there"], specials=SPECIALS)
    tpl = ChatTemplate(tok)
    ids, mask = tpl.render([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}],
                           add_generation_prompt=False)
    assert tok.decode(ids) == "<bos> <user> hi <end> <assistant> hello there <end>"
    assert len(mask) == len(ids) - 1
    assert [tok.vocab[ids[i + 1]] for i, m in enumerate(mask) if m] == ["hello", "there", "<end>"]


def test_tool_loop():
    calls = []
    def fake_model(msgs):
        if msgs[-1]["role"] == "tool":
            return {"content": f"answer {msgs[-1]['content']}"}
        calls.append(1)
        return {"tool": "mul", "args": {"a": 1234, "b": 5678}}
    out = ToolLoop(fake_model, {"mul": lambda a, b: a * b}).run([{"role": "user", "content": "1234*5678?"}])
    assert out[-1]["content"] == "answer 7006652"


def test_harness_arithmetic():
    assert round(chain_success(0.95, 10), 3) == 0.599
    assert round(majority_vote(0.6, 5), 3) == 0.683
    assert round(best_of_n(0.3, 5), 3) == 0.832
    assert loop_input_tokens(2000, 10) == 110000


def test_llama_scale():
    m = LlamaShape()
    assert m.params() == 8_030_261_248                      # matches the published count
    assert m.kv_bytes_per_token() == 131072
    assert m.kv_bytes(131072) == 16 * 2 ** 30
    assert m.lora_params() == 6_815_744


def test_save_load(tmp_path):
    m = small_model(8)
    m.save(str(tmp_path / "m.npz"))
    m2 = TLM.load(str(tmp_path / "m.npz"))
    assert m2.cfg == m.cfg
    assert np.allclose(m2.forward([1, 2, 3]), m.forward([1, 2, 3]))
