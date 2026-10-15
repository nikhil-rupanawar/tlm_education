from tlm import TLM, Config, WordTokenizer, train, SGD
import numpy as np

text = ["I love tea", "I love tea", "I love coffee", "I hate rain"]
tok = WordTokenizer.build(text)
model = TLM(Config(vocab_size=len(tok)), seed=1)

def show(prompt, model, title):
    p = model.probs(tok.encode(prompt))
    print(f"\n{title}: '{prompt}' -> next token")
    for w, x in sorted(zip(tok.vocab, p), key=lambda t: -t[1]):
        print(f"  {w:7s} {x*100:5.1f}%  {'#' * int(x * 40)}")

show("I love", model, "BEFORE training (random weights)")
train(model, [tok.encode(s) for s in text], steps=500, optimizer=SGD(0.1))
show("I love", model, "AFTER training")
show("I", model, "AFTER training")
show("I hate", model, "AFTER training")

rng = np.random.default_rng(0)
print("\nGreedy (always the top token):", tok.decode(model.generate(tok.encode("I love"), max_new=1, temperature=0)))
picks = [tok.decode(model.generate(tok.encode("I love"), max_new=1, temperature=1.0, rng=rng)) for _ in range(12)]
print("Sampled 12 times:", " ".join(picks))
print("Full sentence from 'I':", "I", tok.decode(model.generate(tok.encode("I"), max_new=2, temperature=0)))
