"""python -m tlm {guide|predict|rl|chat|scale|harness}"""
import sys


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "help"
    if cmd == "guide":
        from .guide import run
        run()
    elif cmd == "predict":
        from .predict import main as pmain
        pmain(argv[1:])
    elif cmd == "rl":
        from .rl_demo import main as rmain
        rmain(argv[1:])
    elif cmd == "chat":
        from .chat_demo import repl
        repl()
    elif cmd == "scale":
        from .scale import report
        print(report())
    elif cmd == "harness":
        from .harness import best_of_n, chain_success, loop_input_tokens, majority_vote
        print(f"10 steps at 95% each     {chain_success(0.95, 10):.1%}")
        print(f"majority of 5 at 60%     {majority_vote(0.6, 5):.1%}")
        print(f"best of 5 at 30% + check {best_of_n(0.3, 5):.1%}")
        print(f"10-turn loop, 2k/turn    {loop_input_tokens(2000, 10):,} input tokens")
    else:
        print(__doc__)
        print("  guide    reproduce every tiny-model number in the guide")
        print('  predict  train the tiny model and show its next-token prediction, e.g. predict "I love"')
        print("  rl       reinforcement learning: reward a word, see the KL trade-off (Ch 20)")
        print("  chat     train a small café chat model and talk to it")
        print("  scale    Llama 3.1 8B memory / compute / speed arithmetic")
        print("  harness  reliability, voting and loop-cost arithmetic")


if __name__ == "__main__":
    main()
