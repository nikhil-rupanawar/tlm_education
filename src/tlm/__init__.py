"""tlm: the tiny language model from 'Arithmetic Intelligence', in plain NumPy."""
from .model import TLM, Config, init_params, sample, softmax
from .tokenizer import WordTokenizer
from .train import SGD, Adam, loss_and_grads, numeric_grad, train
from .finetune import LoRA, finetune
from .rl import group_advantages, kl, kl_optimum, rl_next_token_exact, rl_sampled
from .harness import ChatSession, ChatTemplate, ToolLoop, best_of_n, chain_success, majority_vote

__version__ = "0.1.0"
