"""Scale arithmetic (Chapters 15-18) for a Llama-style model. Default: Llama 3.1 8B config."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LlamaShape:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    vocab: int = 128256
    tied: bool = False

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def params(self) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.d_head
        attn = d * d + 2 * d * kv + d * d
        ffn = 3 * d * self.d_ff                       # SwiGLU: gate, up, down
        norms = 2 * d
        emb = self.vocab * d * (1 if self.tied else 2)
        return self.n_layers * (attn + ffn + norms) + emb + d

    def weight_bytes(self, bytes_per_weight: float = 2.0) -> float:
        return self.params() * bytes_per_weight

    def kv_bytes_per_token(self, bytes_per_value: float = 2.0) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.d_head * bytes_per_value

    def kv_bytes(self, tokens: int, users: int = 1, bytes_per_value: float = 2.0) -> float:
        return self.kv_bytes_per_token(bytes_per_value) * tokens * users

    def flops_per_token(self) -> float:
        return 2.0 * self.params()

    def train_flops(self, tokens: float) -> float:
        return 6.0 * self.params() * tokens

    def train_memory_bytes(self, bytes_per_weight: float = 16.0) -> float:
        """Mixed-precision Adam: 2 weight + 2 grad + 12 optimiser (fp32 copy, 2 moments)."""
        return self.params() * bytes_per_weight

    def decode_ceiling_tokens_per_s(self, bandwidth_bytes_s: float = 3.35e12,
                                    bytes_per_weight: float = 2.0) -> float:
        """Idealised single-user, bandwidth-bound ceiling: bandwidth / bytes read per token."""
        return bandwidth_bytes_s / self.weight_bytes(bytes_per_weight)

    def lora_params(self, rank: int = 16, targets=("q", "v")) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.d_head
        shapes = {"q": (d, d), "k": (d, kv), "v": (d, kv), "o": (d, d)}
        return self.n_layers * sum(rank * (a + b) for a, b in (shapes[t] for t in targets))


def report(m: LlamaShape | None = None) -> str:
    m = m or LlamaShape()
    GiB = 2 ** 30
    lines = [
        f"parameters               {m.params():,}",
        f"weights 16-bit / 4-bit   {m.weight_bytes(2)/1e9:.1f} GB / {m.weight_bytes(0.5)/1e9:.1f} GB",
        f"KV cache per token       {m.kv_bytes_per_token()/1024:.0f} KiB",
        f"KV cache 8,192 tokens    {m.kv_bytes(8192)/GiB:.1f} GiB",
        f"KV cache 131,072 tokens  {m.kv_bytes(131072)/GiB:.1f} GiB",
        f"KV 10 users x 32,768     {m.kv_bytes(32768, 10)/GiB:.1f} GiB",
        f"FLOPs per token          {m.flops_per_token()/1e9:.1f} billion",
        f"training FLOPs @15T tok  {m.train_flops(15e12):.2e}",
        f"training memory (Adam)   {m.train_memory_bytes()/1e9:.0f} GB before activations",
        f"decode ceiling (H100)    {m.decode_ceiling_tokens_per_s():.0f} tokens/s  (idealised, 1 user, 16-bit)",
        f"LoRA r=16 on q,v         {m.lora_params():,} trainable ({m.lora_params()/m.params()*100:.2f}%)",
    ]
    return "\n".join(lines)
