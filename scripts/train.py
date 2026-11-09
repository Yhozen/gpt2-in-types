# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy"]
# ///
"""
Train a tiny GPT-2 (same architecture as GPT-2, much smaller) on Tiny Shakespeare
at the character level, and export its float weights to model/weights.json.

    uv run scripts/train.py            # trains, then writes model/weights.json

Architecture follows GPT-2 exactly: learned token + position embeddings, pre-LN
blocks, causal multi-head self-attention, a 4x MLP with tanh-approximated GELU,
a final LayerNorm, and an LM head tied to the token embedding.
"""

import json
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model" / "input.txt"
OUT = ROOT / "model" / "weights.json"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

CONFIG = dict(n_layer=4, n_head=4, n_embd=32, block_size=64)
ITERS, BATCH, LR = 6000, 64, 2e-3
torch.manual_seed(1337)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["n_embd"]
        self.n_head = c["n_head"]
        self.ln_1 = nn.LayerNorm(d)
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)
        self.ln_2 = nn.LayerNorm(d)
        self.c_fc = nn.Linear(d, 4 * d)
        self.mlp_proj = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.mlp_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))
        return x


class GPT(nn.Module):
    def __init__(self, c, vocab_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, c["n_embd"])
        self.wpe = nn.Embedding(c["block_size"], c["n_embd"])
        self.h = nn.ModuleList(Block(c) for _ in range(c["n_layer"]))
        self.ln_f = nn.LayerNorm(c["n_embd"])
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.h:
            x = b(x)
        return self.ln_f(x) @ self.wte.weight.T  # tied LM head


def main():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    text = DATA.read_text()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GPT(CONFIG, len(chars)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())}, device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    bs = CONFIG["block_size"]

    def batch(split):
        d = train if split == "train" else val
        ix = torch.randint(len(d) - bs - 1, (BATCH,))
        x = torch.stack([d[i : i + bs] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + bs + 1] for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in (batch(split) for _ in range(50))]
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    t0 = time.time()
    for it in range(ITERS + 1):
        lr = LR * 0.5 * (1 + math.cos(math.pi * it / ITERS))  # cosine decay
        for g in opt.param_groups:
            g["lr"] = lr
        if it % 500 == 0:
            print(f"iter {it}: {evaluate()} ({time.time() - t0:.0f}s)")
        x, y = batch("train")
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    sd = {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}
    OUT.write_text(json.dumps({"config": {**CONFIG, "vocab_size": len(chars)}, "vocab": chars, "weights": sd}))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
