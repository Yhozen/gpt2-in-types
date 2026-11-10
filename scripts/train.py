# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy"]
# ///
"""
Quantization-aware training of a tiny integer-only GPT-2 for the type checker.

    uv run scripts/train.py        # writes model/qat.json (~10 min on Apple Silicon)

Defaults: TinyStories (Eldan & Li 2023), 4 layers, 8 heads, 64 dims, 64-char
context. Override with DATASET=shakespeare, N_EMBD, N_HEAD, N_LAYER, BLOCK,
QAT_ITERS, QAT_WARMUP, QAT_OUT.

Architecture: GPT-2 (learned token + position embeddings, pre-LN blocks, causal
multi-head attention, 4x GELU MLP, final LayerNorm, LM head), with every
operation replaced by an integer version that is cheap in the type system:

- Linear layers are BitLinear: ternary weights {-1, 0, 1} with one absmean
  scale per matrix (BitNet b1.58, Ma et al. 2024, arXiv:2402.17764; BitLinear
  from Wang et al. 2023, arXiv:2310.11453). A ternary mat-vec is only additions.
- Activations are small integers with static per-tensor scales (static-scale
  integer pipeline as in I-BERT, Kim et al. 2021, arXiv:2101.01321, and Jacob
  et al. 2018, arXiv:1712.05877). Scales are calibrated once (0.999 quantile)
  when quantization switches on and then frozen: learning them (LSQ-style)
  made the integer phase drift and diverge.
- Requantization, GELU and exp are monotone threshold tables or small lookup
  tables (table-lookup inference as in T-MAC, Wei et al. 2024,
  arXiv:2407.00088), so the type checker never multiplies.
- LayerNorm has no affine parameters and quantizes its output to int8.
- q/k/v use 4-bit magnitudes (low-bit KV cache; cf. SAW-INT4, arXiv:2604.19157).
- Softmax is integer-only: e_j = round(127 exp(-d_j u)), L = round(ln(sum e / 127) / u),
  p_j = round(63 exp(-(d_j + L) u)), with d_j = max(s) - s_j in score units u.

The forward pass computes the exact integer values (as float tensors holding
integers) and passes gradients straight through (STE). The integer program is
re-derived in scripts/compile.ts; src/reference/int.ts is the exact spec.
"""

import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model" / "input.txt"
import os as _os
OUT = ROOT / (_os.environ.get("QAT_OUT") or "model/qat.json")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

CFG = dict(
    n_layer=int(os.environ.get("N_LAYER", 4)),
    n_head=int(os.environ.get("N_HEAD", 8)),
    n_embd=int(os.environ.get("N_EMBD", 64)),
    block_size=int(os.environ.get("BLOCK", 64)),
)
DATASET = os.environ.get("DATASET", "tinystories")
TINYSTORIES = ROOT / "model" / "tinystories-valid.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
# Optional: also train on the first TRAIN_MB megabytes of the TinyStories train split.
TRAIN_MB = int(os.environ.get("TRAIN_MB", 0))
# Optional: a subword vocabulary from scripts/tokenizer.py ({"pieces": [...], "eos": id}).
TOKENIZER = os.environ.get("TOKENIZER")
KEEP = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'\"-:;\n")
EOT = "\x00"  # stands for <|endoftext|> until the text is encoded


def normalize(t):
    """TinyStories text reduced to a small ASCII character set; <|endoftext|> becomes EOT."""
    for a, b in (("<|endoftext|>", EOT), ("\u201c", '"'), ("\u201d", '"'), ("\u2019", "'"), ("\u2018", "'"), ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        t = t.replace(a, b)
    return "".join(c for c in t if c in KEEP or c == EOT)


def raw_text():
    """Normalized TinyStories validation file (Eldan & Li 2023, arXiv:2305.07759), with EOT marks."""
    if not TINYSTORIES.exists():
        TINYSTORIES.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINYSTORIES_URL, TINYSTORIES)
    return normalize(TINYSTORIES.read_text(encoding="utf8", errors="ignore"))


def extra_train_text(mb):
    """The first `mb` megabytes of the TinyStories train split, cut at a story boundary."""
    path = ROOT / "model" / f"tinystories-train-{mb}m.txt"
    if not path.exists():
        req = urllib.request.Request(TRAIN_URL, headers={"Range": f"bytes=0-{mb * 1024 * 1024 - 1}"})
        with urllib.request.urlopen(req) as r:
            path.write_bytes(r.read())
    t = path.read_text(encoding="utf8", errors="ignore")
    return normalize(t[: t.rindex("<|endoftext|>") + len("<|endoftext|>")])


def load_text():
    """Tiny Shakespeare as is, or TinyStories with story boundaries as newlines."""
    if DATASET == "shakespeare":
        if not DATA.exists():
            DATA.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL, DATA)
        return DATA.read_text()
    return raw_text().replace(EOT, "\n")


_UNITS = re.compile(EOT + r"| ?[A-Za-z]+| ?[^A-Za-z ]| ")


def units(text):
    """Pre-tokenization: optional space + a word, or optional space + one other character.
    Pieces never span units, so greedy matching per unit equals greedy matching of the text."""
    return _UNITS.findall(text)


def encode_text(text, pieces, eos=None):
    """Greedy longest-match tokenization (the rule of src/reference/int.ts and the type-level Encode).
    EOT becomes the eos token, or a newline when the vocabulary has none."""
    index = {p: i for i, p in enumerate(pieces) if p}
    longest = max(map(len, index))
    cache = {EOT: [eos if eos is not None else index["\n"]]}
    out = []
    for u in units(text):
        ids = cache.get(u)
        if ids is None:
            ids, i = [], 0
            while i < len(u):
                for n in range(min(longest, len(u) - i), 0, -1):
                    if u[i : i + n] in index:
                        ids.append(index[u[i : i + n]])
                        i += n
                        break
                else:
                    raise ValueError(f"cannot encode {u[i]!r}")
            cache[u] = ids
        out.extend(ids)
    return out


def load_data():
    """(pieces, eos, train ids, val ids, val characters). The validation text is always the
    last 10% of the validation file, whatever the vocabulary or the extra training data."""
    if DATASET == "shakespeare":
        text = load_text()
        pieces, eos = sorted(set(text)), None
    else:
        text = raw_text()
        if TOKENIZER:
            tok = json.loads((ROOT / TOKENIZER).read_text())
            pieces, eos = tok["pieces"], tok["eos"]
        else:
            pieces, eos = sorted(set(text) - {EOT}), None
    n = int(0.9 * len(text))
    train_text, val_text = text[:n], text[n:]
    if TRAIN_MB:
        train_text = extra_train_text(TRAIN_MB) + train_text
    return pieces, eos, encode_text(train_text, pieces, eos), encode_text(val_text, pieces, eos), len(val_text)


# Integer ranges (magnitudes).
R_MAX = 255  # residual stream
Z_MAX = 127  # LayerNorm output (resolution sigma / 32)
QKV_MAX = 15  # queries, keys, values
E_MAX = 127  # softmax numerators
P_MAX = 63  # attention probabilities
O_MAX = 63  # attention output (out-proj input)
G_MAX = 63  # pre-GELU
H_MAX = 63  # GELU output (mlp-proj input)

ITERS, WARMUP_FLOAT, BATCH, LR, LR_QAT = 14000, 1500, 64, 3e-3, 1e-3
import os
# Debug switches (comma-separated): float_head, float_softmax, float_weights, no_act_quant
FLAGS = set(filter(None, os.environ.get("QAT_FLAGS", "").split(",")))
if os.environ.get("QAT_ITERS"):
    ITERS, WARMUP_FLOAT = int(os.environ["QAT_ITERS"]), int(os.environ["QAT_WARMUP"])
torch.manual_seed(1337)


def rnd(x):
    """Round half away from zero (the spec's rounding everywhere)."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def ste(x, v):
    """Value v, gradient of x."""
    return x + (v - x).detach()


class Quant(nn.Module):
    """Static per-tensor scale (learned, log-parameterized) to a clamped integer grid."""

    def __init__(self, qmax):
        super().__init__()
        self.qmax = qmax
        self.log_s = nn.Parameter(torch.zeros(()))
        self.calibrated = False

    @property
    def s(self):
        return self.log_s.exp()

    def calibrate(self, x):
        a = x.detach().abs().flatten().float().cpu()
        k = max(1, int(0.999 * a.numel()))
        self.log_s.data = torch.log(a.kthvalue(k).values.clamp(min=1e-4) / self.qmax)
        self.calibrated = True

    def forward(self, y):
        """y (float) -> integer-valued tensor in units of s."""
        x = y / self.s
        if "no_act_quant" in FLAGS:
            return x
        return ste(x, torch.clamp(rnd(x), -self.qmax, self.qmax))


class BitLinear(nn.Module):
    def __init__(self, n_in, n_out, bias=True, wmax=1):
        super().__init__()
        # wmax=1: ternary absmean (BitNet b1.58). wmax>1: symmetric int weights
        # with one absmax scale (used for the LM head, which BitNet also keeps
        # at higher precision).
        self.wmax = wmax
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None

    def ternary(self):
        if self.wmax == 1:
            g = self.weight.abs().mean().clamp(min=1e-5)
        else:
            g = (self.weight.abs().max() / self.wmax).clamp(min=1e-5)
        return torch.clamp(rnd(self.weight / g), -self.wmax, self.wmax), g

    def forward(self, a, in_scale, quant):
        """a: integer-valued input in units in_scale. Returns float output y."""
        if not quant or ("float_weights" in FLAGS) or (self.bias is None and "float_head" in FLAGS):
            return F.linear(a * in_scale, self.weight, self.bias)
        wt, g = self.ternary()
        w = self.weight + (wt * g - self.weight).detach()
        y = F.linear(a * in_scale, w)
        if self.bias is not None:
            # Bias folded into the integer accumulator: B = round(b / (g * in_scale)).
            unit = (g * in_scale).detach()
            y = y + ste(self.bias, rnd(self.bias / unit) * unit)
        return y


def int_layernorm(x, quant):
    """x: integer residual (B,T,d). Returns z in units of 1/32 (normalized)."""
    d = x.shape[-1]
    mu = rnd(x.sum(-1, keepdim=True) / d)
    c = x - mu
    ss = (c * c).sum(-1, keepdim=True)
    sigma_f = torch.sqrt(ss / d + 1e-6)
    if not quant:
        return 32 * c / sigma_f
    # sigma from SS truncated to 3 significant digits (the type-level sqrt table), in exact
    # integer arithmetic: float32 holds integers exactly only below 2^24, and SS reaches
    # n_embd * 510^2 (33M at 128 dimensions).
    ci = c.detach().round().to(torch.int32)
    ssi = (ci * ci).sum(-1, keepdim=True).to(torch.int64)
    p = torch.ones_like(ssi)
    for k in range(3, 12):
        p = torch.where(ssi >= 10**k, 10 ** (k - 2), p)
    ss_m = torch.div(ssi, p, rounding_mode="floor") * p
    # sigma = max(1, round(sqrt(ss_m / d))): the largest s with (2s - 1)^2 d <= 4 ss_m.
    s = torch.floor(torch.sqrt(ss_m.float() / d) + 0.5).to(torch.int64)
    up, down = 2 * s + 1, 2 * s - 1
    s = torch.where(up * up * d <= 4 * ss_m, s + 1, s)
    s = torch.where((s > 0) & (down * down * d > 4 * ss_m), s - 1, s)
    sigma = ste(sigma_f, s.clamp(min=1).to(x.dtype))
    zq = torch.sign(c) * torch.floor((torch.floor(64 * c.abs() / sigma.detach()) + 1) / 2)
    return ste(32 * c / sigma, torch.clamp(zq, -Z_MAX, Z_MAX))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.h = c["n_embd"], c["n_head"]
        self.hd = d // self.h
        self.q, self.k, self.v = BitLinear(d, d), BitLinear(d, d), BitLinear(d, d)
        self.proj = BitLinear(d, d)
        self.fc = BitLinear(d, 4 * d)
        self.mlp = BitLinear(4 * d, d)
        self.qq, self.qk, self.qv = Quant(QKV_MAX), Quant(QKV_MAX), Quant(QKV_MAX)
        self.qo, self.qg, self.qh = Quant(O_MAX), Quant(G_MAX), Quant(H_MAX)
        self.qd1, self.qd2 = None, None  # residual deltas use the model's residual quantizer

    def attn(self, z, quant):
        B, T, C = z.shape
        if not quant:
            q, k, v = (lin(z, 1 / 32, False) for lin in (self.q, self.k, self.v))
            q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return y.transpose(1, 2).reshape(B, T, C), None
        qi, ki, vi = self.qq(self.q(z, 1 / 32, True)), self.qk(self.k(z, 1 / 32, True)), self.qv(self.v(z, 1 / 32, True))
        qi, ki, vi = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in (qi, ki, vi))
        u = (self.qq.s * self.qk.s / math.sqrt(self.hd))
        s = qi @ ki.transpose(-1, -2)  # integer scores
        mask = torch.ones(T, T, dtype=torch.bool, device=z.device).tril()
        s_m = s.masked_fill(~mask, float("-inf"))
        # Mask after scaling: -inf * u would make the gradient w.r.t. u NaN.
        p_float = P_MAX * torch.softmax((s * u).masked_fill(~mask, float("-inf")), dim=-1)
        with torch.no_grad():
            ud = u.detach()
            m = s_m.max(-1, keepdim=True).values
            dd = torch.where(mask, m - s, torch.zeros_like(s))
            e = torch.where(mask, rnd(E_MAX * torch.exp(-dd * ud)), torch.zeros_like(s))
            L = rnd(torch.log(e.sum(-1, keepdim=True) / E_MAX) / ud)
            p_int = torch.where(mask, rnd(P_MAX * torch.exp(-(dd + L) * ud)), torch.zeros_like(s))
        p = p_float if "float_softmax" in FLAGS else ste(p_float, p_int)
        out = p @ vi  # integer
        y = out * (self.qv.s / P_MAX)
        y = y.transpose(1, 2).reshape(B, T, C)
        return y, None

    def forward(self, x, model, quant):
        z = int_layernorm(x, quant)
        y, _ = self.attn(z, quant)
        if quant:
            o = self.qo(y)
            x = model.add_residual(x, self.proj(o, self.qo.s, True), quant)
        else:
            x = model.add_residual(x, self.proj(y, 1.0, False), quant)
        z = int_layernorm(x, quant)
        if quant:
            g = self.qg(self.fc(z, 1 / 32, True))
            hf = F.gelu(g * self.qg.s, approximate="tanh") / self.qh.s
            h = ste(hf, torch.clamp(rnd(hf), -H_MAX, H_MAX))
            x = model.add_residual(x, self.mlp(h, self.qh.s, True), quant)
        else:
            x = model.add_residual(x, self.mlp(F.gelu(self.fc(z, 1 / 32, False), approximate="tanh"), 1.0, False), quant)
        return x


class GPT(nn.Module):
    def __init__(self, c, vocab):
        super().__init__()
        d = c["n_embd"]
        self.wte = nn.Parameter(torch.randn(vocab, d) * 0.5)
        self.wpe = nn.Parameter(torch.randn(c["block_size"], d) * 0.1)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layer"]))
        self.head = BitLinear(d, vocab, bias=False, wmax=7)
        self.log_sr = nn.Parameter(torch.tensor(math.log(4.0 / R_MAX)))  # residual scale

    @property
    def sr(self):
        return self.log_sr.exp()

    def add_residual(self, x, y, quant):
        if not quant:
            return x + y / self.sr.detach()
        dx = y / self.sr
        delta = ste(dx, torch.clamp(rnd(dx), -R_MAX, R_MAX))
        s = x + delta
        return ste(s, torch.clamp(s, -R_MAX, R_MAX))

    def forward(self, idx, quant):
        T = idx.shape[1]
        if quant:
            et = ste(self.wte / self.sr, rnd(self.wte / self.sr))
            ep = ste(self.wpe / self.sr, rnd(self.wpe / self.sr))
            x = et[idx] + ep[:T]
            x = ste(x, torch.clamp(x, -R_MAX, R_MAX))
        else:
            x = (self.wte[idx] + self.wpe[:T]) / self.sr.detach()
        for b in self.blocks:
            x = b(x, self, quant)
        z = int_layernorm(x, quant)
        return self.head(z, 1 / 32, quant)


def main():
    # The integer forward pass runs in float32 (MPS), exact below 2^24. LayerNorm's sum of
    # squares is computed in int64 (int_layernorm); every other intermediate stays far below.
    chars, eos, train_ids, val_ids, val_chars = load_data()
    train, val = torch.tensor(train_ids, dtype=torch.long), torch.tensor(val_ids, dtype=torch.long)
    per_char = len(val) / val_chars  # tokens per character: nats/token -> nats/char
    print(f"vocab {len(chars)}, train {len(train)} tokens, val {len(val)} tokens ({1 / per_char:.2f} chars/token)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = GPT(CFG, len(chars)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())}, device: {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    bs = CFG["block_size"]

    def batch(split):
        d = train if split == "train" else val
        ix = torch.randint(len(d) - bs - 1, (BATCH,))
        x = torch.stack([d[i : i + bs] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + bs + 1] for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def evaluate(quant):
        model.eval()
        losses = [F.cross_entropy(model(x, quant).flatten(0, 1), y.flatten()).item() for x, y in (batch("val") for _ in range(40))]
        model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def evaluate_fixed():
        """Integer loss over fixed, non-overlapping validation windows (comparable across runs)."""
        model.eval()
        count = min(2048, (len(val) - 1) // bs)
        step = (len(val) - bs - 1) // count
        total = 0.0
        for i in range(0, count, BATCH):
            ix = [j * step for j in range(i, min(count, i + BATCH))]
            x = torch.stack([val[j : j + bs] for j in ix]).to(device)
            y = torch.stack([val[j + 1 : j + bs + 1] for j in ix]).to(device)
            total += F.cross_entropy(model(x, True).flatten(0, 1), y.flatten(), reduction="sum").item()
        model.train()
        return total / (count * bs)

    @torch.no_grad()
    def calibrate():
        """Set every activation scale from a float forward pass (0.999 quantile)."""
        x, _ = batch("train")
        T = x.shape[1]
        # Residual scale first: the float network is invariant to it (LayerNorm
        # removes the scale), so it can be set from the float residual stream.
        h = model.wte[x] + model.wpe[:T]
        res = [h]
        for b in model.blocks:
            h = b(h / model.sr, model, False) * model.sr
            res.append(h)
        a = torch.cat([r.abs().flatten() for r in res]).cpu()
        model.log_sr.data = torch.log(a.kthvalue(int(0.9995 * a.numel())).values / R_MAX).to(model.log_sr.device)
        h = (model.wte[x] + model.wpe[:T]) / model.sr
        for b in model.blocks:
            z = int_layernorm(h, False)
            for lin, qz in ((b.q, b.qq), (b.k, b.qk), (b.v, b.qv)):
                qz.calibrate(lin(z, 1 / 32, False))
            y, _ = b.attn(z, False)
            b.qo.calibrate(y)
            h = h + b.proj(y, 1.0, False) / model.sr
            z = int_layernorm(h, False)
            f = b.fc(z, 1 / 32, False)
            b.qg.calibrate(f)
            b.qh.calibrate(F.gelu(f, approximate="tanh"))
            h = h + b.mlp(F.gelu(f, approximate="tanh"), 1.0, False) / model.sr

    t0 = time.time()
    quant = False
    best = [float("inf"), None]
    for it in range(ITERS + 1):
        if it == WARMUP_FLOAT:
            print(f"float val loss {evaluate(False):.4f}; switching to integer QAT", flush=True)
            calibrate()
            quant = True
            if "learn_scales" not in FLAGS:
                # Static scales (PTQ-style observers): learned scales drift and
                # destabilize the integer phase.
                for n_, p_ in model.named_parameters():
                    if "log_s" in n_:
                        p_.requires_grad_(False)
        # Float phase: cosine LR -> LR_QAT. Integer phase: cosine LR_QAT -> 0.
        # Switching to QAT at a high learning rate destabilizes training.
        if it < WARMUP_FLOAT:
            lr = LR_QAT + (LR - LR_QAT) * 0.5 * (1 + math.cos(math.pi * it / WARMUP_FLOAT))
        else:
            lr = LR_QAT * 0.5 * (1 + math.cos(math.pi * (it - WARMUP_FLOAT) / (ITERS - WARMUP_FLOAT)))
        for g in opt.param_groups:
            g["lr"] = lr
        if it % 500 == 0:
            v = evaluate(quant)
            print(f"iter {it}: val {v:.4f} ({'int' if quant else 'float'}, {time.time() - t0:.0f}s)", flush=True)
            if quant and v < best[0]:
                best[0] = v
                best[1] = {k: t.detach().clone() for k, t in model.state_dict().items()}
        x, y = batch("train")
        loss = F.cross_entropy(model(x, quant).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    if best[1] is not None:
        model.load_state_dict(best[1])
    print(f"final integer val loss {evaluate(True):.4f} (best checkpoint)", flush=True)
    fixed = evaluate_fixed()
    print(f"fixed-window integer val loss {fixed:.4f} nats/token = {fixed * per_char:.4f} nats/char", flush=True)

    def lin(l):
        wt, g = l.ternary()
        return {
            "w": wt.to(torch.int8).cpu().tolist(),
            "gamma": g.item(),
            "b": l.bias.detach().cpu().tolist() if l.bias is not None else None,
        }

    out = {
        "config": {**CFG, "vocab_size": len(chars)},
        "ranges": dict(R=R_MAX, Z=Z_MAX, QKV=QKV_MAX, E=E_MAX, P=P_MAX, O=O_MAX, G=G_MAX, H=H_MAX),
        "vocab": chars,
        **({"eos": eos} if eos is not None else {}),
        "sr": model.sr.item(),
        "wte": model.wte.detach().cpu().tolist(),
        "wpe": model.wpe.detach().cpu().tolist(),
        "layers": [
            {
                "q": lin(b.q), "k": lin(b.k), "v": lin(b.v), "proj": lin(b.proj), "fc": lin(b.fc), "mlp": lin(b.mlp),
                "sq": b.qq.s.item(), "sk": b.qk.s.item(), "sv": b.qv.s.item(),
                "so": b.qo.s.item(), "sg": b.qg.s.item(), "sh": b.qh.s.item(),
            }
            for b in model.blocks
        ],
        "head": {**lin(model.head), "wmax": model.head.wmax},
    }
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
