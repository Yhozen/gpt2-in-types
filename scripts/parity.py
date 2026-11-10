# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy"]
# ///
"""
Parity vectors: run the trained checkpoint through the PyTorch integer forward
pass of scripts/train.py (the simulation the reported validation loss comes
from) and write the LM-head accumulators for a few validation windows to
model/parity.json. tests/reference.test.ts asserts that src/reference/int.ts,
whose tables are rebuilt in float64 by scripts/compile.ts, reproduces them.

    uv run scripts/parity.py            # QAT=model/exp/x.json to pick a checkpoint

Also prints the validation loss of the same windows in float64 and in float32
(what training evaluates), and how many accumulators differ between the two.
"""

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = int(os.environ.get("PARITY_WINDOWS", 256))  # evaluated
STORED = int(os.environ.get("PARITY_STORED", 4))  # written to parity.json

spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def load(qat, dtype):
    torch.set_default_dtype(dtype)
    cfg = {k: qat["config"][k] for k in ("n_layer", "n_head", "n_embd", "block_size")}
    model = train.GPT(cfg, qat["config"]["vocab_size"])
    t = lambda v: torch.tensor(v, dtype=dtype)

    def set_lin(lin, d):
        # The exported ternary weights and scale, exactly (no re-quantization).
        lin._wt, lin._g = t(d["w"]), t(d["gamma"])
        lin.weight.data = lin._wt * lin._g
        if d["b"] is not None:
            lin.bias.data = t(d["b"])

    train.BitLinear.ternary = lambda self: (self._wt, self._g)
    model.wte.data, model.wpe.data = t(qat["wte"]), t(qat["wpe"])
    model.log_sr.data = t(math.log(qat["sr"]))
    for b, L in zip(model.blocks, qat["layers"]):
        for name in ("q", "k", "v", "proj", "fc", "mlp"):
            set_lin(getattr(b, name), L[name])
        for quant, key in ((b.qq, "sq"), (b.qk, "sk"), (b.qv, "sv"), (b.qo, "so"), (b.qg, "sg"), (b.qh, "sh")):
            quant.log_s.data = t(math.log(L[key]))
    set_lin(model.head, qat["head"])
    return model.eval()


def main():
    qat_path = ROOT / os.environ.get("QAT", "model/qat.json")
    qat = json.loads(qat_path.read_text())
    # The validation text, tokenized with the checkpoint's own vocabulary.
    text = train.raw_text()
    val_text = text[int(0.9 * len(text)) :]
    val = train.encode_text(val_text, qat["vocab"], qat.get("eos"))
    chars_per_token = len(val_text) / len(val)
    bs = qat["config"]["block_size"]
    step = (len(val) - bs - 1) // WINDOWS
    ids = torch.tensor([val[i * step : i * step + bs + 1] for i in range(WINDOWS)])
    x, y = ids[:, :-1], ids[:, 1:]

    out = {}
    for dtype in (torch.float64, torch.float32):
        model = load(qat, dtype)
        with torch.no_grad():
            logits = model(x, True)
        acc = logits * 32 / qat["head"]["gamma"]
        out[dtype] = (acc.round().long(), (acc - acc.round()).abs().max().item(), F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    a64, a32 = out[torch.float64][0], out[torch.float32][0]
    for dtype, (_, frac, loss) in out.items():
        print(f"{str(dtype):>14}: val loss {loss:.4f} nats/token ({loss / chars_per_token:.4f} nats/char) over {WINDOWS} windows; max distance of an accumulator from an integer {frac:.2e}")
    diff = (a64 != a32).any(-1)
    print(f"positions where float32 and float64 accumulators differ: {diff.sum().item()} of {diff.numel()}")

    dest = ROOT / "model" / "parity.json"
    dest.write_text(json.dumps({
        "source": str(qat_path.relative_to(ROOT)),
        "loss64": out[torch.float64][2],
        "tokens": ids[:STORED].tolist(),
        "acc": a64[:STORED].tolist(),
    }))
    # The full evaluation set, for scripts/parity.ts (not committed).
    (ROOT / ".run").mkdir(exist_ok=True)
    (ROOT / ".run" / "parity-full.json").write_text(json.dumps({"tokens": ids.tolist(), "acc": a64.tolist(), "gamma": qat["head"]["gamma"], "chars_per_token": chars_per_token}))
    print(f"wrote {dest.relative_to(ROOT)} ({STORED} windows) and .run/parity-full.json ({WINDOWS})")


if __name__ == "__main__":
    sys.exit(main())
