# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy"]
# ///
"""
Learn a small subword vocabulary for the type-level GPT-2.

    VOCAB=256 uv run scripts/tokenizer.py     # writes model/vocab-256.json

Byte-pair-encoding merges (Sennrich et al. 2016, arXiv:1508.07909) are learned
on the training part of the normalized TinyStories text, inside pre-tokenized
units (optional space + word), so a piece is at most one word with its leading
space. Only the resulting piece list is kept: text is encoded by greedy longest
match (train.encode_text, src/reference/int.ts, and Encode in src/engine), which
is easy to reproduce exactly in the type system. Every single character stays
a piece, so any text over the character set can be encoded. The last id is the
end-of-story token; its piece is the empty string.
"""

import collections
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def main():
    size = int(os.environ.get("VOCAB", 256))
    text = train.raw_text()
    text = text[: int(0.9 * len(text))]
    chars = sorted(set(text) - {train.EOT})
    words = collections.Counter(u for u in train.units(text) if u != train.EOT)
    seqs = {w: tuple(w) for w in words}
    pieces = list(chars)
    while len(pieces) < size - 1:
        pairs = collections.Counter()
        for w, seq in seqs.items():
            for a, b in zip(seq, seq[1:]):
                pairs[a, b] += words[w]
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        pieces.append(a + b)
        for w, seq in seqs.items():
            if a in seq:
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        out.append(a + b)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                seqs[w] = tuple(out)
    pieces.append("")  # end of story
    dest = ROOT / "model" / f"vocab-{size}.json"
    dest.write_text(json.dumps({"pieces": pieces, "eos": len(pieces) - 1}))
    ids = train.encode_text(text[:2_000_000], pieces, len(pieces) - 1)
    print(f"wrote {dest.relative_to(ROOT)}: {len(pieces)} pieces, {2_000_000 / len(ids):.2f} chars/token")
    print("longest pieces:", sorted(pieces, key=len)[-12:])


if __name__ == "__main__":
    main()
