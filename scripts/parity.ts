// Compare the integer reference (src/reference/int.ts) with the PyTorch
// integer forward pass over the full parity set written by scripts/parity.py
// (.run/parity-full.json), and report the reference's own validation loss.
//
//   uv run scripts/parity.py && node scripts/parity.ts
import { readFileSync } from "node:fs";
import { forward, layerNorm, matvec, type Program } from "../src/reference/int.ts";

const root = new URL("../", import.meta.url);
const read = (p: string) => JSON.parse(readFileSync(new URL(p, root), "utf8"));
const program = read("model/program.json") as Program;
const { tokens, acc, gamma } = read(".run/parity-full.json") as {
  tokens: number[][];
  acc: number[][][];
  gamma: number;
};

let positions = 0;
let mismatched = 0;
let firstMismatch = "";
let loss = 0;
tokens.forEach((window, w) => {
  const cache = program.layers.map(() => ({ k: [] as number[][][], v: [] as number[][][] }));
  for (let pos = 0; pos < window.length - 1; pos++) {
    const x = forward(program, window[pos]!, pos, cache);
    const a = matvec(program.head, layerNorm(x, program));
    positions++;
    if (a.some((v, i) => v !== acc[w]![pos]![i])) {
      mismatched++;
      firstMismatch ||= `window ${w}, position ${pos}`;
    }
    const logits = a.map((v) => (v * gamma) / 32);
    const max = Math.max(...logits);
    const lse = max + Math.log(logits.reduce((s, l) => s + Math.exp(l - max), 0));
    loss += lse - logits[window[pos + 1]!]!;
  }
});
console.log(
  `reference vs PyTorch (float64): ${mismatched} of ${positions} positions differ${firstMismatch && ` (first: ${firstMismatch})`}`,
);
console.log(`reference validation loss: ${(loss / positions).toFixed(4)} nats/char`);
process.exitCode = mismatched ? 1 : 0;
