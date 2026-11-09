import { readFileSync } from "node:fs";
import { describe, expect, test } from "vite-plus/test";
import {
  argmax,
  attendHead,
  decode,
  encode,
  forward,
  generate,
  layerNorm,
  matvec,
  type Program,
  R_MAX,
  rq,
  searchGe,
  searchLt,
  sqrtKey,
  Z_MAX,
} from "../src/reference/int.ts";

const p = JSON.parse(
  readFileSync(new URL("../model/program.json", import.meta.url), "utf8"),
) as Program;
const nonDecreasing = (t: number[]) => t.slice(1).every((v, i) => v >= t[i]!);

describe("threshold tables", () => {
  test("searchGe is the largest q with a >= t[q]", () => {
    const t = [0, 3, 3, 10, 11, 20, 20, 50];
    for (let a = 0; a < 60; a++) {
      const q = searchGe(a, t);
      expect(a >= t[q]!).toBe(true);
      expect(q === t.length - 1 || a < t[q + 1]!).toBe(true);
    }
  });

  test("searchLt is the largest q with a < t[q] on a non-increasing table", () => {
    const t = [Infinity, 40, 30, 30, 9, 5, 1, 1];
    for (let a = 0; a < 50; a++) {
      const q = searchLt(a, t);
      expect(q === 0 || a < t[q]!).toBe(true);
      expect(q === t.length - 1 || a >= t[q + 1]!).toBe(true);
    }
  });

  test("rq is symmetric", () => {
    const t = p.layers[0]!.rqQ;
    for (let a = 0; a < 3000; a += 37) expect(rq(-a, t) + rq(a, t)).toBe(0);
  });

  test("compiled tables are monotone and power-of-two sized", () => {
    for (const L of p.layers) {
      for (const t of [L.rqQ, L.rqK, L.rqV, L.lse, L.rqO, L.rqProj, L.rqFc, L.rqMlp]) {
        expect(nonDecreasing(t)).toBe(true);
        expect(Number.isInteger(Math.log2(t.length))).toBe(true);
      }
      for (const t of [L.exp, L.prob]) expect(nonDecreasing(t.slice(1).reverse())).toBe(true);
    }
    expect(nonDecreasing(p.mean)).toBe(true);
  });

  test("weights are ternary (head int4)", () => {
    for (const L of p.layers)
      for (const m of [L.q, L.k, L.v, L.proj, L.fc, L.mlp])
        expect(m.w.flat().every((w) => w >= -1 && w <= 1)).toBe(true);
    expect(p.head.w.flat().every((w) => w >= -7 && w <= 7)).toBe(true);
  });
});

describe("integer ops", () => {
  test("sqrtKey keeps 3 significant digits", () => {
    expect(sqrtKey(0)).toBe("0");
    expect(sqrtKey(999)).toBe("999");
    expect(sqrtKey(123456)).toBe("123000");
  });

  test("layerNorm output is centered, scaled and clamped", () => {
    const x = Array.from({ length: p.config.n_embd }, (_, i) => ((i * 37) % 200) - 100);
    const z = layerNorm(x, p);
    expect(z.every((v) => Math.abs(v) <= Z_MAX)).toBe(true);
    const mean = z.reduce((a, v) => a + v, 0) / z.length;
    const rms = Math.sqrt(z.reduce((a, v) => a + v * v, 0) / z.length);
    expect(Math.abs(mean)).toBeLessThan(2);
    expect(Math.abs(rms - 32)).toBeLessThan(2); // unit variance at resolution 1/32
  });

  test("attention probabilities sum to about 63", () => {
    const L = p.layers[0]!;
    const q = [3, -2, 5, 0, 1, -7, 2, 4];
    const ks = Array.from({ length: 10 }, (_, j) => q.map((v, i) => ((v * (j + 1) + i) % 15) - 7));
    const ones = ks.map(() => q.map(() => 1));
    const sums = attendHead(L, q, ks, ones);
    for (const s of sums) expect(Math.abs(s - 63)).toBeLessThanOrEqual(ks.length);
  });
});

describe("generation", () => {
  test("residual stream stays in range", () => {
    const out = generate(p, encode(p, "Once"), 30);
    expect(out.length).toBe(34);
    expect(R_MAX).toBe(255);
  });

  test("sampling is seeded and differs from greedy", () => {
    const prompt = encode(p, "Once upon a time");
    const a = decode(p, generate(p, prompt, 24, "sample"));
    expect(decode(p, generate(p, prompt, 24, "sample"))).toBe(a);
    expect(decode(p, generate(p, prompt, 24, "greedy"))).not.toBe(a);
  });

  test("stops at the context size", () => {
    expect(generate(p, encode(p, "Once"), 500).length).toBe(p.config.block_size + 1);
  });
});

describe("parity with training", () => {
  // model/parity.json holds LM-head accumulators computed by the PyTorch integer
  // forward pass of scripts/train.py (see scripts/parity.py). The reference,
  // whose tables scripts/compile.ts rebuilds independently, must reproduce them.
  const parity = JSON.parse(
    readFileSync(new URL("../model/parity.json", import.meta.url), "utf8"),
  ) as { tokens: number[][]; acc: number[][][] };

  test("the reference reproduces the PyTorch integer forward pass exactly", () => {
    expect(parity.tokens.length).toBeGreaterThan(0);
    parity.tokens.forEach((window, w) => {
      const cache = p.layers.map(() => ({ k: [] as number[][][], v: [] as number[][][] }));
      for (let pos = 0; pos < window.length - 1; pos++) {
        const acc = matvec(p.head, layerNorm(forward(p, window[pos]!, pos, cache), p));
        expect(acc, `window ${w}, position ${pos}`).toEqual(parity.acc[w]![pos]);
        expect(argmax(acc)).toBe(argmax(parity.acc[w]![pos]!));
      }
    });
  });
});
