/**
 * Runtime reference for the fixed-point GPT-2 that the type checker runs.
 *
 * Every value is a bigint holding `round(real * 10^3)`. Every operation here is
 * mirrored one-to-one by a type in src/gpt2, with the same rounding, so the two
 * must produce identical digits. This file is the spec; the types are the
 * implementation under test.
 */
import type { FixedModel } from "./quantize.ts";

export const SCALE = 1000n;

/** round(v / 10^k), half away from zero. */
export function roundShift(v: bigint, k: number): bigint {
  const p = 10n ** BigInt(k);
  const half = p / 2n;
  return v < 0n ? -((-v + half) / p) : (v + half) / p;
}

/** round(a / b), half away from zero. */
export function divRound(a: bigint, b: bigint): bigint {
  const neg = a < 0n !== b < 0n;
  const ma = a < 0n ? -a : a;
  const mb = b < 0n ? -b : b;
  const q = (2n * ma + mb) / (2n * mb);
  return neg ? -q : q;
}

/** floor(sqrt(v)) by Newton's method, starting from 10^ceil(digits/2). */
export function isqrt(v: bigint): bigint {
  if (v === 0n) return 0n;
  let r = 10n ** BigInt(Math.ceil(v.toString().length / 2));
  for (;;) {
    const next = (r + v / r) / 2n;
    if (next >= r) return r;
    r = next;
  }
}

/** exp() lookup tables at scale 10^6: e^-i, e^-(d/10), e^-(d/100), e^-(d/1000). */
export const EXP_TABLES = {
  int: Array.from({ length: 8 }, (_, i) => BigInt(Math.round(Math.exp(-i) * 1e6))),
  d1: Array.from({ length: 10 }, (_, d) => BigInt(Math.round(Math.exp(-d / 10) * 1e6))),
  d2: Array.from({ length: 10 }, (_, d) => BigInt(Math.round(Math.exp(-d / 100) * 1e6))),
  d3: Array.from({ length: 10 }, (_, d) => BigInt(Math.round(Math.exp(-d / 1000) * 1e6))),
};

/** e^x for x <= 0, from the digits of -x: e^-(i.abc) = e^-i * e^-.a * e^-.0b * e^-.00c. */
export function exp(x: bigint): bigint {
  const m = -x;
  if (m >= 8000n) return 0n;
  const i = Number(m / 1000n);
  const d1 = Number((m / 100n) % 10n);
  const d2 = Number((m / 10n) % 10n);
  const d3 = Number(m % 10n);
  let r = EXP_TABLES.int[i]!;
  r = roundShift(r * EXP_TABLES.d1[d1]!, 6);
  r = roundShift(r * EXP_TABLES.d2[d2]!, 6);
  r = roundShift(r * EXP_TABLES.d3[d3]!, 6);
  return roundShift(r, 3);
}

/** GELU, tanh approximation, written as x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)). */
export function gelu(x: bigint): bigint {
  const x3 = roundShift(roundShift(x * x, 3) * x, 3);
  const inner = x + roundShift(x3 * 44715n, 6);
  const z = roundShift(inner * 1595769n, 6);
  const e = exp(z < 0n ? z : -z);
  const s = divRound(1_000_000n, 1000n + e);
  return roundShift(x * (z < 0n ? 1000n - s : s), 3);
}

export function dot(x: readonly bigint[], w: readonly bigint[]): bigint {
  let s = 0n;
  for (let i = 0; i < x.length; i++) s += x[i]! * w[i]!;
  return roundShift(s, 3);
}

export function matvec(
  w: readonly (readonly bigint[])[],
  x: readonly bigint[],
  b?: readonly bigint[],
): bigint[] {
  return w.map((row, j) => dot(x, row) + (b ? b[j]! : 0n));
}

export function layerNorm(
  x: readonly bigint[],
  g: readonly bigint[],
  b: readonly bigint[],
): bigint[] {
  const n = BigInt(x.length);
  const mean = divRound(
    x.reduce((a, v) => a + v, 0n),
    n,
  );
  const c = x.map((v) => v - mean);
  const variance = divRound(
    c.reduce((a, v) => a + v * v, 0n),
    n,
  );
  const std = isqrt(variance + 10n); // eps = 1e-5 at scale 10^6
  const inv = divRound(10n ** 9n, std);
  return c.map((v, i) => roundShift(roundShift(v * inv, 6) * g[i]!, 3) + b[i]!);
}

export function softmax(s: readonly bigint[]): bigint[] {
  const m = s.reduce((a, v) => (v > a ? v : a));
  const e = s.map((v) => exp(v - m));
  const inv = divRound(
    10n ** 9n,
    e.reduce((a, v) => a + v, 0n),
  );
  return e.map((v) => roundShift(v * inv, 6));
}

export function argmax(v: readonly bigint[]): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i]! > v[best]!) best = i;
  return best;
}

/** Greedy generation with a KV cache, exactly as the type-level state machine runs it. */
export function generate(model: FixedModel, prompt: number[], steps: number): number[] {
  const { n_head, n_embd, block_size } = model.config;
  const hd = n_embd / n_head;
  const tokens = [...prompt];
  const cache = model.layers.map(() => ({ k: [] as bigint[][], v: [] as bigint[][] }));

  for (let pos = 0; pos < tokens.length && pos < block_size; pos++) {
    let x = model.wte[tokens[pos]!]!.map((v, i) => v + model.wpe[pos]![i]!);
    model.layers.forEach((L, l) => {
      const qkv = matvec(L.attnW, layerNorm(x, L.ln1g, L.ln1b), L.attnB);
      const q = qkv.slice(0, n_embd);
      cache[l]!.k.push(qkv.slice(n_embd, 2 * n_embd));
      cache[l]!.v.push(qkv.slice(2 * n_embd));
      const out: bigint[] = [];
      for (let h = 0; h < n_head; h++) {
        const sl = (v: bigint[]) => v.slice(h * hd, (h + 1) * hd);
        const p = softmax(cache[l]!.k.map((k) => dot(sl(q), sl(k))));
        for (let d = 0; d < hd; d++) {
          out.push(
            dot(
              p,
              cache[l]!.v.map((v) => v[h * hd + d]!),
            ),
          );
        }
      }
      const attn = matvec(L.projW, out, L.projB);
      x = x.map((v, i) => v + attn[i]!);
      const hid = matvec(L.fcW, layerNorm(x, L.ln2g, L.ln2b), L.fcB).map(gelu);
      const y = matvec(L.mlpW, hid, L.mlpB);
      x = x.map((v, i) => v + y[i]!);
    });
    if (pos === tokens.length - 1 && tokens.length - prompt.length < steps) {
      tokens.push(argmax(matvec(model.wte, layerNorm(x, model.lnfG, model.lnfB))));
    }
  }
  return tokens;
}
