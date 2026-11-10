/**
 * Exact integer reference for the type-level GPT-2. This file is the spec:
 * src/engine computes the same function in the type system, digit for digit.
 *
 * Everything is an integer. Every non-linear step is a lookup in a table
 * built by scripts/compile.ts, either a plain table or a monotone threshold
 * table ("largest q with |a| >= t[q]", which the type checker evaluates as a
 * binary search over unary strings).
 */

export interface Lin {
  /** Ternary weights, one row per output. */
  w: number[][];
  /** Bias in accumulator units (null for the LM head). */
  b: number[] | null;
}

/** Monotone non-decreasing thresholds t[0..2^W-1] (t[0] = 0). */
export type Thresholds = number[];

export interface Layer {
  q: Lin;
  k: Lin;
  v: Lin;
  proj: Lin;
  fc: Lin;
  mlp: Lin;
  /** |acc| -> |q|, |k|, |v| (4 bits). */
  rqQ: Thresholds;
  rqK: Thresholds;
  rqV: Thresholds;
  /** Softmax: d -> e (7 bits, decreasing: e = max q with d < exp[q]). */
  exp: Thresholds;
  /** sum(e) -> L (increasing). */
  lse: Thresholds;
  /** d + L -> p (6 bits, decreasing like exp). */
  prob: Thresholds;
  /** |sum p v| -> |o| (6 bits). */
  rqO: Thresholds;
  /** |acc| -> |residual delta| (8 bits). */
  rqProj: Thresholds;
  /** |acc| -> |pre-GELU g| (6 bits). */
  rqFc: Thresholds;
  /** g in [-63, 63] -> GELU output h, indexed by g + 63. */
  gelu: number[];
  rqMlp: Thresholds;
}

export interface Program {
  config: {
    n_layer: number;
    n_head: number;
    n_embd: number;
    block_size: number;
    vocab_size: number;
  };
  vocab: string[];
  /** Embeddings in residual units. */
  etok: number[][];
  epos: number[][];
  layers: Layer[];
  head: Lin;
  /** |sum x| -> |mean| (8 bits). */
  mean: Thresholds;
  /** sum of squares (3 significant digits, see sqrtKey) -> sigma. */
  sqrt: Record<string, number>;
  /** Gumbel noise per position and token, head-accumulator units (seeded sampling). */
  noise: number[][];
}

export const R_MAX = 255;
export const Z_MAX = 127;

/** Largest q with a >= t[q] (t non-decreasing, t[0] = 0). */
export const searchGe = (a: number, t: Thresholds): number => {
  let q = 0;
  for (let bit = t.length >> 1; bit > 0; bit >>= 1) if (a >= t[q + bit]!) q += bit;
  return q;
};
/** Largest q with a < t[q] (t non-increasing); 0 if none. */
export const searchLt = (a: number, t: Thresholds): number => {
  let q = 0;
  for (let bit = t.length >> 1; bit > 0; bit >>= 1) if (a < t[q + bit]!) q += bit;
  return q;
};
/** Signed requantization through an increasing threshold table. */
export const rq = (a: number, t: Thresholds): number =>
  Math.sign(a) * searchGe(Math.abs(a), t) || 0;

const clamp = (x: number, m: number) => Math.max(-m, Math.min(m, x));

/** Key of the sqrt table: the sum of squares truncated to 3 significant digits. */
export const sqrtKey = (ss: number): string => {
  const s = String(ss);
  return s.length <= 3 ? s : s.slice(0, 3) + "0".repeat(s.length - 3);
};

/** LayerNorm without affine parameters, output in units of sigma / 32, clamped to +-127. */
export function layerNorm(x: readonly number[], p: Program): number[] {
  const mu = rq(
    x.reduce((a, v) => a + v, 0),
    p.mean,
  );
  const c = x.map((v) => v - mu);
  const sigma = p.sqrt[sqrtKey(c.reduce((a, v) => a + v * v, 0))]!;
  return c.map(
    (v) =>
      Math.sign(v) *
        Math.min(Z_MAX, Math.floor((Math.floor((64 * Math.abs(v)) / sigma) + 1) / 2)) || 0,
  );
}

/** Ternary mat-vec plus integer bias. */
export const matvec = (l: Lin, x: readonly number[]): number[] =>
  l.w.map((row, j) => row.reduce((s, w, i) => s + w * x[i]!, l.b ? l.b[j]! : 0));

export const argmax = (v: readonly number[]): number =>
  v.reduce((best, x, i) => (x > v[best]! ? i : best), 0);

/** Integer softmax-attention of one head. Returns sum_j p_j v_j per dimension. */
export function attendHead(L: Layer, q: number[], ks: number[][], vs: number[][]): number[] {
  const s = ks.map((k) => k.reduce((a, kv, i) => a + kv * q[i]!, 0));
  const m = Math.max(...s);
  const d = s.map((v) => m - v);
  const e = d.map((v) => searchLt(v, L.exp));
  const lse = searchGe(
    e.reduce((a, v) => a + v, 0),
    L.lse,
  );
  const prob = d.map((v) => searchLt(v + lse, L.prob));
  return q.map((_, i) => prob.reduce((a, pj, j) => a + pj * vs[j]![i]!, 0));
}

/** One position through the network; appends to the KV cache. Returns the final residual. */
export function forward(
  p: Program,
  token: number,
  pos: number,
  cache: { k: number[][][]; v: number[][][] }[],
): number[] {
  const { n_head, n_embd } = p.config;
  const hd = n_embd / n_head;
  let x = p.etok[token]!.map((v, i) => clamp(v + p.epos[pos]![i]!, R_MAX));
  p.layers.forEach((L, l) => {
    const z = layerNorm(x, p);
    const q = matvec(L.q, z).map((a) => rq(a, L.rqQ));
    const k = matvec(L.k, z).map((a) => rq(a, L.rqK));
    const v = matvec(L.v, z).map((a) => rq(a, L.rqV));
    const heads = (t: number[]) =>
      Array.from({ length: n_head }, (_, h) => t.slice(h * hd, (h + 1) * hd));
    cache[l]!.k.push(heads(k));
    cache[l]!.v.push(heads(v));
    const o = heads(q).flatMap((qh, h) =>
      attendHead(
        L,
        qh,
        cache[l]!.k.map((kk) => kk[h]!),
        cache[l]!.v.map((vv) => vv[h]!),
      ).map((a) => rq(a, L.rqO)),
    );
    x = matvec(L.proj, o).map((a, i) => clamp(x[i]! + rq(a, L.rqProj), R_MAX));
    const g = matvec(L.fc, layerNorm(x, p)).map((a) => rq(a, L.rqFc));
    const h = g.map((v) => L.gelu[v + 63]!);
    x = matvec(L.mlp, h).map((a, i) => clamp(x[i]! + rq(a, L.rqMlp), R_MAX));
  });
  return x;
}

/**
 * Generate N tokens after the prompt (stops at the context size). "sample" adds
 * the seeded Gumbel noise of the generating position before the argmax
 * (Gumbel-max sampling at temperature 0.6); "greedy" takes the plain argmax.
 * Seed rotates the stored noise sequence; valid integers are 0..block_size-1.
 */
export function generate(
  p: Program,
  prompt: number[],
  n: number,
  mode: "sample" | "greedy" = "sample",
  seed = 0,
): number[] {
  if (!Number.isInteger(seed) || seed < 0 || seed >= p.config.block_size) {
    throw new RangeError(`seed must be an integer from 0 to ${p.config.block_size - 1}`);
  }
  const tokens = [...prompt];
  const cache = p.layers.map(() => ({ k: [] as number[][][], v: [] as number[][][] }));
  for (let pos = 0; pos < tokens.length && pos < p.config.block_size; pos++) {
    const x = forward(p, tokens[pos]!, pos, cache);
    if (pos === tokens.length - 1 && tokens.length - prompt.length < n) {
      const acc = matvec(p.head, layerNorm(x, p));
      const noise = p.noise[(pos + seed) % p.config.block_size]!;
      tokens.push(argmax(mode === "sample" ? acc.map((a, v) => a + noise[v]!) : acc));
    }
  }
  return tokens;
}

export const encode = (p: Program, s: string): number[] =>
  s.split("").map((c) => {
    const i = p.vocab.indexOf(c);
    if (i < 0) throw new Error(`character ${JSON.stringify(c)} is not in the vocabulary`);
    return i;
  });
export const decode = (p: Program, ids: number[]): string => ids.map((i) => p.vocab[i]).join("");
