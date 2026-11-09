// Compile the quantization-aware-trained model (model/qat.json, from
// scripts/train.py) into:
//   model/program.json      the integer program run by src/reference/int.ts
//   src/model/program.ts    the same program as TypeScript types for src/engine
//
// All rounding and every table is computed here, once, in float64, so the
// reference and the type-level engine share them exactly.
//
// Run: node scripts/compile.ts
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import type { Layer, Lin, Program, Thresholds } from "../src/reference/int.ts";

const root = new URL("../", import.meta.url);
const qat = JSON.parse(readFileSync(new URL(process.env.QAT ?? "model/qat.json", root), "utf8"));
const { n_layer, n_head, n_embd, block_size } = qat.config;
const hd = n_embd / n_head;
const R = qat.ranges as Record<string, number>;

/** Round half away from zero (the spec's rounding). */
const rnd = (x: number) => Math.sign(x) * Math.floor(Math.abs(x) + 0.5) || 0;
const geluTanh = (x: number) =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

/** Increasing thresholds for |a| -> round(|a| * m), 2^W entries. */
const linear = (m: number, w: number): Thresholds =>
  Array.from({ length: 2 ** w }, (_, q) => (q === 0 ? 0 : Math.ceil((q - 0.5) / m)));

/** Thresholds for a non-increasing f on d >= 0: t[q] = 1 + max{d : f(d) >= q}. */
function decreasing(f: (d: number) => number, w: number): Thresholds {
  const t: Thresholds = Array.from({ length: 2 ** w }, () => 0);
  for (let d = 0; ; d++) {
    const v = f(d);
    for (let q = 1; q <= Math.min(v, 2 ** w - 1); q++) t[q] = d + 1;
    if (v === 0) break;
  }
  t[0] = Number.MAX_SAFE_INTEGER;
  return t;
}

/** Increasing thresholds for f on [lo, hi]: t[l] = min{s : f(s) >= l}; hi + 1 when unreachable. */
function increasing(f: (s: number) => number, lo: number, hi: number): Thresholds {
  const max = f(hi);
  const w = Math.max(1, Math.ceil(Math.log2(max + 1)));
  const t: Thresholds = Array.from({ length: 2 ** w }, (_, l) => (l === 0 ? 0 : hi + 1));
  for (let s = hi; s >= lo; s--) for (let l = 1; l <= f(s); l++) t[l] = s;
  return t;
}

const lin = (l: { w: number[][]; gamma: number; b: number[] | null }, inScale: number): Lin => ({
  w: l.w,
  b: l.b ? l.b.map((b) => rnd(b / (l.gamma * inScale))) : null,
});

const sr: number = qat.sr;
const layers: Layer[] = qat.layers.map((L: Record<string, any>) => {
  const u = (L.sq * L.sk) / Math.sqrt(hd);
  const sMax = R.E! * block_size;
  return {
    q: lin(L.q, 1 / 32),
    k: lin(L.k, 1 / 32),
    v: lin(L.v, 1 / 32),
    proj: lin(L.proj, L.so),
    fc: lin(L.fc, 1 / 32),
    mlp: lin(L.mlp, L.sh),
    rqQ: linear(L.q.gamma / (32 * L.sq), 4),
    rqK: linear(L.k.gamma / (32 * L.sk), 4),
    rqV: linear(L.v.gamma / (32 * L.sv), 4),
    exp: decreasing((d) => rnd(R.E! * Math.exp(-d * u)), 7),
    lse: increasing((s) => rnd(Math.log(s / R.E!) / u), R.E!, sMax),
    prob: decreasing((d) => rnd(R.P! * Math.exp(-d * u)), 6),
    rqO: linear(L.sv / (R.P! * L.so), 6),
    rqProj: linear((L.proj.gamma * L.so) / sr, 8),
    rqFc: linear(L.fc.gamma / (32 * L.sg), 6),
    gelu: Array.from({ length: 127 }, (_, i) =>
      Math.max(-63, Math.min(63, rnd(geluTanh((i - 63) * L.sg) / L.sh))),
    ),
    rqMlp: linear((L.mlp.gamma * L.sh) / sr, 8),
  };
});

const sqrt: Record<string, number> = {};
const sigma = (ss: number) => Math.max(1, rnd(Math.sqrt(ss / n_embd)));
for (let s = 0; s < 1000; s++) sqrt[String(s)] = sigma(s);
for (let digits = 4; digits <= 8; digits++)
  for (let m = 100; m < 1000; m++)
    sqrt[String(m) + "0".repeat(digits - 3)] = sigma(m * 10 ** (digits - 3));

/**
 * Sampling noise: Gumbel-max with a fixed seed. noise[pos][v] = round(T * g * 32 / gamma_head)
 * in head-accumulator units, g ~ Gumbel(0, 1) from a fixed LCG, T = 0.6. argmax(acc + noise)
 * samples from softmax(logits / T); the seed makes it reproducible.
 */
const TEMPERATURE = 0.6;
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) + 0.5) / 2 ** 31;
const noise: number[][] = Array.from({ length: block_size }, () =>
  (qat.vocab as string[]).map(() =>
    Math.round(-Math.log(-Math.log(rand())) * TEMPERATURE * (32 / qat.head.gamma)),
  ),
);

const program: Program = {
  config: qat.config,
  vocab: qat.vocab,
  etok: qat.wte.map((r: number[]) => r.map((v) => rnd(v / sr))),
  epos: qat.wpe.map((r: number[]) => r.map((v) => rnd(v / sr))),
  layers,
  head: lin(qat.head, 1 / 32),
  mean: Array.from({ length: 256 }, (_, q) => (q === 0 ? 0 : Math.ceil((q - 0.5) * n_embd))),
  sqrt,
  noise,
};
writeFileSync(new URL("model/program.json", root), JSON.stringify(program));

// ---------------------------------------------------------------- types

const u = (n: number) => "1".repeat(n);
const bits = (q: number, w: number) => q.toString(2).padStart(w, "0");
const widthOf = (t: Thresholds) => Math.log2(t.length);
const str = (s: string) => JSON.stringify(s);
/**
 * A threshold table as a binary decision tree: [threshold, ifLess, ifGreaterOrEqual]
 * with outputs at the leaves. Thresholds are written as ToS<n>, which the checker
 * resolves once when the table is declared, so walking the tree costs a couple of
 * instantiations per level (see Walk in src/engine).
 * dir "ge": q = largest with |a| >= t[q] (t non-decreasing).
 * dir "lt": q = largest with  a <  t[q] (t non-increasing).
 */
/**
 * A threshold table as a nested conditional type over the magnitude M: one
 * prefix match per level, no tuple destructuring (cheaper than walking tree
 * data; used for the per-vector requantizations).
 */
function condTree(
  t: Thresholds,
  leaf: (q: number) => string,
  dir: "ge" | "lt" = "ge",
  lo = 0,
  hi = t.length - 1,
): string {
  if (lo === hi) return leaf(lo);
  const mid = (lo + hi + 1) >> 1;
  const high = condTree(t, leaf, dir, mid, hi);
  const low = condTree(t, leaf, dir, lo, mid - 1);
  return `M extends \`\${ToS<${t[mid]}>}\${string}\` ? ${dir === "ge" ? high : low} : ${dir === "ge" ? low : high}`;
}
/** Vector requantization alias: sign handled outside the magnitude tree. */
function rqAlias(name: string, t: Thresholds, out: "num" | "una"): string {
  const leaf =
    out === "num" ? (q: number) => String(q) : (q: number) => (q === 0 ? '""' : `ToS<${q}>`);
  const neg = out === "num" ? "NegNum" : "NegS";
  return `type ${name}_m<M extends string> = ${condTree(t, leaf)};
export type ${name}<A> = { [K in keyof A]: A[K] extends \`-\${infer M}\` ? ${neg}<${name}_m<M>> : ${name}_m<A[K] & string> };
`;
}

function tree(
  t: Thresholds,
  dir: "ge" | "lt",
  leaf: (q: number) => string,
  lo = 0,
  hi = t.length - 1,
): string {
  if (lo === hi) return leaf(lo);
  const mid = (lo + hi + 1) >> 1;
  const low = tree(t, dir, leaf, lo, mid - 1);
  const high = tree(t, dir, leaf, mid, hi);
  return dir === "ge" ? `[ToS<${t[mid]}>, ${low}, ${high}]` : `[ToS<${t[mid]}>, ${high}, ${low}]`;
}
const num = (q: number) => String(q);
const una = (q: number) => (q === 0 ? '""' : `ToS<${q}>`);

/**
 * One BitLinear as a type alias over split inputs (XP[i] = positive part of x_i,
 * XN[i] = negative magnitude, unary): row j concatenates the terms that add
 * (XP for w = +1, XN for w = -1) into one string and those that subtract into
 * another, repeated |w| times for the int4 head, and subtracts. (Offsetting the
 * inputs to be non-negative halves the spans but makes every string ~127
 * characters per input longer, which measured slower.)
 */
function matrix(name: string, l: Lin): string {
  const rows = l.w.map((row, j) => {
    const p: string[] = [];
    const n: string[] = [];
    const b = l.b ? l.b[j]! : 0;
    if (b > 0) p.push(`\${ToS<${b}>}`);
    if (b < 0) n.push(`\${ToS<${-b}>}`);
    row.forEach((w, i) => {
      for (let r = 0; r < Math.abs(w); r++) {
        (w > 0 ? p : n).push(`\${XP[${i}]}`);
        (w > 0 ? n : p).push(`\${XN[${i}]}`);
      }
    });
    return `  SubU<\`${p.join("")}\`, \`${n.join("")}\`>,`;
  });
  return `export type ${name}<XP extends Strs, XN extends Strs> = [\n${rows.join("\n")}\n];\n`;
}
const out: string[] = [];
const lse = layers.map((L) => widthOf(L.lse));
out.push(`// Generated by scripts/compile.ts from model/qat.json. Do not edit.
//
// The integer GPT-2 program as types. Values are signed unary strings ("111" is 3,
// "-11" is -2, "" is 0). Each matrix is a type alias whose row j concatenates the
// inputs with weight +1 into one string and those with weight -1 into another
// (biases are constant runs of 1s), and subtracts: a ternary mat-vec with no
// multiplication. Threshold tables are keyed by W-bit strings for binary search.
import type { NegNum, NegS, SubU, ToS } from "../num/ustr.ts";

type Strs = readonly string[];

export type Config = { nLayer: ${n_layer}; nHead: ${n_head}; nEmbd: ${n_embd}; blockSize: ${block_size} };
export type Vocab = [${program.vocab.map(str).join(", ")}];
export type CharToId = { ${program.vocab.map((c, i) => `${str(c)}: ${i}`).join("; ")} };
export type Etok = [${program.etok.map((r) => `[${r.join(", ")}]`).join(",\n  ")}];
export type Epos = [${program.epos.map((r) => `[${r.join(", ")}]`).join(",\n  ")}];
export type Mean = ${tree(program.mean, "ge", una)};
export type Sqrt = { ${Object.entries(sqrt)
  .map(([k, v]) => `"${k}": ${v}`)
  .join("; ")} };
`);
layers.forEach((L, l) => {
  out.push(`export type Layer${l} = {
  rqQ: ${tree(L.rqQ, "ge", num)};
  rqK: ${tree(L.rqK, "ge", num)};
  rqV: ${tree(L.rqV, "ge", num)};
  exp: ${tree(L.exp, "lt", una)};
  lse: ${tree(L.lse, "ge", una)};
  prob: ${tree(L.prob, "lt", num)};
  rqO: ${tree(L.rqO, "ge", una)};
  rqProj: ${tree(L.rqProj, "ge", una)};
  rqFc: ${tree(L.rqFc, "ge", num)};
  gelu: { ${L.gelu.map((h, i) => `"${i - 63}": ${str(h < 0 ? `-${u(-h)}` : u(h))}`).join("; ")} };
  rqMlp: ${tree(L.rqMlp, "ge", una)};
};
`);
  for (const m of ["q", "k", "v", "proj", "fc", "mlp"] as const)
    out.push(matrix(`L${l}_${m}`, L[m]));
});
out.push(`export type Layers = [${layers.map((_, l) => `Layer${l}`).join(", ")}];\n`);
const RQ_TABLES = [
  ["rqQ", "num"],
  ["rqK", "num"],
  ["rqV", "num"],
  ["rqProj", "una"],
  ["rqFc", "num"],
  ["rqMlp", "una"],
] as const;
layers.forEach((L, l) => {
  for (const [name, kind] of RQ_TABLES) out.push(rqAlias(`L${l}_${name}`, L[name], kind));
});
layers.forEach((L, l) => {
  const una = (q: number) => (q === 0 ? '""' : `ToS<${q}>`);
  out.push(`type L${l}_exp_m<M extends string> = ${condTree(L.exp, una, "lt")};
type L${l}_prob_m<M extends string> = ${condTree(L.prob, String, "lt")};
`);
});
out.push(`/** Softmax numerators e_j for distances D (decreasing table), per layer. */
export type ExpVec<L extends number, D> = ${layers
  .map((_, l) => `L extends ${l}\n  ? { [J in keyof D]: L${l}_exp_m<D[J] & string> }\n  : `)
  .join("")}never;
/** Probabilities p_j for distances D shifted by the log-sum-exp Lse, per layer. */
export type ProbVec<L extends number, D, Lse extends string> = ${layers
  .map(
    (_, l) =>
      `L extends ${l}\n  ? { [J in keyof D]: L${l}_prob_m<\`\${D[J] & string}\${Lse}\`> }\n  : `,
  )
  .join("")}never;
`);
out.push(`/** Dispatch a layer's vector requantization by name and index. */
export type RQVec<N extends string, L extends number, A> = ${layers
  .flatMap((_, l) =>
    RQ_TABLES.map(([m]) => `[N, L] extends ["${m}", ${l}]\n  ? L${l}_${m}<A>\n  : `),
  )
  .join("")}never;
`);
out.push(`/** Sampling noise per position (Gumbel, T = ${TEMPERATURE}, fixed seed), head-accumulator units. */
export type Noise = [\n${noise.map((r) => `  [${r.map((v) => (v === 0 ? '""' : `ToS<${v}>`)).join(", ")}],`).join("\n")}\n];\n`);
out.push(matrix("Head", program.head));
{
  const v = (i: number) => `A${i}`;
  const all = Array.from({ length: n_embd }, (_, i) => i);
  const heads = Array.from({ length: n_head }, (_, h) => all.slice(h * hd, (h + 1) * hd));
  out.push(`/** Split a ${n_embd}-vector into ${n_head} heads of ${hd}, and back. */
export type Heads<X> = X extends readonly [${all.map((i) => `infer ${v(i)}`).join(", ")}]
  ? [${heads.map((h) => `[${h.map(v).join(", ")}]`).join(", ")}]
  : never;
export type FlatHeads<H> = H extends readonly [${heads.map((_, h) => `infer H${h} extends readonly string[]`).join(", ")}]
  ? [${heads.map((_, h) => `...H${h}`).join(", ")}]
  : never;
`);
}
out.push(`/** Dispatch a layer's matrix by name and index (type aliases cannot be passed around). */
export type Mat<N extends string, L extends number, XP extends Strs, XN extends Strs> = ${layers
  .flatMap((_, l) =>
    ["q", "k", "v", "proj", "fc", "mlp"].map(
      (m) => `[N, L] extends ["${m}", ${l}]\n  ? L${l}_${m}<XP, XN>\n  : `,
    ),
  )
  .join("")}never;
`);

// Shared tables.
const unaryTable = (w: number, f: (q: number) => number = (q) => q) =>
  `{ ${Array.from({ length: 2 ** w }, (_, q) => `"${bits(q, w)}": ${str(u(f(q)))}`).join("; ")} }`;
const numberTable = (w: number) =>
  `{ ${Array.from({ length: 2 ** w }, (_, q) => `"${bits(q, w)}": ${q}`).join("; ")} }`;
const maxW = Math.max(8, ...lse);
out.push(`
/** W-bit string -> number. */
${[4, 6, 7, 8].map((w) => `export type BitsN${w} = ${numberTable(w)};`).join("\n")}
/** W-bit string -> unary string. */
${[6, 7, 8].map((w) => `export type BitsU${w} = ${unaryTable(w)};`).join("\n")}
/** LayerNorm: floor(64|c| / sigma) as 8 bits -> |z| = min(127, (q + 1) >> 1) in unary. */
export type BitsZ = ${unaryTable(8, (q) => Math.min(R.Z!, (q + 1) >> 1))};
/** |c| in unary (0..510) -> c^2 as two unary limbs [hi, lo] with c^2 = 512 hi + lo. */
export type Squares = { ${Array.from({ length: 2 * R.R! + 1 }, (_, c) => `${str(u(c))}: [${una((c * c) >> 9)}, ${una((c * c) & 511)}]`).join("; ")} };
/** Carry of the low limb: floor(LO / 512), as a tree over LO with unary leaves [carry, 512 carry]. */
export type Carry = ${tree(
  Array.from({ length: 64 }, (_, k) => 512 * k),
  "ge",
  (k) => `[${una(k)}, ${una(512 * k)}]`,
)};
/**
 * sigma from the sum of squares SS = 512 HI + LO (LO < 512): a tree of
 * [ToS<hi + 1>, ToS<hi>, ToS<lo>, ifLess, ifGreaterOrEqual] nodes comparing
 * SS >= 512 hi + lo; leaves are sigma in unary. Same function as the sqrt
 * table (3 significant digits), so the reference is unchanged.
 */
export type Sigma = ${sigmaTree()};
/** q * k for q, k in [-15, 15]: positive part and negative part, unary. */
export type ProdP = { ${range(-15, 15)
  .map(
    (a) =>
      `"${a}": { ${range(-15, 15)
        .map((b) => `"${b}": ${str(a * b > 0 ? u(a * b) : "")}`)
        .join("; ")} }`,
  )
  .join(";\n  ")} };
export type ProdN = { ${range(-15, 15)
  .map(
    (a) =>
      `"${a}": { ${range(-15, 15)
        .map((b) => `"${b}": ${str(a * b < 0 ? u(-a * b) : "")}`)
        .join("; ")} }`,
  )
  .join(";\n  ")} };
/** p * v for p in [0, 63], v in [-15, 15], signed unary (ToS resolves once per entry). */
export type ProdPV = { ${range(0, 63)
  .map(
    (p) =>
      `"${p}": { ${range(-15, 15)
        .map((v) => `"${v}": ${p * v === 0 ? '""' : `ToS<${p * v}>`}`)
        .join("; ")} }`,
  )
  .join(";\n  ")} };
`);
void maxW;

function sigmaTree(): string {
  const sig = (ss: number) => sqrt[sqrtKeyOf(ss)]!;
  const maxSS = n_embd * (2 * R.R!) ** 2;
  const t = Array.from({ length: 512 }, (_, s) => {
    if (s <= 1) return 0;
    if (sig(maxSS) < s) return maxSS + 1;
    let lo = 0;
    let hi = maxSS;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (sig(mid) >= s) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  });
  const node = (lo: number, hi: number): string => {
    if (lo === hi) return una(Math.max(1, lo));
    const mid = (lo + hi + 1) >> 1;
    const T = t[mid]!;
    return `[${una((T >> 9) + 1)}, ${una(T >> 9)}, ${una(T & 511)}, ${node(lo, mid - 1)}, ${node(mid, hi)}]`;
  };
  return node(0, 511);
}
function sqrtKeyOf(ss: number) {
  const s = String(ss);
  return s.length <= 3 ? s : s.slice(0, 3) + "0".repeat(s.length - 3);
}

function range(a: number, b: number) {
  return Array.from({ length: b - a + 1 }, (_, i) => a + i);
}

mkdirSync(new URL("src/model/", root), { recursive: true });
writeFileSync(new URL("src/model/program.ts", root), out.join("\n"));
console.log(
  `wrote model/program.json and src/model/program.ts (${(out.join("\n").length / 1024).toFixed(0)} KiB); lse widths ${lse.join(",")}`,
);
