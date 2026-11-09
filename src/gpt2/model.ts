/**
 * GPT-2 as a type-level state machine.
 *
 * TypeScript allows about 5M type instantiations per statement before it gives
 * up with TS2589, so a whole forward pass cannot happen inside one type. The
 * model runs instead as a sequence of ticks: `Step<M, S>` does one bounded
 * piece of work (one sublayer for one position) and returns the next state.
 * A "clock" file chains the ticks, one per statement:
 *
 *   type S0 = Init<Model, "ROMEO:", 16>;
 *   type S1 = Step<Model, S0>;
 *   type S2 = Step<Model, S1>;
 *   ...
 *
 * Per position the ticks are: embed, then for each layer qkv, one attn tick
 * per head, oproj, fc, gelu, proj, and finally head (only once the prompt is consumed), which appends the
 * argmax token. Prompt positions only fill the KV cache. Mirrors `generate` in
 * src/reference/fixed.ts.
 */
import type { AddVec, Argmax, Gelu, LayerNorm, Mat, MatVec, Softmax, Vec } from "./ops.ts";
import type { Dot } from "../num/index.ts";

/** Weights for one transformer block; Q/K/V are pre-split into heads. */
export interface LayerWeights {
  ln1g: Vec;
  ln1b: Vec;
  qW: readonly Mat[];
  qB: readonly Vec[];
  kW: readonly Mat[];
  kB: readonly Vec[];
  vW: readonly Mat[];
  vB: readonly Vec[];
  projW: Mat;
  projB: Vec;
  ln2g: Vec;
  ln2b: Vec;
  fcW: Mat;
  fcB: Vec;
  mlpW: Mat;
  mlpB: Vec;
}

export interface Model {
  nLayer: number;
  blockSize: number;
  vocab: readonly string[];
  charToId: Record<string, number>;
  wte: Mat;
  wpe: Mat;
  layers: readonly LayerWeights[];
  lnfG: Vec;
  lnfB: Vec;
}

type Phase = "embed" | "qkv" | "attn" | "oproj" | "fc" | "gelu" | "proj" | "head" | "done";

/** KV cache of one layer, [keys, values]: per position, per head, a head-sized vector. */
type Cache = readonly [k: readonly (readonly Vec[])[], v: readonly (readonly Vec[])[]];

/**
 * Machine state. It is a tuple, not an object: tuple elements are resolved as
 * soon as the tuple is built, while object members are resolved lazily and
 * would chain each state back to the previous one. Read fields with `S[I["x"]]`.
 */
export type State = readonly [
  phase: Phase,
  tokens: readonly number[],
  /** Counters are tuples; the count is their length. */
  pos: readonly unknown[],
  layer: readonly unknown[],
  head: readonly unknown[],
  left: readonly unknown[],
  /** Residual stream. */
  x: Vec,
  /** Scratch: queries per head after qkv, MLP hidden after fc/gelu. */
  h: readonly unknown[],
  /** Attention output, filled one head per attn tick. */
  o: Vec,
  kv: readonly Cache[],
];

/** Index of each State field. */
type I = { phase: 0; tokens: 1; pos: 2; layer: 3; head: 4; left: 5; x: 6; h: 7; o: 8; kv: 9 };

// ---------------------------------------------------------------- helpers

/** S with the fields in P replaced. */
type Set<S extends State, P> = [
  P extends { phase: infer V } ? V : S[0],
  P extends { tokens: infer V } ? V : S[1],
  P extends { pos: infer V } ? V : S[2],
  P extends { layer: infer V } ? V : S[3],
  P extends { head: infer V } ? V : S[4],
  P extends { left: infer V } ? V : S[5],
  P extends { x: infer V } ? V : S[6],
  P extends { h: infer V } ? V : S[7],
  P extends { o: infer V } ? V : S[8],
  P extends { kv: infer V } ? V : S[9],
];
type Tuple<N extends number, T extends unknown[] = []> = T["length"] extends N
  ? T
  : Tuple<N, [...T, 0]>;

// Tail-recursive (accumulator) so prompts and outputs can exceed the ~50-level
// limit on non-tail recursion.
type Encode<
  M extends Model,
  S extends string,
  Acc extends unknown[] = [],
> = S extends `${infer C}${infer R}` ? Encode<M, R, [...Acc, M["charToId"][C]]> : Acc;
type Decode<M extends Model, T, Acc extends string = ""> = T extends readonly [
  infer I extends number,
  ...infer R,
]
  ? Decode<M, R, `${Acc}${M["vocab"][I]}`>
  : Acc;

// Mapped types only preserve tuples when they map over a bare type parameter
// (`{ [K in keyof T]: ... }`), so each tuple map gets its own alias.

/** Per-head projection: one MatVec per head's weight block. */
type PerHead<Ws, Bs, A> = { [H in keyof Ws]: MatVec<Ws[H], A, Bs[H & keyof Bs]> };
/** Append this position's K and V to cache I. */
type Append<KV, I extends string, K, V> = {
  [J in keyof KV]: J extends I
    ? KV[J] extends Cache
      ? [[...KV[J][0], K], [...KV[J][1], V]]
      : never
    : KV[J];
};
/** Pick head H out of every cached position. */
type Column<C, H> = { [J in keyof C]: C[J][H & keyof C[J]] };
type GeluAll<X> = { [K in keyof X]: Gelu<X[K]> };
type EmptyCache<Ls> = { [L in keyof Ls]: [[], []] };

type LayerOf<M extends Model, S extends State> = M["layers"][S[I["layer"]]["length"]];
type CacheOf<S extends State> = S[I["kv"]][S[I["layer"]]["length"]];

// ---------------------------------------------------------------- ticks

// Each tick computes its results in `extends infer` checks before building the
// next state. Object type members are resolved lazily, so a value written
// straight into the state object would only be computed when a later statement
// reads it. Forcing them here keeps each tick's work inside its own statement.

type Embed<M extends Model, S extends State> =
  AddVec<
    M["wte"][S[I["tokens"]][S[I["pos"]]["length"]]],
    M["wpe"][S[I["pos"]]["length"]]
  > extends infer X extends Vec
    ? Set<S, { phase: "qkv"; layer: []; x: X }>
    : never;

/** Project to per-head Q, K, V and append K, V to this layer's cache. */
type Qkv<M extends Model, S extends State, L extends LayerWeights = LayerOf<M, S>> =
  LayerNorm<S[I["x"]], L["ln1g"], L["ln1b"]> extends infer A extends Vec
    ? PerHead<L["qW"], L["qB"], A> extends infer Q extends readonly Vec[]
      ? PerHead<L["kW"], L["kB"], A> extends infer K extends readonly Vec[]
        ? PerHead<L["vW"], L["vB"], A> extends infer V extends readonly Vec[]
          ? Append<S[I["kv"]], `${S[I["layer"]]["length"]}`, K, V> extends infer KV extends
              readonly Cache[]
            ? Set<S, { phase: "attn"; h: Q; kv: KV; head: []; o: [] }>
            : never
          : never
        : never
      : never
    : never;

/** Causal attention of one head over all cached positions. */
type AttnHead<Q, K, V> =
  Softmax<Scores<Q, K>> extends infer P extends Vec
    ? { [D in keyof Q]: Dot<P, Column<V, D>> }
    : never;
type Scores<Q, K> = { [J in keyof K]: Dot<Q, K[J]> };

/**
 * One attention head per tick: at long contexts all heads together exceed the
 * per-statement instantiation budget.
 */
type Attn<
  S extends State,
  C extends Cache = CacheOf<S>,
  H extends string = `${S[I["head"]]["length"]}`,
> =
  AttnHead<
    S[I["h"]][S[I["head"]]["length"]],
    Column<C[0], H>,
    Column<C[1], H>
  > extends infer O extends Vec
    ? [...S[I["head"]], 0]["length"] extends S[I["h"]]["length"]
      ? Set<S, { phase: "oproj"; o: [...S[I["o"]], ...O]; head: [] }>
      : Set<S, { o: [...S[I["o"]], ...O]; head: [...S[I["head"]], 0] }>
    : never;

/** Attention output projection + residual. */
type OProj<M extends Model, S extends State, L extends LayerWeights = LayerOf<M, S>> =
  AddVec<S[I["x"]], MatVec<L["projW"], S[I["o"]], L["projB"]>> extends infer X extends Vec
    ? Set<S, { phase: "fc"; x: X; o: [] }>
    : never;

type Fc<M extends Model, S extends State, L extends LayerWeights = LayerOf<M, S>> =
  MatVec<L["fcW"], LayerNorm<S[I["x"]], L["ln2g"], L["ln2b"]>, L["fcB"]> extends infer H extends Vec
    ? Set<S, { phase: "gelu"; h: H }>
    : never;

type GeluTick<S extends State> =
  GeluAll<S[I["h"]]> extends infer H extends Vec ? Set<S, { phase: "proj"; h: H }> : never;

/** MLP output + residual, then move to the next layer, the head, or the next position. */
type Proj<M extends Model, S extends State, L extends LayerWeights = LayerOf<M, S>> =
  AddVec<S[I["x"]], MatVec<L["mlpW"], S[I["h"]], L["mlpB"]>> extends infer X extends Vec
    ? [...S[I["layer"]], 0]["length"] extends M["nLayer"]
      ? [...S[I["pos"]], 0]["length"] extends S[I["tokens"]]["length"]
        ? Set<S, { phase: "head"; x: X }>
        : Set<S, { phase: "embed"; x: X; pos: [...S[I["pos"]], 0] }>
      : Set<S, { phase: "qkv"; x: X; layer: [...S[I["layer"]], 0] }>
    : never;

/** Final LayerNorm, tied LM head, greedy argmax; append the token. */
type Head<M extends Model, S extends State> =
  Argmax<MatVec<M["wte"], LayerNorm<S[I["x"]], M["lnfG"], M["lnfB"]>>> extends infer T extends
    number
    ? S[I["left"]] extends readonly [unknown, ...infer Left]
      ? Left extends []
        ? Set<S, { phase: "done"; tokens: [...S[I["tokens"]], T]; left: Left }>
        : [...S[I["pos"]], 0]["length"] extends M["blockSize"]
          ? Set<S, { phase: "done"; tokens: [...S[I["tokens"]], T]; left: Left }>
          : Set<
              S,
              {
                phase: "embed";
                tokens: [...S[I["tokens"]], T];
                left: Left;
                pos: [...S[I["pos"]], 0];
              }
            >
      : never
    : never;

// ---------------------------------------------------------------- public API

/** Initial state: tokenize the prompt and set up an empty KV cache. */
export type Init<M extends Model, Prompt extends string, N extends number> = [
  phase: N extends 0 ? "done" : "embed",
  tokens: Encode<M, Prompt>,
  pos: [],
  layer: [],
  head: [],
  left: Tuple<N>,
  x: [],
  h: [],
  o: [],
  kv: EmptyCache<M["layers"]>,
];

/** Run one tick. */
export type Step<M extends Model, S extends State> = Resolve<Tick<M, S>>;

/**
 * Read every field of the new state so it is computed now. Without this a
 * field nobody touched would stay a reference to the previous state's field,
 * and after ~100 ticks the chain trips the instantiation depth limit.
 */
type Resolve<S> = S extends readonly [
  infer A,
  infer B,
  infer C,
  infer D,
  infer E,
  infer F,
  infer G,
  infer H,
  infer J,
  infer K,
]
  ? [A, B, C, D, E, F, G, H, J, K]
  : never;

type Tick<M extends Model, S extends State> = S[I["phase"]] extends "embed"
  ? Embed<M, S>
  : S[I["phase"]] extends "qkv"
    ? Qkv<M, S>
    : S[I["phase"]] extends "attn"
      ? Attn<S>
      : S[I["phase"]] extends "oproj"
        ? OProj<M, S>
        : S[I["phase"]] extends "fc"
          ? Fc<M, S>
          : S[I["phase"]] extends "gelu"
            ? GeluTick<S>
            : S[I["phase"]] extends "proj"
              ? Proj<M, S>
              : S[I["phase"]] extends "head"
                ? Head<M, S>
                : S;

/** The text so far (prompt + generated). */
export type Output<M extends Model, S extends State> = Decode<M, S[I["tokens"]]>;

/** Number of ticks needed to generate N tokens from a P-character prompt. */
export const ticksFor = (promptLength: number, n: number, nLayer: number, nHead: number): number =>
  n === 0 ? 0 : (promptLength + n - 1) * (1 + nLayer * (nHead + 5)) + n;
