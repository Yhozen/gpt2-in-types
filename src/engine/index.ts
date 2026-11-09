/**
 * The integer GPT-2 (src/reference/int.ts) as one type:
 *
 *   type Text = Generate<"ROMEO:", 16>;   // hover it
 *
 * Values are signed unary strings ("111" = 3, "-11" = -2, "" = 0; see
 * src/num/ustr.ts). Additions are string concatenation, comparisons are prefix
 * matches, and everything non-linear is a table lookup or a binary search
 * over a threshold table, so no step of the model multiplies.
 *
 * This engine is written for the compiled program in src/model/program.ts
 * (head size 8, 4 heads); see scripts/compile.ts.
 */
import type { AddS, NegS, SubU, ToS } from "../num/ustr.ts";
import type * as M from "../model/program.ts";

type Strs = readonly string[];
type Nums = readonly number[];

// ---------------------------------------------------------------- helpers

export type PosPart<S> = S extends `-${string}` ? "" : S;
export type NegPart<S> = S extends `-${infer A}` ? A : "";
/** Signed vector -> [positive parts, negative magnitudes] (the matrices' inputs). */
/** @internal */
export type Split<X> = [{ [K in keyof X]: PosPart<X[K]> }, { [K in keyof X]: NegPart<X[K]> }];

export type Negate<N> = N extends 0
  ? 0
  : `-${N & number}` extends `${infer V extends number}`
    ? V
    : never;
export type Dbl<S extends string> = `${S}${S}`;

/**
 * Evaluate a compiled threshold table: a binary decision tree of
 * [threshold, ifLess, ifGreaterOrEqual] nodes with outputs at the leaves (see
 * scripts/compile.ts). Each level is one prefix match on a unary string.
 */
export type Walk<A extends string, N> = N extends readonly [
  infer T extends string,
  infer Lo,
  infer Hi,
]
  ? A extends `${T}${string}`
    ? Walk<A, Hi>
    : Walk<A, Lo>
  : N;

/** Signed requantization through a tree with unary leaves. */
export type RQU<A, T> = A extends `-${infer Mag}`
  ? NegS<Walk<Mag, T> & string>
  : Walk<A & string, T>;
/** Same with number leaves. */
export type RQN<A, T> = A extends `-${infer Mag}` ? Negate<Walk<Mag, T>> : Walk<A & string, T>;
export type RQAllU<A, T> = { [K in keyof A]: RQU<A[K], T> };
export type RQAllN<A, T> = { [K in keyof A]: RQN<A[K], T> };

/** Residual stream values stay in [-255, 255]. */
type R255 = M.BitsU8["11111111"];
export type Clamp<S> = S extends `-${R255}${string}`
  ? `-${R255}`
  : S extends `${R255}${string}`
    ? R255
    : S;
export type AddClamped<X, D> = {
  [K in keyof X]: Clamp<AddS<X[K] & string, D[K & keyof D] & string>>;
};

// ---------------------------------------------------------------- LayerNorm

/** @internal */
export type SumS<X, P extends string = "", N extends string = ""> = X extends readonly [
  infer H,
  ...infer R,
]
  ? H extends `-${infer A}`
    ? SumS<R, P, `${N}${A}`>
    : SumS<R, `${P}${H & string}`, N>
  : SubU<P, N>;
/** Sum of squares as two unary limbs [HI, LO] (c^2 = 512 hi + lo). */
export type SumSq<C, HI extends string = "", LO extends string = ""> = C extends readonly [
  infer H,
  ...infer R,
]
  ? M.Squares[(H extends `-${infer A}` ? A : H) & keyof M.Squares] extends [
      infer h extends string,
      infer l extends string,
    ]
    ? SumSq<R, `${HI}${h}`, `${LO}${l}`>
    : never
  : [HI, LO];
/** Normalize LO < 512 by carrying into HI. */
export type Normalize<S> = S extends [infer HI extends string, infer LO extends string]
  ? Walk<LO, M.Carry> extends [infer c extends string, infer c512 extends string]
    ? LO extends `${c512}${infer Rest}`
      ? [`${HI}${c}`, Rest]
      : never
    : never
  : never;
/** Walk the sigma tree: compare SS = 512 HI + LO against each node's threshold. */
export type WalkSigma<HI extends string, LO extends string, N> = N extends readonly [
  infer H1 extends string,
  infer H0 extends string,
  infer L0 extends string,
  infer Lo,
  infer Hi,
]
  ? HI extends `${H1}${string}`
    ? WalkSigma<HI, LO, Hi>
    : HI extends `${H0}${string}`
      ? LO extends `${L0}${string}`
        ? WalkSigma<HI, LO, Hi>
        : WalkSigma<HI, LO, Lo>
      : WalkSigma<HI, LO, Lo>
  : N;
export type Rev<S extends string, Acc extends string = ""> = S extends `${infer H}${infer R}`
  ? Rev<R, `${H}${Acc}`>
  : Acc;
export type Zeros<S extends string, Acc extends string = ""> = S extends `${string}${infer R}`
  ? Zeros<R, `${Acc}0`>
  : Acc;
/** The sum of squares truncated to 3 significant digits (the sqrt table's key). */
/** @internal */
export type SqrtKey<SS extends string> =
  Rev<SS> extends `${infer A}${infer B}${infer C}${infer Rest}`
    ? `${A}${B}${C}${Zeros<Rest>}`
    : Rev<SS>;
/** [128 sigma, 64 sigma, ..., sigma] for the long division. */
export type Divisors<S1 extends string> =
  Dbl<S1> extends infer S2 extends string
    ? Dbl<S2> extends infer S4 extends string
      ? Dbl<S4> extends infer S8 extends string
        ? Dbl<S8> extends infer S16 extends string
          ? Dbl<S16> extends infer S32 extends string
            ? Dbl<S32> extends infer S64 extends string
              ? [Dbl<S64>, S64, S32, S16, S8, S4, S2, S1]
              : never
            : never
          : never
        : never
      : never
    : never;
/** Binary long division: bits of floor(A / D[last]), up to 8 bits (saturates at 255). */
export type LongDiv<A extends string, D, Bits extends string = ""> = D extends readonly [
  infer H extends string,
  ...infer R,
]
  ? A extends `${H}${infer Rem}`
    ? LongDiv<Rem, R, `${Bits}1`>
    : LongDiv<A, R, `${Bits}0`>
  : Bits;
/** |z| = min(127, (floor(64|c| / sigma) + 1) >> 1). */
export type ZMag<C extends string, D> = M.BitsZ[LongDiv<Dbl<Dbl<Dbl<Dbl<Dbl<Dbl<C>>>>>>, D> &
  keyof M.BitsZ];
export type ZAll<C, D> = {
  [K in keyof C]: C[K] extends `-${infer A}` ? NegS<ZMag<A, D>> : ZMag<C[K] & string, D>;
};
export type Center<X, NegMu extends string> = { [K in keyof X]: AddS<X[K] & string, NegMu> };

/** LayerNorm (no affine): signed unary in, z in units of sigma / 32 out. */
/** @internal */
export type LayerNorm<X> =
  RQU<SumS<X>, M.Mean> extends infer Mu extends string
    ? Center<X, NegS<Mu>> extends infer C extends Strs
      ? Normalize<SumSq<C>> extends [infer HI extends string, infer LO extends string]
        ? Divisors<WalkSigma<HI, LO, M.Sigma> & string> extends infer D extends Strs
          ? ZAll<C, D>
          : never
        : never
      : never
    : never;

// ---------------------------------------------------------------- attention

/** Scores of one query against every cached key: concat of q_i * k_i parts. */
export type Score<QP, QN, K, P extends string = "", N extends string = ""> = K extends readonly [
  infer k,
  ...infer KR,
]
  ? QP extends readonly [infer qp, ...infer QPR]
    ? QN extends readonly [infer qn, ...infer QNR]
      ? Score<
          QPR,
          QNR,
          KR,
          `${P}${qp[`${k & number}` & keyof qp] & string}`,
          `${N}${qn[`${k & number}` & keyof qn] & string}`
        >
      : never
    : never
  : SubU<P, N>;
export type Scores<QP, QN, KH> = { [J in keyof KH]: Score<QP, QN, KH[J]> };
export type RowsP<Q> = { [I in keyof Q]: M.ProdP[`${Q[I] & number}` & keyof M.ProdP] };
export type RowsN<Q> = { [I in keyof Q]: M.ProdN[`${Q[I] & number}` & keyof M.ProdN] };

/** Signed a > b on unary strings. */
export type Gt<A, B> = A extends `-${infer a}`
  ? B extends `-${infer b}`
    ? b extends `${a}1${string}`
      ? true
      : false
    : false
  : B extends `-${string}`
    ? A extends ""
      ? B extends "-"
        ? false
        : true
      : true
    : A extends `${B & string}1${string}`
      ? true
      : false;
export type MaxS<
  X,
  Best = X extends readonly [infer F, ...unknown[]] ? F : "",
> = X extends readonly [infer H, ...infer R] ? MaxS<R, Gt<H, Best> extends true ? H : Best> : Best;
export type Cat<X, Acc extends string = ""> = X extends readonly [infer H, ...infer R]
  ? Cat<R, `${Acc}${H & string}`>
  : Acc;

export type ExpAll<D, L> = { [J in keyof D]: Walk<D[J] & string, L["exp" & keyof L]> };
export type ProbAll<D, Lse extends string, L> = {
  [J in keyof D]: Walk<`${D[J] & string}${Lse}`, L["prob" & keyof L]>;
};
export type Weighted<Pr, V, Dim> = {
  [J in keyof Pr]: PV<
    M.ProdPV[`${Pr[J] & number}` & keyof M.ProdPV],
    V[J & keyof V][Dim & keyof V[J & keyof V]]
  >;
};
type PV<Row, Val> = Row[`${Val & number}` & keyof Row];

/** One head: integer softmax over the cache, then sum_j p_j v_j, requantized. */
/** @internal */
export type AttendHead<Q, KH, VH, L> =
  Scores<RowsP<Q>, RowsN<Q>, KH> extends infer S extends Strs
    ? MaxS<S> extends infer Mx extends string
      ? { [J in keyof S]: AddS<Mx, NegS<S[J]>> } extends infer D extends Strs
        ? Walk<Cat<ExpAll<D, L>>, L["lse" & keyof L]> extends infer Lse extends string
          ? ProbAll<D, Lse, L> extends infer Pr extends Nums
            ? { [I in keyof Q]: RQU<SumS<Weighted<Pr, VH, I>>, L["rqO" & keyof L]> }
            : never
          : never
        : never
      : never
    : never;

/** Split a 32-vector into 4 heads of 8. */
export type Heads<X> = X extends readonly [
  infer A0,
  infer A1,
  infer A2,
  infer A3,
  infer A4,
  infer A5,
  infer A6,
  infer A7,
  infer B0,
  infer B1,
  infer B2,
  infer B3,
  infer B4,
  infer B5,
  infer B6,
  infer B7,
  infer C0,
  infer C1,
  infer C2,
  infer C3,
  infer C4,
  infer C5,
  infer C6,
  infer C7,
  infer D0,
  infer D1,
  infer D2,
  infer D3,
  infer D4,
  infer D5,
  infer D6,
  infer D7,
]
  ? [
      [A0, A1, A2, A3, A4, A5, A6, A7],
      [B0, B1, B2, B3, B4, B5, B6, B7],
      [C0, C1, C2, C3, C4, C5, C6, C7],
      [D0, D1, D2, D3, D4, D5, D6, D7],
    ]
  : never;
export type Flat4<H> = H extends readonly [
  infer A extends Strs,
  infer B extends Strs,
  infer C extends Strs,
  infer D extends Strs,
]
  ? [...A, ...B, ...C, ...D]
  : never;
/** For head h: the h-th slice of every cached position. */
export type Column<C, H> = { [J in keyof C]: C[J][H & keyof C[J]] };
export type AllHeads<QH, KC, VC, L> = {
  [H in keyof QH]: AttendHead<QH[H], Column<KC, H>, Column<VC, H>, L>;
};

// ---------------------------------------------------------------- block

export type Lay<L extends number> = M.Layers[L];

/** @internal */
/** x + OFF as unsigned unary (|x| <= OFF): the matrices' input encoding. */
export type Offset<X, OFF extends string> = { [K in keyof X]: AddS<X[K] & string, OFF> };
type Off127 = ToS<127>;
type Off63 = ToS<63>;
type OffsetOf<N extends string> = N extends "proj" | "mlp" ? Off63 : Off127;
export type Matvec<N extends string, L extends number, X> =
  Offset<X, OffsetOf<N>> extends infer XO extends Strs ? M.Mat<N, L, XO> : never;

/** Replace layer L's cache with [keys, values] + this position. */
export type AppendKV<KV, L extends number, K, V> = {
  [I in keyof KV]: I extends `${L}`
    ? KV[I] extends readonly [
        infer Ks extends readonly unknown[],
        infer Vs extends readonly unknown[],
      ]
      ? [[...Ks, K], [...Vs, V]]
      : never
    : KV[I];
};

/** One transformer block at one position. Returns [x, kv]. */
/** @internal */
export type Block<L extends number, X, KV extends readonly unknown[]> =
  LayerNorm<X> extends infer Z extends Strs
    ? RQAllN<Matvec<"q", L, Z>, Lay<L>["rqQ"]> extends infer Q extends Nums
      ? RQAllN<Matvec<"k", L, Z>, Lay<L>["rqK"]> extends infer K extends Nums
        ? RQAllN<Matvec<"v", L, Z>, Lay<L>["rqV"]> extends infer V extends Nums
          ? AppendKV<KV, L, Heads<K>, Heads<V>> extends infer KV2 extends readonly unknown[]
            ? KV2[L] extends readonly [infer KC, infer VC]
              ? Flat4<AllHeads<Heads<Q>, KC, VC, Lay<L>>> extends infer O extends Strs
                ? AddClamped<
                    X,
                    RQAllU<Matvec<"proj", L, O>, Lay<L>["rqProj"]>
                  > extends infer X2 extends Strs
                  ? RQAllN<Matvec<"fc", L, LayerNorm<X2>>, Lay<L>["rqFc"]> extends infer G extends
                      Nums
                    ? {
                        [I in keyof G]: Lay<L>["gelu"][`${G[I] & number}` & keyof Lay<L>["gelu"]];
                      } extends infer H extends Strs
                      ? AddClamped<
                          X2,
                          RQAllU<Matvec<"mlp", L, H>, Lay<L>["rqMlp"]>
                        > extends infer X3 extends Strs
                        ? [X3, KV2]
                        : never
                      : never
                    : never
                  : never
                : never
              : never
            : never
          : never
        : never
      : never
    : never;

/** @internal */
export type Layers<
  X,
  KV extends readonly unknown[],
  L extends unknown[] = [],
> = L["length"] extends M.Config["nLayer"]
  ? [X, KV]
  : Block<L["length"], X, KV> extends [infer X2 extends Strs, infer KV2 extends readonly unknown[]]
    ? Layers<X2, KV2, [...L, 0]>
    : never;

/** @internal */
export type Embed<T extends number, P extends number> = EmbedRow<M.Etok[T], M.Epos[P]>;
export type EmbedRow<E, P> = { [I in keyof E]: Clamp<AddS<ToS<E[I]>, ToS<P[I & keyof P]>>> };

/** Index of the first maximum. */
export type Argmax<
  X,
  I extends unknown[] = [],
  Best extends number = 0,
  B = X extends readonly [infer F, ...unknown[]] ? F : "",
> = X extends readonly [infer H, ...infer R]
  ? Gt<H, B> extends true
    ? Argmax<R, [...I, 0], I["length"], H>
    : Argmax<R, [...I, 0], Best, B>
  : Best;
/** @internal */
export type NextToken<X, Pos extends number, Mode> =
  Offset<LayerNorm<X>, Off127> extends infer XO extends Strs
    ? Argmax<Mode extends "greedy" ? M.Head<XO> : AddNoise<M.Head<XO>, M.Noise[Pos]>>
    : never;
type AddNoise<A, N> = { [K in keyof A]: AddS<A[K] & string, N[K & keyof N] & string> };

// ---------------------------------------------------------------- generation

/** @internal */
export type Encode<S extends string, Acc extends number[] = []> = S extends `${infer C}${infer R}`
  ? Encode<R, [...Acc, M.CharToId[C & keyof M.CharToId]]>
  : Acc;
export type Decode<T, Acc extends string = ""> = T extends readonly [infer I, ...infer R]
  ? Decode<R, `${Acc}${M.Vocab[I & keyof M.Vocab] & string}`>
  : Acc;
export type Tuple<N extends number, T extends unknown[] = []> = T["length"] extends N
  ? T
  : Tuple<N, [...T, 0]>;
/** @internal */
export type EmptyKV<L extends unknown[] = []> = L["length"] extends M.Config["nLayer"]
  ? []
  : [[[], []], ...EmptyKV<[...L, 0]>];

/**
 * Generation state: [tokens, position counter, tokens still to generate, KV cache].
 * Steps runs positions until N more tokens exist (or the context is full), so a
 * long generation can be split across statements: the CLI allows ~5M type
 * instantiations per statement; an editor hover has no such cap.
 */
export type State = readonly [number[], unknown[], readonly unknown[]];
export type Start<Prompt extends string> = [Encode<Prompt>, [], EmptyKV];

/** Run positions until N more tokens are generated. */
export type Steps<S, N extends number, Mode = "sample"> = S extends readonly [
  infer Tokens extends number[],
  infer Pos extends unknown[],
  infer KV extends readonly unknown[],
]
  ? Run<Tokens, Pos, Tuple<N>, KV, Mode>
  : never;

/** The text of a state (prompt + generated so far). */
export type Output<S> = S extends readonly [infer Tokens, ...unknown[]] ? Decode<Tokens> : never;

/** Process position Pos; append the next token when the prompt is consumed. */
export type Run<
  Tokens extends number[],
  Pos extends unknown[],
  Left extends unknown[],
  KV extends readonly unknown[],
  Mode,
> = Pos["length"] extends M.Config["blockSize"]
  ? [Tokens, Pos, KV]
  : [...Pos, 0]["length"] extends Tokens["length"]
    ? Left extends [unknown, ...infer Left2 extends unknown[]]
      ? Layers<Embed<Tokens[Pos["length"]], Pos["length"]>, KV> extends [
          infer X,
          infer KV2 extends readonly unknown[],
        ]
        ? Run<[...Tokens, NextToken<X, Pos["length"], Mode>], [...Pos, 0], Left2, KV2, Mode>
        : never
      : [Tokens, Pos, KV]
    : // A prompt position: only the KV cache matters.
      Layers<Embed<Tokens[Pos["length"]], Pos["length"]>, KV> extends [
          unknown,
          infer KV2 extends readonly unknown[],
        ]
      ? Run<Tokens, [...Pos, 0], Left, KV2, Mode>
      : never;

/** The prompt plus N generated characters (seeded sampling; "greedy" for argmax). */
export type Generate<
  Prompt extends string,
  N extends number,
  Mode extends "sample" | "greedy" = "sample",
> = Output<Steps<Start<Prompt>, N, Mode>>;

/** Final residual stream of every position of a token sequence (for tests). */
export type Trace<
  Tokens extends number[],
  Pos extends unknown[] = [],
  KV extends readonly unknown[] = EmptyKV,
  Out extends unknown[] = [],
> = Pos["length"] extends Tokens["length"]
  ? Out
  : Layers<Embed<Tokens[Pos["length"]], Pos["length"]>, KV> extends [
        infer X,
        infer KV2 extends readonly unknown[],
      ]
    ? Trace<Tokens, [...Pos, 0], KV2, [...Out, X]>
    : never;
