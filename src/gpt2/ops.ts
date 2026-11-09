/**
 * Neural-network operations on fixed-point vectors (Nums at scale 10^3).
 * Each one mirrors the function of the same name in src/reference/fixed.ts.
 */
import type {
  Abs,
  Add,
  AddU,
  Cmp,
  DivRound,
  Dot,
  ISqrt,
  IsNeg,
  Mul,
  Neg,
  RoundShift,
  Sub,
  Strip,
  Sum,
  SumSq,
} from "../num/index.ts";
import type { ExpD1, ExpD2, ExpD3, ExpInt } from "./exp-table.ts";

export type Vec = readonly string[];
export type Mat = readonly Vec[];

/** Little-endian Num constants. */
type E9 = "0000000001"; // 10^9
type R3 = "005"; // half of 10^3, for RoundShift by 3 digits
type R6 = "000005"; // half of 10^6

// Vector ops take unconstrained parameters: callers pass mapped and indexed
// types, and intersecting those with `Vec` to satisfy a constraint would change
// what the mapped types below produce.

export type AddVec<A, B> = { [K in keyof A]: Add<A[K], B[K & keyof B]> };

/** W x + b, with W stored one row per output. */
export type MatVec<W, X, B = undefined> = {
  [K in keyof W]: B extends undefined ? Dot<X, W[K]> : Add<Dot<X, W[K]>, B[K & keyof B]>;
};

// ---------------------------------------------------------------- exp / gelu

/** e^X for X <= 0, from the digits of -X: e^-(i.abc) = e^-i * e^-.a * e^-.0b * e^-.00c (tables at 10^6). */
export type Exp<X> = ExpDigits<`${Abs<X>}000`>;
type ExpDigits<M extends string> = M extends `${infer D3}${infer D2}${infer D1}${infer I}`
  ? Strip<I> extends infer N extends keyof ExpInt
    ? RoundShift<
        RoundShift<
          Mul<
            RoundShift<
              Mul<RoundShift<Mul<ExpInt[N], ExpD1[D1 & keyof ExpD1]>, R6>, ExpD2[D2 & keyof ExpD2]>,
              R6
            >,
            ExpD3[D3 & keyof ExpD3]
          >,
          R6
        >,
        R3
      >
    : "0"
  : never;

/** GELU (tanh form) as x * sigmoid(2 sqrt(2/pi) (x + 0.044715 x^3)). */
export type Gelu<X> = GeluZ<X, RoundShift<Mul<RoundShift<Mul<X, X>, R3>, X>, R3>>;
// Split into steps so no single declaration nests too deeply for the checker.
type GeluZ<X, X3> = GeluS<
  X,
  RoundShift<Mul<Add<X, RoundShift<Mul<X3, "51744">, R6>>, "9675951">, R6>
>;
type GeluS<X, Z> = GeluOut<X, Z, Sigmoid<Exp<Neg<Abs<Z>>>>>;
type GeluOut<X, Z, S> = RoundShift<Mul<X, IsNeg<Z> extends true ? Sub<"0001", S> : S>, R3>;

/** 1 / (1 + E) at scale 10^3, i.e. 10^6 / (10^3 + E). */
type Sigmoid<E> = E extends string ? DivRound<"0000001", AddU<"0001", E>> : never;

// ---------------------------------------------------------------- layer norm

export type LayerNorm<X, G, B> =
  DivRound<Sum<X>, Len<X>> extends infer Mean extends string
    ? { [K in keyof X]: Sub<X[K], Mean> } extends infer C extends Vec
      ? DivRound<E9, ISqrt<AddU<DivRound<SumSq<C>, Len<X>>, "01">>> extends infer Inv extends string
        ? {
            [K in keyof C]: Add<
              RoundShift<Mul<RoundShift<Mul<C[K], Inv>, R6>, G[K & keyof G]>, R3>,
              B[K & keyof B]
            >;
          }
        : never
      : never
    : never;

/** Length of a tuple as a little-endian Num. */
type Len<X> = X extends Vec ? FromLen<`${X["length"]}`> : never;
type FromLen<S extends string, Acc extends string = ""> = S extends `${infer H}${infer R}`
  ? FromLen<R, `${H}${Acc}`>
  : Acc;

// ---------------------------------------------------------------- softmax / argmax

type First<X> = X extends readonly [infer F, ...unknown[]] ? F : never;

export type MaxOf<X, M = First<X>> = X extends readonly [
  infer x extends string,
  ...infer R extends Vec,
]
  ? MaxOf<R, Cmp<x, M> extends ">" ? x : M>
  : M;

export type Softmax<S> =
  MaxOf<S> extends infer M extends string
    ? { [K in keyof S]: Exp<Sub<S[K], M>> } extends infer E extends Vec
      ? DivRound<E9, Sum<E>> extends infer Inv extends string
        ? { [K in keyof E]: RoundShift<Mul<E[K], Inv>, R6> }
        : never
      : never
    : never;

/** Index of the first maximum. */
export type Argmax<
  X,
  I extends unknown[] = [],
  Best extends number = 0,
  M = First<X>,
> = X extends readonly [infer x extends string, ...infer R extends Vec]
  ? Cmp<x, M> extends ">"
    ? Argmax<R, [...I, 0], I["length"], x>
    : Argmax<R, [...I, 0], Best, M>
  : Best;
