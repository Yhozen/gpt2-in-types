/**
 * Type-level signed integer arithmetic on decimal digit strings.
 *
 * A `Num` is a string of decimal digits stored least-significant digit first,
 * with an optional leading "-" for negatives. It has no high-order zeros, and
 * zero is "0" (never "-0").
 *
 *   1234  -> "4321"
 *   -50   -> "-05"
 *   0     -> "0"
 *
 * Little-endian order means addition and multiplication walk both operands from
 * the front of the string, and multiplying by 10^k is prepending k zeros.
 * Digit-level work is done by lookups in the tables in ./tables.ts.
 */
import type { AddTable, CmpTable, MulTable, SubTable } from "./tables.ts";

export type Digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";
type Carry = "0" | "1";

// ---------------------------------------------------------------- unsigned

/** |A| + |B| */
export type AddU<
  A extends string,
  B extends string,
  C extends Carry = "0",
  Acc extends string = "",
> = A extends `${infer a extends Digit}${infer AR}`
  ? B extends `${infer b extends Digit}${infer BR}`
    ? AddTable[`${a}${b}${C}`] extends `${infer D}${infer C2 extends Carry}`
      ? AddU<AR, BR, C2, `${Acc}${D}`>
      : never
    : C extends "0"
      ? `${Acc}${A}`
      : AddTable[`${a}0${C}`] extends `${infer D}${infer C2 extends Carry}`
        ? AddU<AR, "", C2, `${Acc}${D}`>
        : never
  : B extends ""
    ? C extends "0"
      ? Acc
      : `${Acc}1`
    : AddU<B, "", C, Acc>;

/** Drop high-order zeros; "" becomes "0". */
export type Strip<S extends string> = S extends `${infer R}0` ? Strip<R> : S extends "" ? "0" : S;

/** |A| - |B|, assuming |A| >= |B|. */
export type SubU<A extends string, B extends string> = Strip<SubRaw<A, B>>;
type SubRaw<
  A extends string,
  B extends string,
  C extends Carry = "0",
  Acc extends string = "",
> = A extends `${infer a extends Digit}${infer AR}`
  ? B extends `${infer b extends Digit}${infer BR}`
    ? SubTable[`${a}${b}${C}`] extends `${infer D}${infer C2 extends Carry}`
      ? SubRaw<AR, BR, C2, `${Acc}${D}`>
      : never
    : C extends "0"
      ? `${Acc}${A}`
      : SubTable[`${a}0${C}`] extends `${infer D}${infer C2 extends Carry}`
        ? SubRaw<AR, "", C2, `${Acc}${D}`>
        : never
  : Acc;

/** Compare magnitudes: "<", "=" or ">". */
export type CmpU<
  A extends string,
  B extends string,
  R extends string = "=",
> = A extends `${infer a extends Digit}${infer AR}`
  ? B extends `${infer b extends Digit}${infer BR}`
    ? CmpU<AR, BR, CmpTable[`${a}${b}`] extends "=" ? R : CmpTable[`${a}${b}`]>
    : ">"
  : B extends ""
    ? R
    : "<";

/** |A| * d for a single digit d. */
type MulDigit<
  A extends string,
  D extends Digit,
  C extends Digit = "0",
  Acc extends string = "",
> = A extends `${infer a extends Digit}${infer AR}`
  ? MulTable[`${a}${D}${C}`] extends `${infer R}${infer C2 extends Digit}`
    ? MulDigit<AR, D, C2, `${Acc}${R}`>
    : never
  : C extends "0"
    ? Acc
    : `${Acc}${C}`;

/** |A| * |B| (schoolbook). */
export type MulU<A extends string, B extends string> = A extends "0"
  ? "0"
  : B extends "0"
    ? "0"
    : MulLoop<A, B, "", "0">;
type MulLoop<
  A extends string,
  B extends string,
  Shift extends string,
  Acc extends string,
> = B extends `${infer b extends Digit}${infer BR}`
  ? MulLoop<
      A,
      BR,
      `${Shift}0`,
      b extends "0"
        ? Acc
        : b extends "1"
          ? AddU<Acc, `${Shift}${A}`>
          : AddU<Acc, `${Shift}${MulDigit<A, b>}`>
    >
  : Acc;

// ---------------------------------------------------------------- signed

// The signed operations below take unconstrained parameters on purpose: when a
// generic declaration passes a deferred conditional type into a parameter
// constrained to `string`, the checker expands that type's constraint
// recursively and reports TS2589. Unconstrained parameters avoid the check.

export type Neg<A> = A extends "0" ? "0" : A extends `-${infer M}` ? M : `-${A & string}`;
export type Abs<A> = A extends `-${infer M}` ? M : A & string;
export type IsNeg<A> = A extends `-${string}` ? true : false;

/** Magnitude X minus magnitude Y, as a signed Num. */
export type SubM<X extends string, Y extends string> =
  CmpU<X, Y> extends "<" ? `-${SubU<Y, X>}` : SubU<X, Y>;

export type Add<A, B> = A extends `-${infer a}`
  ? B extends `-${infer b}`
    ? `-${AddU<a, b>}`
    : SubM<B & string, a>
  : B extends `-${infer b}`
    ? SubM<A & string, b>
    : AddU<A & string, B & string>;

export type Sub<A, B> = Add<A, Neg<B>>;

export type Mul<A, B> = A extends `-${infer a}`
  ? B extends `-${infer b}`
    ? MulU<a, b>
    : Neg<MulU<a, B & string>>
  : B extends `-${infer b}`
    ? Neg<MulU<A & string, b>>
    : MulU<A & string, B & string>;

/** Signed comparison. */
export type Cmp<A, B> = A extends `-${infer a}`
  ? B extends `-${infer b}`
    ? CmpU<b, a>
    : "<"
  : B extends `-${string}`
    ? ">"
    : CmpU<A & string, B & string>;

export type Max<A, B> = Cmp<A, B> extends "<" ? B : A;

/** Drop the N low-order digits (N given as a string of that length). */
type Drop<S extends string, N extends string> = N extends `${string}${infer NR}`
  ? S extends `${string}${infer SR}`
    ? Drop<SR, NR>
    : ""
  : S;

/**
 * Divide by 10^k, rounding half away from zero. `Half` is 5 * 10^(k-1) as a
 * little-endian digit string, e.g. "005" for k = 3.
 */
export type RoundShift<A, Half extends string> = A extends `-${infer M}`
  ? Neg<Strip<Drop<AddU<M, Half>, Half>>>
  : Strip<Drop<AddU<A & string, Half>, Half>>;

// ---------------------------------------------------------------- conversion

type Reverse<S extends string, Acc extends string = ""> = S extends `${infer H}${infer R}`
  ? Reverse<R, `${H}${Acc}`>
  : Acc;

/** Number literal -> Num. */
export type FromNumber<N extends number | bigint> = `${N}` extends `-${infer M}`
  ? `-${Reverse<M>}`
  : Reverse<`${N}`>;

/** Num -> number literal. */
export type ToNumber<A> = A extends `-${infer M}`
  ? `-${Reverse<M>}` extends `${infer N extends number}`
    ? N
    : never
  : Reverse<A & string> extends `${infer N extends number}`
    ? N
    : never;

/**
 * Fixed-point dot product: round(sum(x_i * w_i) / 10^3).
 * Positive and negative products are summed separately so the loop only does
 * unsigned additions; the two sums are combined once at the end.
 */
export type Dot<X, W, P extends string = "0", N extends string = "0"> = X extends readonly [
  infer x extends string,
  ...infer XR extends string[],
]
  ? W extends readonly [infer w extends string, ...infer WR extends string[]]
    ? x extends `-${infer xm}`
      ? w extends `-${infer wm}`
        ? Dot<XR, WR, AddU<P, MulU<xm, wm>>, N>
        : Dot<XR, WR, P, AddU<N, MulU<xm, w>>>
      : w extends `-${infer wm}`
        ? Dot<XR, WR, P, AddU<N, MulU<x, wm>>>
        : Dot<XR, WR, AddU<P, MulU<x, w>>, N>
    : never
  : RoundShift<SubM<P, N>, "005">;

// ---------------------------------------------------------------- division

type Succ = {
  "0": "1";
  "1": "2";
  "2": "3";
  "3": "4";
  "4": "5";
  "5": "6";
  "6": "7";
  "7": "8";
  "8": "9";
  "9": "9";
};

/** Subtract B from R while R >= B; returns [count, remainder]. Count is at most 9 here. */
type Fit<R extends string, B extends string, Q extends Digit = "0"> =
  CmpU<R, B> extends "<" ? [Q, R] : Fit<SubU<R, B>, B, Succ[Q]>;

/** Long division of magnitudes, most significant digit first. Returns [quotient, remainder]. */
type DivLoop<
  AMsb extends string,
  B extends string,
  Q extends string,
  R extends string,
> = AMsb extends `${infer D}${infer Rest}`
  ? Fit<R extends "0" ? D : `${D}${R}`, B> extends [
      infer QD extends string,
      infer R2 extends string,
    ]
    ? DivLoop<Rest, B, `${QD}${Q}`, R2>
    : never
  : [Strip<Q>, R];

/** floor(|A| / |B|) as [quotient, remainder]. */
export type DivModU<A extends string, B extends string> = DivLoop<Reverse<A>, B, "", "0">;
export type DivU<A extends string, B extends string> = DivModU<A, B>[0];

/** round(A / B), half away from zero: floor((2|A| + |B|) / 2|B|) with the sign of A*B. */
export type DivRound<A, B> =
  DivU<AddU<AddU<Abs<A>, Abs<A>>, Abs<B>>, AddU<Abs<B>, Abs<B>>> extends infer Q extends string
    ? IsNeg<A> extends IsNeg<B>
      ? Q
      : Neg<Q>
    : never;

// ---------------------------------------------------------------- square root

/** 10^ceil(len(V)/2): one zero per pair of digits, then a 1. */
type SqrtStart<V extends string, Acc extends string = ""> = V extends `${string}${string}${infer R}`
  ? SqrtStart<R, `${Acc}0`>
  : V extends ""
    ? `${Acc}1`
    : `${Acc}01`;

type Newton<V extends string, R extends string> =
  DivU<AddU<R, DivU<V, R>>, "2"> extends infer N extends string
    ? CmpU<N, R> extends "<"
      ? Newton<V, N>
      : R
    : never;

/** floor(sqrt(V)) for V >= 0, by Newton's method from 10^ceil(digits/2). */
export type ISqrt<V> = V extends "0" ? "0" : Newton<V & string, SqrtStart<V & string>>;

// ---------------------------------------------------------------- sums

/** Sum of a list of signed Nums (positives and negatives kept apart until the end). */
export type Sum<X, P extends string = "0", N extends string = "0"> = X extends readonly [
  infer x extends string,
  ...infer XR extends string[],
]
  ? x extends `-${infer m}`
    ? Sum<XR, P, AddU<N, m>>
    : Sum<XR, AddU<P, x>, N>
  : SubM<P, N>;

/** Sum of squares (always >= 0). */
export type SumSq<X, Acc extends string = "0"> = X extends readonly [
  infer x extends string,
  ...infer XR extends string[],
]
  ? SumSq<XR, AddU<Acc, MulU<Abs<x>, Abs<x>>>>
  : Acc;
