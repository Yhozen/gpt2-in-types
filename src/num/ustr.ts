/**
 * Unary strings: the magnitude n is the string of n "1"s.
 *
 * The checker interns string literal types and builds template literals and
 * prefix matches natively, so concatenation (`${A}${B}`), subtraction
 * (A extends `${B}${infer R}`) and comparison cost a handful of type
 * instantiations no matter how long the strings are. (Tuples, by contrast,
 * cost about one instantiation per element and are capped below 10,000.)
 *
 * Signed values are a sign ("" or "-") plus a magnitude: { s, m } is avoided
 * on purpose (objects resolve lazily); a signed unary is the string itself
 * with an optional "-" prefix, e.g. "-111" is -3 and "" is 0.
 */

/** Decimal digit -> unary. */
type UD = {
  "0": "";
  "1": "1";
  "2": "11";
  "3": "111";
  "4": "1111";
  "5": "11111";
  "6": "111111";
  "7": "1111111";
  "8": "11111111";
  "9": "111111111";
};
type Ten<S extends string> = `${S}${S}${S}${S}${S}${S}${S}${S}${S}${S}`;
type FromDigits<
  D extends string,
  Acc extends string = "",
> = D extends `${infer H extends keyof UD}${infer R}` ? FromDigits<R, `${Ten<Acc>}${UD[H]}`> : Acc;

/** Number literal -> signed unary string. */
export type ToS<N> = N extends number
  ? `${N}` extends `-${infer M}`
    ? `-${FromDigits<M>}`
    : FromDigits<`${N}`>
  : never;

/** Unsigned: is A >= B? */
export type GeU<A extends string, B extends string> = A extends `${B}${string}` ? true : false;

/** Signed a + b. */
export type AddS<A extends string, B extends string> = A extends `-${infer a}`
  ? B extends `-${infer b}`
    ? `-${a}${b}`
    : SubU<B, a>
  : B extends `-${infer b}`
    ? SubU<A, b>
    : `${A}${B}`;

/** Unsigned a - b as a signed unary. */
export type SubU<A extends string, B extends string> = A extends `${B}${infer R}`
  ? R
  : B extends `${A}${infer R}`
    ? `-${R}`
    : never;

export type NegS<A extends string> = A extends `-${infer a}` ? a : A extends "" ? "" : `-${A}`;

/**
 * Threshold search: the largest W-bit q with |A| >= T[q], built one bit at a
 * time from the top (binary search). T maps W-bit strings to thresholds
 * (number literals); Z is W zeros. Returns q as a W-bit string.
 */
export type SearchGe<
  A extends string,
  T,
  Z extends string,
  P extends string = "",
> = Z extends `0${infer Zr}`
  ? A extends `${ToS<T[`${P}1${Zr}` & keyof T]>}${string}`
    ? SearchGe<A, T, Zr, `${P}1`>
    : SearchGe<A, T, Zr, `${P}0`>
  : P;
