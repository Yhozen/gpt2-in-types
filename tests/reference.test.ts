import { describe, expect, test } from "vite-plus/test";
import {
  divRound,
  exp,
  gelu,
  isqrt,
  layerNorm,
  roundShift,
  softmax,
} from "../src/reference/fixed.ts";

const f = (v: bigint) => Number(v) / 1000;

describe("fixed-point reference", () => {
  test("roundShift rounds half away from zero", () => {
    expect(roundShift(1500n, 3)).toBe(2n);
    expect(roundShift(-1500n, 3)).toBe(-2n);
    expect(roundShift(1499n, 3)).toBe(1n);
  });

  test("divRound rounds half away from zero", () => {
    expect(divRound(7n, 2n)).toBe(4n);
    expect(divRound(-7n, 2n)).toBe(-4n);
    expect(divRound(7n, -3n)).toBe(-2n);
  });

  test("isqrt is floor(sqrt)", () => {
    for (let v = 0n; v < 2000n; v++) {
      const r = isqrt(v);
      expect(r * r <= v && (r + 1n) * (r + 1n) > v).toBe(true);
    }
  });

  test("exp tracks Math.exp", () => {
    for (let x = 0; x >= -9000; x -= 37) {
      expect(Math.abs(f(exp(BigInt(x))) - Math.exp(x / 1000))).toBeLessThan(0.0015);
    }
  });

  test("gelu tracks the tanh approximation", () => {
    const ref = (x: number) =>
      0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
    for (let x = -6000; x <= 6000; x += 41) {
      expect(Math.abs(f(gelu(BigInt(x))) - ref(x / 1000))).toBeLessThan(0.003);
    }
  });

  test("softmax sums to ~1", () => {
    const p = softmax([1200n, -300n, 50n, 2000n]);
    expect(Math.abs(Number(p.reduce((a, v) => a + v, 0n)) - 1000)).toBeLessThanOrEqual(2);
  });

  test("layerNorm normalizes", () => {
    const x = [1000n, 2000n, 3000n, 4000n];
    const one = x.map(() => 1000n);
    const zero = x.map(() => 0n);
    expect(layerNorm(x, one, zero).map(f)).toEqual([-1.342, -0.447, 0.447, 1.342]);
  });
});
