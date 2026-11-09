import { existsSync } from "node:fs";
import { describe, expect, test } from "vite-plus/test";
import { generate } from "../src/reference/fixed.ts";
import { generateFloat } from "../src/reference/float.ts";
import { WEIGHTS_PATH, decode, encode, loadFloat, quantize } from "../src/reference/quantize.ts";

describe.runIf(existsSync(WEIGHTS_PATH))("fixed-point model vs float64 model", () => {
  const float = loadFloat();
  const model = quantize(float);
  const prompts = ["ROMEO:", "JULIET:", "First Citizen:\n", "KING HENRY", "O, ", "What is"];

  test.each(prompts)("greedy output for %j mostly agrees", (prompt) => {
    const ids = encode(model.vocab, prompt);
    const fixed = generate(model, ids, 24).slice(ids.length);
    const ref = generateFloat(float, ids, 24).slice(ids.length);
    // Greedy decoding diverges for good after the first different token, so
    // compare the agreeing prefix: it should cover most of the continuation.
    const prefix = fixed.findIndex((t, i) => t !== ref[i]);
    const agreed = prefix < 0 ? fixed.length : prefix;
    expect(
      agreed,
      `fixed: ${JSON.stringify(decode(model.vocab, fixed))}\nfloat: ${JSON.stringify(decode(model.vocab, ref))}`,
    ).toBeGreaterThanOrEqual(8);
  });
});
