# gpt2-in-types

GPT-2 running inside the TypeScript type checker. No JavaScript runs at
inference time: the forward pass, the KV cache, the sampling loop and the
tokenizer are all type aliases, and the generated text comes out as the type
of a value.

```console
$ node scripts/generate.ts "ROMEO:" 16
prompt: "ROMEO:", 16 tokens, 793 ticks

--- type checker (27.8s, 435864142 instantiations, 768544K) ---
ROMEO:
I will the shal

--- matches fixed-point reference: yes ---
--- float64 model agrees ---
```

Inspired by [GPT-2 in CSS](https://shubhampatil.dev/blog/gpt2-css)
([repo](https://github.com/ShubhamPatilsd/gpt2-in-css)), which made a browser's
layout engine do the matrix math. Here the compute engine is
[tsrs](https://github.com/oxc-project/tsrs), the Rust port of the TypeScript 7
checker.

## What runs where

| Part                                                                                          | Where                                        |
| --------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Training (PyTorch)                                                                            | `scripts/train.py`, once, offline            |
| Float weights → fixed-point type                                                              | `scripts/export.ts`, once, offline           |
| Tokenizer, embeddings, LayerNorm, attention, KV cache, GELU MLP, LM head, argmax, decode loop | **the type checker** (`src/gpt2`, `src/num`) |
| Exact runtime reference (BigInt) to check it                                                  | `src/reference/fixed.ts`                     |

The architecture is GPT-2's: learned token and position embeddings, pre-LN
blocks, causal multi-head self-attention, a 4× MLP with tanh-GELU, a final
LayerNorm and an LM head tied to the token embedding. The size is not: this is
a character-level model with 4 layers, 4 heads and 32 dimensions (55k
parameters) trained on Tiny Shakespeare, against GPT-2 small's 12 layers, 12
heads, 768 dimensions and 124M parameters. See [Scaling](#scaling).

## How it works

**Numbers.** The type system has no floats, so every value is a fixed-point
integer at scale 10³ (`-1.235` is stored as `-1235`). Integers are decimal digit
strings, least significant digit first (`"-5321"`), so addition and
multiplication can walk both operands from the front with template literal
`infer`, using 10×10 digit lookup tables (`src/num`). Long division, Newton's
integer square root and a digit-decomposed `exp` (e^-(i.abc) = e^-i · e^-.a ·
e^-.0b · e^-.00c, from four small tables) cover LayerNorm, softmax and GELU.

**Ticks.** TypeScript stops after about 5M type instantiations per statement
(TS2589), and one token needs ~25M. So the model is a state machine:
`Step<Model, S>` does one bounded piece of work and returns the next state, and
a generated "clock" file chains the ticks, one statement each. Per position the
ticks are: embed, then per layer qkv, one attention tick per head, output
projection, fc, gelu, proj, and finally the LM head once the prompt is
consumed.

```ts
type S0 = Init<Model, "ROMEO:", 16>;
type S1 = Step<Model, S0>;
type S2 = Step<Model, S1>;
// …
export type Text = Output<Model, S793>;
```

The checker resolves types lazily, which fights this design in two ways, and
both show up as TS2589 rather than as wrong answers:

- A value written into the next state without being read is only computed when
  a later statement reads it, so the work piles up into one statement. Each
  tick computes its results in `extends infer` checks first.
- A state field that a tick leaves alone stays a reference to the previous
  state's field. After ~100 ticks that chain exceeds the instantiation depth
  limit. The state is a tuple, and `Step` re-reads every element of it, so each
  state is fully concrete when its statement ends.

Recursion over positions or characters is tail-recursive throughout; TypeScript
allows ~1000 levels of tail recursion but only ~50 of the other kind.

**Exactness.** `src/reference/fixed.ts` implements the same fixed-point
arithmetic with BigInt and the same rounding at every step, so the type
checker must reproduce it digit for digit. `tests/types/ops.gen.ts` checks
every op on random inputs; the examples and `scripts/generate.ts` check whole
generations. A float64 implementation (`src/reference/float.ts`) measures how
far the quantized model drifts from the trained network.

## Usage

Requires Node.js 26 and pnpm (installed on demand via `devEngines`).

```sh
vp install
node scripts/generate.ts "JULIET:" 24   # run the model in the type checker
vp run examples                          # check the committed example clocks
vp run check                             # format, lint, type check (incl. op tests)
vp test                                  # reference + quantization tests
```

To retrain (needs [uv](https://docs.astral.sh/uv/)):

```sh
vp run train      # PyTorch, writes model/weights.json
vp run export     # writes src/model/weights.ts
vp run codegen    # regenerates digit/exp tables and op tests
```

## Performance

Measured on an Apple Silicon Mac, 16 new tokens after a 6-character prompt
(793 ticks, ~436M type instantiations), same output from both:

| Checker                 | Time  | Peak memory |
| ----------------------- | ----- | ----------- |
| tsrs 0.23 (`--fast`)    | 28 s  | 0.75 GB     |
| tsgo (TypeScript 7.0.2) | 118 s | 8.9 GB      |

The full 64-character context (`"First Citizen:"` + 50 tokens, 2381 ticks, 1.3B
instantiations) takes 78 s and 2.2 GB with tsrs.

**Accuracy.** The type checker always matches the fixed-point reference
exactly. The fixed-point model matches the float64 model on short continuations
(`vp test` checks at least 8 agreeing characters for several prompts); on long
ones greedy decoding eventually branches on a near-tie, e.g. after 28
characters for `"First Citizen:"`.

## Scaling

Cost is roughly 700 instantiations per multiply-add, and tsrs does about 15–20M
instantiations a second. GPT-2 small needs ~124M multiply-adds per token, which
puts it at about 85 billion instantiations, over an hour of checking and far
more memory than the checker's caches can hold, per token. Its weights as
digit strings would also be a ~1 GB source file. The path there is the same one
the CSS project took: cheaper arithmetic (bigger lookup tables, W8A8
quantization), cache-friendly evaluation, and splitting the model across files.
