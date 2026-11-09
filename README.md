# gpt2-in-types

GPT-2 running inside the TypeScript type checker. Hover a type in your editor
and the tooltip is the model's output:

```ts
import type { Generate } from "./src/engine/index.ts";

export type Text = Generate<"ROMEO:", 24>;
//          ^? type Text = "ROMEO:\nWhe the crome the love "
```

No JavaScript runs at inference time. Tokenizer, embeddings, LayerNorm,
causal multi-head attention with a KV cache, the GELU MLP, the LM head,
sampling and the decode loop are all type aliases, evaluated by
[tsrs](https://github.com/oxc-project/tsrs) (the Rust port of the TypeScript 7
checker) when the editor asks for the hover.

Inspired by [GPT-2 in CSS](https://shubhampatil.dev/blog/gpt2-css).

## Results

Measured on an Apple Silicon Mac, through the real language server
(`node scripts/hover.ts`, which drives `tsrs --lsp` the way VS Code does):

| Hover                                      | Time  | Output                                                             |
| ------------------------------------------ | ----- | ------------------------------------------------------------------ |
| `Generate<"ROMEO:", 24>`                   | ~2 s  | `ROMEO:⏎Whe the crome the love `                                   |
| `Generate<"ROMEO:", 58>` (full 64 context) | ~12 s | `ROMEO:⏎Whe the crome the love thy love sie.⏎⏎⏎KI LOMMENTESTER:⏎I` |
| `Generate<"ROMEO:", 24, "greedy">`         | ~2 s  | `ROMEO:⏎The the he sear the the`                                   |

Every output is checked digit for digit against an exact integer reference
(`src/reference/int.ts`). The CLI-checked version of the same generation
(`examples/romeo.ts`, split into statements, see below) takes 1.2 s / 0.4 GB
with tsrs and 11 s / 4.7 GB with tsgo (TypeScript 7.0).

The model is GPT-2's architecture at toy scale: 4 layers, 4 heads, 32
dimensions, 64-character context, character-level, trained on Tiny
Shakespeare (56k parameters; integer validation loss 2.13, against 1.75 for a
float model of the same size). GPT-2 small is 124M parameters; see
[Scaling](#scaling).

## How it works

### The cost model of a type checker

Everything follows from what is cheap and what is expensive for the checker.
I measured these before designing the engine:

| Operation                                                         | Cost                                                   |
| ----------------------------------------------------------------- | ------------------------------------------------------ |
| Template literal concat `` `${A}${B}` ``, any length              | ~3 instantiations                                      |
| Prefix match `` A extends `${B}${infer R}` `` (subtract, compare) | ~3–10 instantiations; the string compare is native     |
| Property lookup in a large object type                            | ~1–2 instantiations; the hash lookup is native         |
| Tuple spread or tuple subtraction                                 | ~1–2 instantiations _per element_; tuples cap at 9,999 |
| Decimal digit-by-digit arithmetic (this project's first version)  | ~700 instantiations per multiply-add                   |

And the limits: the CLI aborts a single statement after ~5M instantiations
(TS2589), every evaluation stops at 100 levels of nesting, non-tail recursion
at ~50 levels and tail recursion at 1000. An editor hover is not subject to
the 5M budget (a 12M-instantiation type hovers fine) but is subject to the
depth limit.

So the engine represents numbers as **unary strings** (`"111"` is 3, `"-11"`
is −2). Addition is concatenation and comparison is a prefix match, both
native and independent of length. It never multiplies.

### An integer GPT-2 with no multiplications

The model is trained, quantization-aware, to run as integers with only
additions and lookups:

- **Ternary weights.** Every linear layer is a BitLinear with weights in
  {−1, 0, 1} and one absmean scale per matrix
  ([BitNet b1.58](https://arxiv.org/abs/2402.17764), Ma et al. 2024; BitLinear
  from [BitNet](https://arxiv.org/abs/2310.11453), Wang et al. 2023). A matrix
  compiles to a type alias whose row _j_ is one template literal: the inputs
  with weight +1 concatenated, minus those with weight −1. Inputs are offset
  to be non-negative so each nonzero weight is one span, and biases are
  constant runs of 1s. The LM head uses int4 weights, each input repeated |w|
  times (BitNet also keeps the head at higher precision).
- **Lookup tables instead of arithmetic.** Requantization, GELU, `exp` and the
  softmax normalizer are monotone functions of an integer, so each one is
  compiled into a binary decision tree of thresholds
  (`[ToS<t>, ifLess, ifGreaterOrEqual]`). Evaluating one is a few prefix
  matches. This is the table-lookup approach of
  [T-MAC](https://arxiv.org/abs/2407.00088) (Wei et al., EuroSys '25) and
  [bitnet.cpp](https://arxiv.org/abs/2502.11880), applied to the non-linear
  ops as well.
- **Integer-only non-linear ops with static scales**, as in
  [I-BERT](https://arxiv.org/abs/2101.01321) (Kim et al., ICML '21) and
  [Jacob et al. 2018](https://arxiv.org/abs/1712.05877):
  - LayerNorm (no affine parameters): the mean comes from a threshold search.
    σ comes from the sum of squares, accumulated as two unary limbs
    (c² = 512·hi + lo) and looked up in a tree built from a
    3-significant-digit square-root table. The output, `round(32·c/σ)`, comes
    from binary long division against doublings of σ.
  - Softmax: `e_j = round(127·exp(−d_j·u))`, `L = round(ln(Σe/127)/u)`,
    `p_j = round(63·exp(−(d_j + L)·u))` with `d_j = max(s) − s_j`, each one a
    tree.
  - q/k/v use 4-bit magnitudes. Low-bit KV caches are viable per
    [SAW-INT4](https://arxiv.org/abs/2604.19157) and
    [TurboQuant](https://arxiv.org/abs/2504.19874); here they use plain static
    scales with no rotation.
- **Sampling** is Gumbel-max with a seeded noise table (temperature 0.6) added
  to the head accumulators before the argmax. `"greedy"` skips it.

Training (`scripts/train.py`) runs the exact integer forward pass with
straight-through gradients: 1.5k float steps, then the scales are calibrated
and frozen and 12.5k integer steps follow. `scripts/compile.ts` turns the
result into `model/program.json` (integers and tables, for the reference) and
`src/model/program.ts` (the same program as types).

### Making it hoverable

A hover evaluates the requested type on demand, so laziness matters:

- Every intermediate result is forced with `extends infer X extends ...`
  before it is stored, so nothing stays a deferred reference to earlier work.
  Deep chains of those trip the depth limit.
- Mapped types are only applied to bare type parameters (that is what keeps
  tuples as tuples), and loops are tail-recursive.
- The CLI's 5M budget per statement is the one place where hover and CLI
  differ. The hover alias computes everything in one go. `examples/romeo.ts`,
  checked by `vp run examples`, runs the same generation as a chain of
  `Steps<State, 6>` statements and asserts the result.

## Usage

Requires Node.js 26 and pnpm (installed on demand via `devEngines`).

```sh
vp install
node scripts/generate.ts "JULIET:" 24          # hover via tsrs --lsp, compare to the reference
node scripts/generate.ts "JULIET:" 24 --greedy
node scripts/hover.ts examples/hover.ts Text    # hover any type, like an editor
vp run examples                                 # CLI check of examples/romeo.ts
vp run check                                    # format, lint, type check (incl. engine tests)
vp test                                         # reference and compiled-table tests
```

In VS Code or Cursor, open `examples/hover.ts` and hover `Text`. The committed
`.vscode/settings.json` points the TypeScript 7 extension at tsrs.

To retrain (needs [uv](https://docs.astral.sh/uv/)):

```sh
vp run train     # PyTorch QAT, writes model/qat.json (~6 min on Apple Silicon)
vp run compile   # program.json, src/model/program.ts, engine tests, examples/romeo.ts
```

## The playbook, triaged

How the 2025–26 inference playbook maps onto a type checker. Its "hardware" is
a single-threaded, memoizing interpreter where lookups and string
concatenation are cheap and everything else is not.

| Technique (papers)                                                                           | Used?               | Why                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Low-precision compute (BitNet b1.58; ResServe for FP4)                                       | **Yes**             | The core idea: ternary weights turn every mat-vec into concatenation. FP4 formats and residue compensation (ResServe) are GPU-specific.                                                                                                           |
| Table lookup instead of compute (T-MAC, bitnet.cpp; Engram's O(1) lookup)                    | **Yes**             | A property lookup in an object type is a native hash lookup, the cheapest operation available. All non-linear ops are tables or trees. Engram's conditional n-gram memory itself is not implemented.                                              |
| Low-bit KV cache (SAW-INT4, TurboQuant)                                                      | **Yes**, simplified | q/k/v with 4-bit magnitudes and static scales. Their rotations would likely recover some quality.                                                                                                                                                 |
| Reusing computation, storage for compute (KV cache; Mooncake)                                | **Yes**, in spirit  | Positions reuse cached K/V. The checker memoizes every instantiation, and the CLI example chains statement-sized states.                                                                                                                          |
| Sparse attention (DeepSeek-V3.2 DSA, LongCat, AttentionIndex, Multi-layer Indexer)           | No                  | With 64 positions, dense attention over the cache is cheaper than any indexer. It would matter at longer contexts.                                                                                                                                |
| Linear or recurrent attention (Kimi Linear, Qwen3-Next)                                      | No, best next step  | Attention is why hover time grows with context (2 s at 24 tokens, 12 s at 58). A fixed-size recurrent state would make the cost per token constant, but it is a different architecture from GPT-2.                                                |
| Speculative decoding (DSpark, EAGLE-3, DFlash)                                               | No                  | Speculation trades parallel compute for fewer sequential passes. The checker has no parallelism to spare, so verifying k drafts costs about as much as generating them. "Speculative Decoding: Performance or Illusion?" makes the general point. |
| MoE and expert parallelism (Least-Loaded EP, wide EP)                                        | No                  | One small dense model on one thread.                                                                                                                                                                                                              |
| Kernels and serving (FlashAttention-4, HiSparse, DualPath, async scheduling, disaggregation) | No                  | Hardware- and cluster-specific. The closest analogue is splitting work to fit a budget, as the CLI example does per statement.                                                                                                                    |
| SIMD                                                                                         | In effect           | One template-literal instantiation adds about 20 values natively, much like one vector instruction. T-MAC's lookup tables are the SIMD table-lookup (TBL/PSHUF) idea.                                                                             |

Every paper above was checked to exist and to say what is quoted here. Two
claims in the playbook about DeepInfer (the +5.4% EP64 figure and
time-budgeted chunked prefill) could not be confirmed against its README, so
neither is relied on.

## Scaling

The engine costs about 300k type instantiations per position at this size,
against about 25M for this project's first, digit-arithmetic version (which
could not be hovered at all). Matrix work grows with d², attention with
context length. GPT-2 small (d = 768, 12 layers, 1024 context) would be
roughly 600× the matrix work per position plus far longer attention: hours per
token, and more memory than the checker's caches can hold. Getting there would
take a recurrent attention variant, wider tables (T-MAC-style grouping of
several inputs per lookup), and probably splitting the model across files.
