# gpt2-in-types

GPT-2 running inside the TypeScript type checker. Hover a type in your editor
and the tooltip is the model's output:

```ts
import type { Generate } from "./src/engine/index.ts";

export type Text = Generate<"Once upon a time", 24, "greedy">;
//          ^? type Text = "Once upon a time, there was a little gir"
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

| Hover                                                          | Time   | Output                                                             |
| -------------------------------------------------------------- | ------ | ------------------------------------------------------------------ |
| `Generate<"Once upon a time", 24, "greedy">`                   | ~3.5 s | `Once upon a time, there was a little gir`                         |
| `Generate<"Once upon a time", 24>` (sampled)                   | ~3.6 s | `Once upon a time, there was a proble whi`                         |
| `Generate<"Once upon a time", 48>` (full 64-character context) | ~8.5 s | `Once upon a time, there was a proble whise. The little girl was ` |

Every output is checked digit for digit against an exact integer reference
(`src/reference/int.ts`), and the reference in turn reproduces the PyTorch
integer forward pass used in training exactly (`vp run parity`: 0 of 16,384
positions differ). The CLI-checked version of the 24-character
generation (`examples/story.ts`, split into statements, see below) takes about
4 s.

The model is GPT-2's architecture at toy scale: 4 layers, 8 heads, 64
dimensions, 64-character context, character-level, trained on TinyStories
(Eldan & Li 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759); 211k
parameters, integer validation loss 1.14 nats per character on a fixed set of
256 validation windows, computed by the reference itself). For comparison, the smallest
pretrained TinyStories model (TinyStories-1M) is 64 dimensions and 8 layers
with a 50k-token vocabulary, and GPT-2 small is 124M parameters. See
[Experiments](#experiments) and [Scaling](#scaling).

**Editor setup matters:** tsrs's performance modes (`TSRS_FAST=1`) make this
workload about 3× faster, but the language server does not enable them by
default. `.vscode/settings.json` points the TypeScript 7 extension at
`tsdk/tsgo`, a two-line wrapper that sets the variable.

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
at ~50 levels and tail recursion at 1000. An editor hover has the same 5M
budget, counted differently: see [Making it hoverable](#making-it-hoverable).

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
  with weight +1 concatenated, minus those with weight −1. Each input is
  split into a positive part and a negative magnitude (one of them empty), so
  a weight of −1 just swaps which part goes where, and biases are constant
  runs of 1s. The LM head uses int4 weights, each input repeated |w|
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
- **The 5M budget.** The checker gives up with `any` once 5M instantiations
  have accumulated since its counter was last zeroed. The CLI zeroes it for
  every statement. A hover never does, but reading tsrs's source
  (`crates/tsrs_checker`) shows the counter is also zeroed, as a side effect,
  whenever a literal type node is resolved for the first time. While the
  model's tables are cold they supply such resets, which is why a first hover
  worked and a second hover in the same editor session returned `any` (and
  why deeper models failed part-way through the context). So the compiled
  program includes `Fuel`, a pool of 254k literal nodes keyed by position and
  two tokens, and the engine looks one up before every block (`Refuel`).
  `vp run check:hover` hovers three full-context generations in one session
  to guard this. It relies on an implementation detail of tsrs 0.23, which is
  why the version is pinned.
- The CLI budget is per statement and the pool does not lift it reliably, so
  `examples/story.ts`, checked by `vp run examples`, runs the same generation
  as a chain of `Advance<State, 2>` statements (two positions each) and
  asserts the result.
- A prompt with a character the model does not know hovers as an error message
  (`"Error: the character '3' is not in the model's vocabulary"`), not `any`.

## Usage

Requires Node.js 26 and pnpm (installed on demand via `devEngines`).

```sh
vp install
node scripts/generate.ts "Tom and Lily" 24      # hover via tsrs --lsp, compare to the reference
node scripts/generate.ts "Tom and Lily" 24 --greedy
node scripts/hover.ts examples/hover.ts Text    # hover any type, like an editor
vp run examples                                 # CLI check of examples/story.ts
vp run check                                    # format, lint, type check (incl. engine tests)
vp test                                         # reference, compiled tables, parity vectors
vp run check:hover                              # three full-context hovers in one editor session
```

In VS Code or Cursor, open `examples/hover.ts` and hover `Text`. The committed
`.vscode/settings.json` points the TypeScript 7 extension at tsrs.

To retrain (needs [uv](https://docs.astral.sh/uv/)):

```sh
vp run train     # PyTorch QAT, writes model/qat.json (~10 min on Apple Silicon)
vp run compile   # program.json, src/model/program.ts, engine tests, examples/story.ts
vp run parity    # model/parity.json: PyTorch vs reference, and the reference's validation loss
```

## Experiments

All runs use the same integer architecture and training recipe (1.5k float
steps, then 12.5k integer steps with calibrated, frozen scales). Hover times are
for the default prompt plus 24 characters with `TSRS_FAST=1`.

| Data        | Layers × dims | Params | Integer val loss | Hover, 24 chars | Notes                                                                                                                   |
| ----------- | ------------- | ------ | ---------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Shakespeare | 4 × 32        | 56k    | 2.13             | 1.4 s           | Previous default. `ROMEO:⏎Whe the crome the love `                                                                      |
| TinyStories | 4 × 32        | 56k    | ~2.32 (stopped)  | –               | Stalled; stopped to free the GPU                                                                                        |
| TinyStories | **4 × 64**    | 211k   | **1.11**         | **3.6 s**       | **Default.** Greedy: `Once upon a time, there was a little girl named Lily.`                                            |
| TinyStories | 8 × 64        | 410k   | 1.04             | 7.3 s           | Best text (`Tom and Lily went to the park. They were so happy to see the`), hovers take about twice as long (see below) |

What made the difference:

- **The data, more than the size.** TinyStories' simple vocabulary is what the
  TinyStories paper is about: tiny models write coherent English on it. At the
  same 64-dimension width, Shakespeare-style text stays word salad.
- **`TSRS_FAST=1`:** 15 s → 4.9 s for the same hover.
- **Input encoding.** Offsetting the matrix inputs to be non-negative halves
  the template spans, but every input string gets ~127 characters longer;
  splitting inputs into positive and negative parts measured faster (15.0 s vs
  18.4 s before fast mode), so the engine uses the split.
- **KV cache rebuilt by splicing**, not mapping, so untouched layers stay
  resolved instead of becoming chains of lazy references.

Not used: the pretrained TinyStories checkpoints. Their float weights would
need integer retraining anyway, and their 50,257-token BPE vocabulary would
make the LM head alone ~50k rows per generated token (about 20× the whole
per-token cost here) and require a BPE tokenizer in types.

The 8-layer model used to return `any` beyond roughly 44 positions. That was
the 5M budget (see [Making it hoverable](#making-it-hoverable)): at late
positions a single position of that model costs more than 5M counted
instantiations. With a budget reset before every block it now runs to the full
64-character context, also on repeated hovers. A 16-layer, 32-dimension model
(untrained, built to bisect the limit) does too.

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
| Linear or recurrent attention (Kimi Linear, Qwen3-Next)                                      | No, best next step  | Attention is why hover time grows with context (the 48-character hover takes more than twice the 24-character one). A fixed-size recurrent state would make the cost per token constant, but it is a different architecture from GPT-2.           |
| Speculative decoding (DSpark, EAGLE-3, DFlash)                                               | No                  | Speculation trades parallel compute for fewer sequential passes. The checker has no parallelism to spare, so verifying k drafts costs about as much as generating them. "Speculative Decoding: Performance or Illusion?" makes the general point. |
| MoE and expert parallelism (Least-Loaded EP, wide EP)                                        | No                  | One small dense model on one thread.                                                                                                                                                                                                              |
| Kernels and serving (FlashAttention-4, HiSparse, DualPath, async scheduling, disaggregation) | No                  | Hardware- and cluster-specific. The closest analogue is splitting work to fit a budget, as the CLI example does per statement.                                                                                                                    |
| SIMD                                                                                         | In effect           | One template-literal instantiation adds about 20 values natively, much like one vector instruction. T-MAC's lookup tables are the SIMD table-lookup (TBL/PSHUF) idea.                                                                             |

Every paper above was checked to exist and to say what is quoted here. Two
claims in the playbook about DeepInfer (the +5.4% EP64 figure and
time-budgeted chunked prefill) could not be confirmed against its README, so
neither is relied on.

## Scaling

The engine costs about 1.2M type instantiations per position at this size (300k at 32 dimensions),
against about 25M for this project's first, digit-arithmetic version (which
could not be hovered at all). Matrix work grows with d², attention with
context length. GPT-2 small (d = 768, 12 layers, 1024 context) would be
roughly 600× the matrix work per position plus far longer attention: hours per
token, and more memory than the checker's caches can hold. Getting there would
take a recurrent attention variant, wider tables (T-MAC-style grouping of
several inputs per lookup), and probably splitting the model across files.
