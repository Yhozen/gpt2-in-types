# gpt2-in-types

GPT-2 running inside the TypeScript type checker. Hover a type in your editor
and the tooltip is a story the model just wrote:

```ts
import type { Generate } from "./src/engine/index.ts";

export type Text = Generate<"Once upon a time", 64>;
//          ^? type Text = "Once upon a time, there was a little girl named Lily.
//             She had a big, red ball that she loved very much. One day, she found
//             a big box in the park. Amy was very happy and wanted to play with the ball."
```

No JavaScript runs at inference time. Tokenizer, embeddings, LayerNorm,
causal multi-head attention with a KV cache, the GELU MLP, the LM head,
sampling and the decode loop are all type aliases, evaluated by
[tsrs](https://github.com/oxc-project/tsrs) (the Rust port of the TypeScript 7
checker) when the editor asks for the hover.

Inspired by [GPT-2 in CSS](https://shubhampatil.dev/blog/gpt2-css). How it
got here, dead ends included: [docs/JOURNEY.md](docs/JOURNEY.md).

## Results

Measured on an Apple Silicon Mac, through the real language server
(`node scripts/hover.ts`, which drives `tsrs --lsp` the way VS Code does), all
in one session ([`examples/hover.ts`](examples/hover.ts)):

| Hover                                                       | Time  | Output                                                                                                                                                                                                   |
| ----------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tokenize<"Once upon a time">`                              | 0.7 s | `["Once", " upon", " a", " time"]`                                                                                                                                                                       |
| `Generate<"Once upon a time", 64>` (sampled)                | 36 s  | `Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved very much. One day, she found a big box in the park. Amy was very happy and wanted to play with the ball.` |
| `Generate<"One day, a girl named Mia found", 64, "greedy">` | 34 s  | `One day, a girl named Mia found a big box. She was very happy. Mia wanted to show her mom. She asked her mom, "Mom, can I have the box?" Her mom said, "Yes, Mia`                                       |
| `Generate<"The little dog", 64, "sample", 3>`               | 18 s  | `The little dog was so happy and they all played together in the rain. They had so much fun together and became good friends.⏎` (stops at its end-of-story token)                                        |

A full 64-token context takes 35–75 s and about 8 GB of memory in the language
server. Every output is checked token for token against an exact integer
reference (`src/reference/int.ts`), and the reference in turn reproduces the
PyTorch integer forward pass used in training exactly (`vp run parity`: 0 of
16,384 positions differ). The sample above is noise offset 0, the default, not
a pick; the other two prompts and the offset 3 were chosen by reading the
reference's outputs for offsets 0–15.

The model is GPT-2's architecture at toy scale: 8 layers, 8 heads, 128
dimensions and a 64-token context over a 256-piece subword vocabulary (about
2.2 characters per token, so ~140 characters of context), trained on
TinyStories (Eldan & Li 2023,
[arXiv:2305.07759](https://arxiv.org/abs/2305.07759)). It has 1.66M parameters
and an integer validation loss of 0.71 nats per character on 256 fixed
validation windows, computed by the reference itself; the previous default, a
4-layer, 64-dimension character model, scored 1.14 and wrote
`Once upon a time, there was a proble whise. The little girl was s`. For
comparison, the smallest pretrained TinyStories model (TinyStories-1M) is 64
dimensions and 8 layers with a 50k-token vocabulary, and GPT-2 small is 124M
parameters. See [Experiments](#experiments) and [Scaling](#scaling).

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
  to the head accumulators before the argmax. A fourth `Generate` argument
  selects a reproducible offset in that table; `"greedy"` skips the noise.

- **Tokens:** 256 of them, the 63 characters, 192 frequent words and word
  parts such as `" upon"`, `"ing"` and `" Lily"`, and an end-of-story token.
  Text is split greedily, longest piece first (`scripts/tokenizer.py`). The rule is
  trivial to state in Python, TypeScript and types alike: the type-level
  `Encode` walks a trie of the pieces one character at a time. `Tokenize<S>`
  shows the split.

Training (`scripts/train.py`) runs the exact integer forward pass with
straight-through gradients: 3k float steps, then the scales are calibrated and
frozen and 15k integer steps follow, with the learning rate decaying to zero.
`scripts/compile.ts` turns the result into `model/program.json` (integers and
tables, for the reference) and `src/model/program.ts` (the same program as
types, 22 MB).

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
  program includes `Fuel`, a pool of literal nodes keyed by position, layer
  and a rolling hash of the tokens so far, and the engine looks one up before
  every block (`Refuel`). It relies on an implementation detail of tsrs 0.23,
  which is why the version is pinned.
- **The 16M relation cache.** Every structural comparison the checker makes is
  cached, the cache is never emptied, and each comparison may only do
  `(16M − cache size) / 8` units of work. Past 16M entries every conditional
  type takes its false branch, so a hover shows `never` (the CLI reports
  TS2859). Tuple loops therefore use mutable patterns
  (`X extends [infer H, ...infer R]`): against a mutable tuple a match is the
  same type, accepted by identity with no cache entry, while a
  `readonly [...]` pattern cost one entry per loop step and limited the
  8-layer model to three full-context hovers per session. Editing the file
  also starts a fresh checker.
- `vp run check:hover` hovers five full-context generations in one session,
  including different seeds for the same prompt, to guard both limits.
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
node scripts/generate.ts "Tom and Lily" 24 --seed 1
node scripts/hover.ts examples/hover.ts Tokens Text   # hover types in one session, like an editor
vp run examples                                 # CLI check of examples/story.ts
vp run check                                    # format, lint, type check (incl. engine tests)
vp test                                         # reference, compiled tables, parity vectors
vp run check:hover                              # five full-context hovers in one editor session
```

The second argument of `Generate` and of `scripts/generate.ts` counts tokens,
not characters. In VS Code or Cursor, open `examples/hover.ts` and hover
`Tokens`, then `Text`. The committed `.vscode/settings.json` points the
TypeScript 7 extension at tsrs. Give the language server time and memory: a
full story is about half a minute to a minute and ~8 GB.

For another reproducible sample, change the fourth argument:

```ts
type Story = Generate<"Once upon a time", 64, "sample", 1>;
```

Seed 0 preserves the original output. The default model supports literal integer
seeds 0–63 (in general, 0 to context length minus one), exported as `SamplingSeed`.
These are cyclic offsets into the stored noise sequence, not new PRNG runs;
different seeds can still produce the same text. Greedy output is independent of
the seed. When continuing with `Steps` or `Advance`, pass the same fourth seed
argument to each call. `--seed` also works with the CLI's `--out` mode.

To retrain (needs [uv](https://docs.astral.sh/uv/)):

```sh
# PyTorch QAT, writes model/qat.json (~20 min on Apple Silicon, plus a 600 MB download)
TOKENIZER=model/vocab-256.json TRAIN_MB=600 N_LAYER=8 N_EMBD=128 \
  QAT_ITERS=18000 QAT_WARMUP=3000 vp run train
vp run compile   # program.json, src/model/program.ts, engine tests, examples/story.ts
vp run parity    # model/parity.json: PyTorch vs reference, and the reference's validation loss
```

Plain `vp run train` still trains the earlier 4 × 64 character model.
`VOCAB=256 uv run scripts/tokenizer.py` relearns the vocabulary.

## Experiments

All runs use the same integer architecture. Losses are integer validation
losses in nats per character, so character and subword models compare
directly. Hover times are with `TSRS_FAST=1`.

Character models (1.5k float steps, then 12.5k integer steps with calibrated,
frozen scales, trained on the TinyStories validation file; 64-character
context; hover of the default prompt plus 24 characters):

| Data        | Layers × dims | Params | Val loss        | Hover | Notes                                                                             |
| ----------- | ------------- | ------ | --------------- | ----- | --------------------------------------------------------------------------------- |
| Shakespeare | 4 × 32        | 56k    | 2.13            | 1.4 s | First default. `ROMEO:⏎Whe the crome the love `                                   |
| TinyStories | 4 × 32        | 56k    | ~2.32 (stopped) | –     | Stalled; stopped to free the GPU                                                  |
| TinyStories | 4 × 64        | 211k   | 1.11            | 3.6 s | Previous default. Greedy: `Once upon a time, there was a little girl named Lily.` |
| TinyStories | 8 × 64        | 410k   | 1.04            | 7.3 s | `Tom and Lily went to the park. They were so happy to see the`                    |

Subword models (256-token vocabulary, 64-token context, extra data from the
TinyStories train split; validation is always the same held-out characters;
full-context hovers):

| Layers × dims | Params | Training                       | Val loss | Hover    | Notes                                |
| ------------- | ------ | ------------------------------ | -------- | -------- | ------------------------------------ |
| 8 × 64        | 435k   | 600-step probe, 200 MB         | 1.57     | –        | Had barely started learning          |
| 4 × 128       | 865k   | 600-step probe, 200 MB         | 1.16     | 24–42 s  |                                      |
| 8 × 128       | 1.66M  | 600-step probe, 200 MB         | 1.13     | ~45 s    | Picked for the long run              |
| **8 × 128**   | 1.66M  | 3k float + 15k integer, 600 MB | **0.71** | **36 s** | **Default.** See [Results](#results) |

What made the difference:

- **The data, more than the size.** TinyStories' simple vocabulary is what the
  TinyStories paper is about: tiny models write coherent English on it. At the
  same 64-dimension width, Shakespeare-style text stays word salad.
- **Subword tokens instead of more positions.** Doubling the character context
  to 128 cost more than it gained: attention and the per-hover budgets grow
  with positions. 64 subword tokens cover ~140 characters for the price of a
  bigger LM head, and let the model spend its capacity on words instead of
  spelling.
- **More data for a wider model.** The 22 MB validation file was enough for
  211k parameters; the 1.66M-parameter model trains on 600 MB of the train
  split as well.
- **A short float phase.** Switching to integers after 3k float steps, then
  decaying the learning rate to zero over 15k integer steps, worked. Switching
  at 20k left the integer phase stuck near its starting loss, and a 37k-step
  integer phase at a high learning rate drifted upward (see
  [the journey](docs/JOURNEY.md)).
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

### Longer context on Colab

The 4-layer, 64-dimension model was also trained with 128- and 256-character
contexts on a Colab Tesla T4, using the same 14,000-step recipe and seed 1337.
Both runs completed; their checkpoints and full training logs are saved here.

| Context | Loss on shared targets (nats/char) | Checkpoint                                           |
| ------- | ---------------------------------- | ---------------------------------------------------- |
| 64      | 1.1064                             | Previous default (`git show 772a00c:model/qat.json`) |
| 128     | 1.0732                             | [Experimental](model/context128.json)                |
| 256     | 1.5824                             | [Experimental](model/context256.json)                |

`scripts/evaluate-context.ts` scores the same 4,096 target characters for all
three models: the last 64 predictions in each of 64 fixed validation windows.
Each model receives its own context length. This avoids comparing different
random validation batches. The 128-context checkpoint scores about 3% better;
the 256-context checkpoint scores about 43% worse. These are single training
runs on validation data, not a separate test set or evidence that context alone
caused the difference. [Evaluation outputs](experiments/context-128.json) include
sampled and greedy text, checkpoint hashes and a hash of the normalized validation
text; [training logs](experiments/context-training.json) include the training-script
hash. The runs took about 43 and 86 minutes, partly sharing the GPU.

The compiled reference matches PyTorch's **float64** integer simulation at all
8,192 checked positions for 128 context and all 16,384 for 256 context. Float32
and float64 do not produce identical accumulators for these checkpoints
(6,976 and 16,198 positions differ respectively). The full-window float64
validation losses are 1.1534 and 1.6221; those averages include the early
predictions with little context and are different from the shared-target table.

Neither replaced the 64-character model, and the subword model has since
replaced all three. With the 128 checkpoint, two full-context hovers in one
server session matched the reference, but the third returned `never`; the same
third prompt worked in a fresh server. That is the symptom of the relation
cache limit found later (see [Making it hoverable](#making-it-hoverable)); it
was not retested after that fix. These checkpoints
are experimental: all three full-context hovers fail for the 256 checkpoint,
starting with `any` even in a fresh server. A successful fresh hover is not
sufficient evidence of editor reliability. The attempted budget-reset and materialization changes did
not fix that regression and were not included in the engine.

To reproduce in a scratch checkout (compilation regenerates tracked files):

```sh
QAT=model/context128.json vp run compile
QAT=model/context128.json PARITY_WINDOWS=64 vp run parity
QAT=model/context128.json node scripts/evaluate-context.ts
vp run check:hover  # exposed the repeated-hover failure before the cache fix

# Restore the default model and its parity vectors afterward.
vp run compile
vp run parity
```

Use `model/context256.json` for the 256-context run. The checked-in evaluation
outputs are [64](experiments/context-64.json), [128](experiments/context-128.json)
and [256](experiments/context-256.json).

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

The engine cost about 1.2M type instantiations per position for the 4 × 64
character model (300k at 32 dimensions), against about 25M for this project's
first, digit-arithmetic version (which could not be hovered at all). Matrix
work grows with layers × d², attention with context length, so the 8 × 128
default does about eight times the matrix work of 4 × 64; a full story takes
half a minute to a minute and ~8 GB. GPT-2 small (d = 768, 12 layers, 1024
context) would be roughly 50× the default's matrix work per position, plus a
50k-row LM head and far longer attention: hours per token, and more memory
than the checker's caches can hold. Getting there would
take a recurrent attention variant, wider tables (T-MAC-style grouping of
several inputs per lookup), and probably splitting the model across files.
