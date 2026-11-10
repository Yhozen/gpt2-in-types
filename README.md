# gpt2-in-types

I tried to port GPT-2 to the TypeScript type system. GPT-2 didn't fit (yet), so
I trained a language model that does. Hover a type in your editor and the
tooltip is a story it just wrote:

```ts
import type { Generate } from "./src/engine/index.ts";

export type Text = Generate<"Once upon a time", 64>;
//          ^? type Text = "Once upon a time, there was a little girl named Lily.
//             She had a big, red ball that she loved very much. One day, she found
//             a big box in the park. Amy was very happy and wanted to play with the ball."
```

No JavaScript runs. The tokenizer, embeddings, LayerNorm, causal multi-head
attention with a KV cache, the GELU MLP, the LM head, sampling and the decode
loop are all type aliases, evaluated by
[tsrs](https://github.com/oxc-project/tsrs) (the Rust port of the TypeScript 7
checker) when the editor asks what the type is. It takes about half a minute
and 8 GB of memory, which is a lot for a tooltip.

## But can it run an LLM?

"But can it run Doom?" is settled for TypeScript types:
[it can](https://socket.dev/blog/typescript-types-running-doom), at 12 days for
the first frame. Then I saw
[GPT-2 in CSS](https://shubhampatil.dev/blog/gpt2-css) run the real GPT-2 in a
browser with no JavaScript, and remembered
[TypeScript issue #14833](https://github.com/microsoft/TypeScript/issues/14833):
the type system is Turing complete. So in theory a type checker can run a
language model. In practice "Turing complete" says nothing about _fast_, and
the checker has no multiplication, no floats and no `exp`, and gives up after a
few million steps. What made it worth trying was tsrs, a checker fast enough
that the idea seemed possible.

The goal was GPT-2, hence the name. This is what I have so far.

## What it is, and what it isn't

**It is not OpenAI's GPT-2.** It is a model with GPT-2's architecture (learned
token and position embeddings, pre-LayerNorm blocks, causal multi-head
attention, GELU MLPs), shrunk and retrained so that it never multiplies:

|            | GPT-2 small    | This model                                                                    |
| ---------- | -------------- | ----------------------------------------------------------------------------- |
| Parameters | 124M           | 1.66M                                                                         |
| Shape      | 12 layers, 768 | 8 layers, 8 heads, 128 dimensions                                             |
| Context    | 1,024 tokens   | 64 tokens (~140 characters)                                                   |
| Vocabulary | 50,257 tokens  | 256 pieces                                                                    |
| Weights    | OpenAI's       | ternary {−1, 0, 1}, trained from scratch                                      |
| Trained on | the web        | [TinyStories](https://arxiv.org/abs/2305.07759) (stories for three-year-olds) |

So it writes children's stories and nothing else, and it will tell you that
Lily's ball belongs to Amy.

**Why the real one didn't fit.** GPT-2 small needs about 124 million
multiply-adds per token, and the type checker cannot multiply. Doing the
arithmetic digit by digit in types (this project's first version) costs about
700 type instantiations per multiply-add: roughly 87 billion per token, or
about four hours per token at the speed tsrs manages on this workload. GPT-2
in CSS could run the real weights because CSS has `calc()` and a font engine
that computes weighted sums natively. A type checker's only native arithmetic
is string concatenation, so the model here is trained to need nothing else.

|                 | GPT-2 in CSS                     | This project                                |
| --------------- | -------------------------------- | ------------------------------------------- |
| Native multiply | yes: `calc()` and variable fonts | none: addition is string concatenation      |
| Weights         | real GPT-2 small, 9-bit integers | ternary, trained from scratch               |
| Speed           | 15–45 s per token                | ~0.5–1 s per token (64 tokens in one hover) |
| Context         | 36 tokens                        | 64 tokens                                   |

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
16,384 positions differ). The first sample is noise offset 0, the default, not
a pick; the other two prompts and the offset 3 were chosen by reading the
reference's outputs for offsets 0–15.

The model's integer validation loss is 0.71 nats per character (perplexity 2.0
per character) on 256 fixed validation windows, computed by the reference
itself. All the models that came before it are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

**Editor setup matters:** tsrs's performance modes (`TSRS_FAST=1`) make this
workload about 3× faster, but the language server does not enable them by
default. `.vscode/settings.json` points the TypeScript 7 extension at
`tsdk/tsgo`, a two-line wrapper that sets the variable.

## The journey

It took two days, several dead ends, and one bug that could only be explained
by reading the checker's source.

### 1. Arithmetic out of digit strings: it ran, and hovered as `any`

The first version did the honest thing: numbers as strings of decimal digits,
with addition and multiplication implemented digit by digit in types, like
long arithmetic on paper. A 4-layer, 32-dimension character model trained on
Tiny Shakespeare matched a BigInt reference digit for digit.

Each multiply-add cost about 700 type instantiations and one position of the
model about 25 million. The command line gives up on any statement after 5
million, so generation was split into one statement per step. In the editor,
the hover showed `type Text = any`. A model you cannot hover is not much of a
demo.

### 2. A model that only adds

The second attempt started by measuring what the checker is fast at (the table
is under [How it works](#the-cost-model-of-a-type-checker)). Two operations
cost the same at any size: concatenating strings, and checking whether one
string starts with another. If a number is a string of that many `1`s,
addition is concatenation and comparison is a prefix match. Multiplication is
still impossible, so the model must not multiply: ternary weights, lookup
tables for everything non-linear, and training that runs the exact integer
arithmetic the checker will use.

One position went from 25 million instantiations to 1.2 million, and
`Generate<"ROMEO:", 24>` hovered in about 2 seconds:

```text
ROMEO:
Whe the crome the love
```

It ran. It was word salad.

### 3. Better data beats a bigger model

Switching the training data from Shakespeare to TinyStories, at the same size:

```ts
type Text = Generate<"Once upon a time", 24, "greedy">;
// "Once upon a time, there was a little girl named Lily."
```

### 4. The ghost `any`

Then the first hover in an editor session worked, and a second, different
prompt returned `any`. The checker gives up once 5 million instantiations pile
up since its counter was last reset. The command line resets it for every
statement. A hover never does, except that, as a side effect, the counter is
zeroed whenever a literal type is resolved for the first time. On a cold start
the model's own lookup tables supplied those resets; once they were warm, the
next long hover died.

The fix is a little absurd: the compiled model carries a pool of otherwise
unused literal types (`Fuel`), and the engine looks one up before every
transformer block, purely to reset the counter.

### 5. Subwords: more story per position

A hover's cost is set by the number of positions, not characters. A
128-character context scored 3% better and broke the editor session; 256 was
worse on both counts. So instead of more positions, each one carries more
text: a 256-piece vocabulary where ` little`, ` named` and ` friends` are one
token each, about 2.2 characters per position. The tokenizer is a trie walk,
in types like everything else.

### 6. The second ghost: a cache fixed by deleting `readonly`

With hovers allowed to take a minute, the model could grow to 8 layers and 128
dimensions. Then hovers 1 to 3 in a session were fine, hover 4 returned
`never` after a minute, and hover 5 returned `never` instantly.

Every time the checker compares two structured types it caches the answer.
That cache is never emptied, and once it holds 16 million entries every
comparison fails. The entries came from one word. Loops over tuples were
written `X extends readonly [infer H, ...infer R]`; every tuple in the engine
is mutable, and a mutable tuple matched against a readonly pattern is a
different type, compared element by element and cached. That is one entry per
step of every loop over a vector, millions per hover. Without `readonly` the
match is the same type, accepted by identity, with no cache entry. Five
full-context hovers in one session now pass.

### 7. What did not work

Distilling from a 33M-parameter teacher gained nothing measurable. Training
the integer model for twice as long made it worse (0.79 nats per character
against 0.69). A 512-piece vocabulary was too much for a small model. The
numbers for all of these are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### Where it stands

The character model that used to be the default wrote
`Once upon a time, there was a proble whise. The little girl was s`. The
current one writes the stories in [Results](#results). It is not GPT-2 yet:
that would be roughly 50× the matrix work per position, a 50k-row LM head and
a 1,024-token context, which means hours per token and more memory than the
checker's caches can hold. Getting closer would take a recurrent attention
variant, wider lookup tables and probably splitting the model across files.

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

### A GPT-2-shaped model with no multiplications

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

More in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): every model trained, the
longer-context runs, the inference papers this borrows from and how the cost
scales.

## Credits

- [GPT-2 in CSS](https://shubhampatil.dev/blog/gpt2-css) by Shubham Patil, the inspiration.
- [Doom in TypeScript types](https://socket.dev/blog/typescript-types-running-doom) by Dimitri Mitropoulos, for raising the bar on what a type checker should be asked to do.
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023) for the data.
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) and [BitNet](https://arxiv.org/abs/2310.11453) for ternary weights.
- [T-MAC](https://arxiv.org/abs/2407.00088) and [bitnet.cpp](https://arxiv.org/abs/2502.11880) for table-lookup inference.
- [I-BERT](https://arxiv.org/abs/2101.01321) and [Jacob et al. 2018](https://arxiv.org/abs/1712.05877) for integer-only inference.
- [tsrs](https://github.com/oxc-project/tsrs), the checker everything runs in.
