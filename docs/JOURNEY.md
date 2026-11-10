# How a type checker learned to tell stories

This is the story of `gpt2-in-types`, a GPT-2-style language model that runs
inside the TypeScript type checker. You hover a type in your editor, and the
tooltip is a short story the model just wrote. No JavaScript runs: tokenizing,
embeddings, attention, the MLP, sampling and the decode loop are all type
aliases, evaluated by [tsrs](https://github.com/oxc-project/tsrs) (the Rust
port of the TypeScript 7 checker) when the editor asks what the type is.

It took two days, several dead ends, and one bug that could only be explained
by reading the checker's source. This is what happened, in order, with the
numbers.

## 0. The dare

[GPT-2 in CSS](https://shubhampatil.dev/blog/gpt2-css) ran the real GPT-2
small in a browser with no JavaScript. TypeScript's type system is famously
[Turing complete](https://github.com/microsoft/TypeScript/issues/14833), so the
question was obvious: can a type checker run a language model too?

Turing complete says nothing about _fast_. The type checker has no numbers you
can add, no multiplication, no floats, no `exp`. It has literal types,
conditional types, template literals, mapped types and recursion, and it gives
up after a few million steps.

## 1. Version 1: arithmetic out of digit strings (it worked, in the CLI)

The first version (commit `1121832`) did the honest thing: numbers as strings
of decimal digits, with addition and multiplication implemented digit by digit
in types, like long arithmetic on paper. A 4-layer, 32-dimension character
model trained on Tiny Shakespeare ran exactly, matched a BigInt reference
digit for digit, and wrote text.

But each multiply-add cost about **700 type instantiations**, and one position
of the model cost about **25 million**. The CLI aborts any statement after 5
million (`TS2589: Type instantiation is excessively deep and possibly
infinite`), so generation was split into a generated "clock" file, one step per
statement. In the editor, hovering the result showed:

```ts
type Text = any;
```

A model you cannot hover is not much of a demo.

## 2. Measure the machine first

So the second attempt started by measuring what the checker is actually fast
at:

| Operation                                            | Cost                                         |
| ---------------------------------------------------- | -------------------------------------------- |
| Template literal concat `` `${A}${B}` ``, any length | ~3 instantiations                            |
| Prefix match `` A extends `${B}${infer R}` ``        | ~3–10, the string compare is native          |
| Property lookup in a large object type               | ~1–2, a native hash lookup                   |
| Tuple spread / subtraction                           | ~1–2 _per element_, capped at 9,999 elements |
| Decimal digit arithmetic (version 1)                 | ~700 per multiply-add                        |

Two operations are effectively free regardless of size: **concatenating
strings** and **checking whether one string starts with another**. If a number
is a string of that many `1`s (unary), addition is concatenation, subtraction
and comparison are prefix matches, and none of them cost more for bigger
numbers.

Multiplication is still impossible. So the model must not multiply.

## 3. Version 2: a model that only adds (it hovers)

The rest of the design follows from that (commit `d030700`):

- **Ternary weights.** Every matrix is trained with weights in {−1, 0, +1}
  ([BitNet b1.58](https://arxiv.org/abs/2402.17764)). A mat-vec row becomes
  "concatenate the inputs with weight +1, subtract the ones with weight −1":
  one template literal per row, compiled ahead of time into a type alias.
- **Everything nonlinear is a lookup.** Requantization, GELU, `exp` and the
  softmax normalizer are monotone functions of an integer, compiled into
  decision trees of thresholds; walking one is a few prefix matches
  ([T-MAC](https://arxiv.org/abs/2407.00088) does table lookups for the same
  reason on CPUs).
- **Integer LayerNorm and softmax** with static scales, in the style of
  [I-BERT](https://arxiv.org/abs/2101.01321). The square root comes from a
  3-significant-digit table, the division from binary long division against
  doublings of σ.
- **Quantization-aware training.** The PyTorch trainer runs the exact integer
  forward pass with straight-through gradients, so the network learns to work
  with the arithmetic it will get.

One position now cost about 1.2 million instantiations instead of 25 million,
and `Generate<"ROMEO:", 24>` hovered in about 2 seconds. The output:

```text
ROMEO:
Whe the crome the love
```

It ran. It was word salad.

## 4. Better data beats a bigger model

Switching the training data from Shakespeare to
[TinyStories](https://arxiv.org/abs/2305.07759) (stories written in the
vocabulary of a three-year-old, designed so tiny models can learn coherent
English) changed everything at the same size (commit `68eabb3`):

```ts
type Text = Generate<"Once upon a time", 24, "greedy">;
// "Once upon a time, there was a little girl named Lily."
```

Two other findings from that round:

- tsrs has an opt-in performance mode (`TSRS_FAST=1`) that the language server
  does not enable by default. It made hovers about **3× faster**, so the repo
  ships a two-line wrapper the editor uses.
- Compiling the lookup trees into nested conditional aliases instead of data
  walked by a generic type cut another ~30%.

## 5. The ghost `any`

Then a strange bug: the _first_ hover in an editor session worked, a
_second_ different prompt returned `any`, and an 8-layer model failed about
two thirds of the way through its context, every time.

It was not the depth limit, and the total amount of work did not predict it
either: 64 generation steps written out literally in one type worked, while
the same 64 steps through a generic helper type failed. Reading the checker's
source explained why. The checker gives
up once **5 million instantiations** pile up since its counter was last reset.
The CLI resets it for every statement. A hover never does, _except_ that, as a
side effect, the counter is zeroed whenever a literal type node is resolved
for the first time. On a cold start the model's own lookup tables provide
those resets, so the first hover survives; once they are warm, nothing resets
the counter and the next long hover dies.

The fix is a little absurd: the compiled program carries a pool of unused
literal types (`Fuel`), and the engine looks one up before every transformer
block (`Refuel`), so every block starts with a fresh budget (commits
`c71554e`, `43ea16f`). Each block of each position draws its own node, keyed by
a hash of the tokens so far, so a later hover with a different story finds
fresh ones. Since then, repeated full-context hovers in one session work, and
so does the 8-layer model.

## 6. What did not work

Not every idea paid off. These were measured and dropped:

| Idea                                                            | Result                                                                                                                 |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Distill from TinyStories-33M (a 33M-parameter model as teacher) | 1.1088 vs 1.1096–1.1105 nats/char without it: within noise. Building the teacher's targets took 97 minutes and 2.8 GB. |
| A 128-character context                                         | 3% better loss, but the third hover in a session returned `never`                                                      |
| A 256-character context                                         | 43% worse loss, and hovers failed even in a fresh session                                                              |
| A 512-piece vocabulary at 64 dimensions                         | 1.29 nats/char vs 1.15 for 256 pieces: too much vocabulary for a model that small                                      |

The longer contexts were trained on a Colab T4; the distillation needed a
cross-tokenizer conversion of the teacher's predictions into next-character
probabilities. The 128-character model's `never` on the third hover has the
same symptom as the cache limit in section 8, found later; it was not retested.

## 7. Subwords: more story per position

The cost of a hover is set by the number of _positions_, not characters. So
instead of more positions, make each one carry more text: a 256-piece
subword vocabulary (common words like ` little`, ` named`, ` friends` are one
token) gives about 2.2 characters per position. A 64-position context now
covers about 140 characters, and a 24-token hover shows about three times as
much text, faster:

```text
Once upon a time, there was a little girl named Lily. One day, a little girl named Kitt
```

Tokenizing in types is a trie walk with greedy longest match, a rule simple
enough that the trainer, the reference and the types implement it identically.
The vocabulary also gained an end-of-story token, so a hover can stop on its
own.

## 8. The second ghost: a cache that never forgets

With hovers allowed to take minutes, the obvious move was a bigger model. An
8-layer, 128-dimension probe (1.66M parameters, eight times the original
model) hovered a full 64-token context in about 45 seconds and matched the
reference. Then the session test failed again: hovers 1 to 3 fine, hover 4
`never` after a minute, hover 5 `never` instantly. Hover 4 alone, in a fresh
server, was fine.

`never` instead of `any` meant a different limit. This one is in the relation
checker. Every time the checker answers "is A assignable to B" for two
structured types it records the answer in a cache, and the work it allows for
the next comparison is `(16,000,000 − cache size) / 8`. The cache belongs to the
checker and is never emptied, so once it holds 16 million answers every
comparison fails, every conditional type takes its false branch, and the
engine's false branches are `never`. The CLI says so out loud
(`TS2859: Excessive complexity comparing types`); a hover just shows `never`.

Two cheap experiments confirmed it. Filling the cache with unrelated work first
made a single generation fail in the CLI; and editing the file between hovers
3 and 4 made hover 4 pass, because a new program gets a new checker.

The entries came from one character in the source. Loops over tuples were
written `X extends readonly [infer H, ...infer R]`, to accept any tuple. Every
tuple in the engine is mutable, though, and a mutable `[a, b, c]` matched
against a readonly pattern instantiates to `readonly [a, b, c]`: a different
type, compared element by element and cached. One entry per step of every
loop over a vector, millions per hover. With the mutable pattern
`X extends [infer H, ...infer R]` the match instantiates to the very same type
and the checker accepts it by identity, with no comparison and no cache entry.
After that change five full-context hovers in one session pass, and peak memory
went down.

## 9. Bigger, now that minutes are fine

Once a hover is allowed to take a minute, the model can grow. Short probes
(600 training steps each, same 256-piece vocabulary, 200 MB of TinyStories)
picked the shape:

| Layers × dims | Parameters | nats/char after 600 steps | Full-context hover |
| ------------- | ---------- | ------------------------- | ------------------ |
| 8 × 64        | 435k       | 1.5705                    | –                  |
| 4 × 128       | 865k       | 1.1582                    | 24–42 s            |
| 8 × 128       | 1.66M      | 1.1341                    | ~45 s              |

At 600 steps the 64-dimension model had barely started learning, the
128-dimension ones were already close to the old default, and 8 × 128 was
best. For comparison, the 4 × 64 character model that had been the default
scores 1.1414 after its full 14,000 steps.

The long runs were a lesson in quantization-aware training, where the switch
from float to integer arithmetic freezes the activation scales:

- **Switch at step 3,000, then 37,000 integer steps.** The learning rate stayed
  high for too long: the loss bottomed out at step 12,000 and then rose
  steadily (1.742 to 1.866 nats per token by step 16,000). Stopped.
- **Switch at step 20,000.** The float model got to 1.48 nats per token, the
  best yet, but the integer model calibrated from it never trained: 5.04 nats
  per token after 6,000 integer steps, barely better than guessing uniformly
  among 256 pieces (5.55). Stopped.
- **Switch at step 3,000, then 15,000 integer steps decaying to zero.** The
  recipe that had worked at the small size, given more steps and data.
- **The same, with 30,000 integer steps.** More steps should help, so this ran
  next. It did not: the loss bottomed out at 1.73 nats per token around step
  12,000, climbed to 1.85 by step 25,000, and only partly recovered as the
  learning rate reached zero (0.79 nats per character at the end, against 0.69
  for the 15,000-step run). The likely culprit, a guess not tested, is the
  frozen scales: calibrated once at the switch, they fit the weights less and
  less as those keep moving. In this setup the integer phase wants to be short.

The third run is the default now. Its integer model scores 0.71 nats per
character on the fixed validation windows (the reference's own number), against
1.14 for the 4 × 64 character model: in perplexity per character, 2.0 instead
of 3.1. The integer reference matches PyTorch at all 16,384 checked positions,
and every hover matches the reference token for token.

What that buys, hovered in one editor session:

```text
Tokenize<"Once upon a time">                  0.7 s
["Once", " upon", " a", " time"]

Generate<"Once upon a time", 64>              36 s
Once upon a time, there was a little girl named Lily. She had a big, red ball
that she loved very much. One day, she found a big box in the park. Amy was
very happy and wanted to play with the ball.

Generate<"One day, a girl named Mia found", 64, "greedy">     34 s
One day, a girl named Mia found a big box. She was very happy. Mia wanted to
show her mom. She asked her mom, "Mom, can I have the box?" Her mom said, "Yes, Mia

Generate<"The little dog", 64, "sample", 3>   18 s
The little dog was so happy and they all played together in the rain. They had
so much fun together and became good friends.
```

The last one stops by itself: the model chose the end-of-story token. For
comparison, the character model's full-context hover, from the same prompt:

```text
Once upon a time, there was a proble whise. The little girl was s
```

It still makes mistakes a reader will catch (Lily becomes Amy halfway through),
and a session takes ~8 GB, but it is a story, written by a type checker.

## How it compares with GPT-2 in CSS

GPT-2 in CSS runs OpenAI's real GPT-2 small, which we could not. The
difference is what each engine can do natively:

|                 | GPT-2 in CSS                                                                                                              | GPT-2 in types                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Native multiply | yes: `calc()` floats, and the weights live in variable fonts whose glyph widths the font engine computes as weighted sums | none: addition is string concatenation                     |
| Weights         | real GPT-2 small, 9-bit integers with a per-row scale                                                                     | ternary, trained from scratch                              |
| Speed           | 15–45 s per token (32 tokens take ~15 minutes)                                                                            | ~0.5–1 s per token (a 64-token story in one 35–75 s hover) |
| Context         | 36 tokens                                                                                                                 | 64 tokens (~140 characters)                                |

Running the real GPT-2 small here would mean about 124 million real
multiply-adds per token. With version 1's digit arithmetic that is about 87
billion instantiations, roughly four hours per token at the speed tsrs manages
on this workload. Ternary weights turn every multiply into "add, skip or
subtract", which is the one thing a type checker does natively, at the price of
training our own model.

## Credits and sources

- [GPT-2 in CSS](https://shubhampatil.dev/blog/gpt2-css) by Shubham Patil, the inspiration.
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023) for the data.
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) and [BitNet](https://arxiv.org/abs/2310.11453) for ternary weights.
- [T-MAC](https://arxiv.org/abs/2407.00088) and [bitnet.cpp](https://arxiv.org/abs/2502.11880) for table-lookup inference.
- [I-BERT](https://arxiv.org/abs/2101.01321) and [Jacob et al. 2018](https://arxiv.org/abs/1712.05877) for integer-only inference.
- [tsrs](https://github.com/oxc-project/tsrs), the checker everything runs in.
