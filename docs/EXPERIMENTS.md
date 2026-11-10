# Experiments

The lab notebook behind the [README](../README.md): every model that was
trained, what was measured, and the ideas that were dropped.

All runs use the same integer architecture. Losses are integer validation
losses in nats per character, so character and subword models compare
directly. Hover times are with `TSRS_FAST=1`.

## Models

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

| Layers × dims | Params | Training                       | Val loss | Hover    | Notes                                    |
| ------------- | ------ | ------------------------------ | -------- | -------- | ---------------------------------------- |
| 8 × 64        | 435k   | 600-step probe, 200 MB         | 1.57     | –        | Had barely started learning              |
| 4 × 128       | 865k   | 600-step probe, 200 MB         | 1.16     | 24–42 s  |                                          |
| 8 × 128       | 1.66M  | 600-step probe, 200 MB         | 1.13     | ~45 s    | Picked for the long run                  |
| **8 × 128**   | 1.66M  | 3k float + 15k integer, 600 MB | **0.71** | **36 s** | **Default.** See [Results](../README.md) |

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
  at 20k left the integer phase stuck near its starting loss. Longer integer
  phases did worse: the loss rises while the learning rate is still high, and
  30k decaying steps ended at 0.79 nats/char instead of 0.69 (see
  [Training runs](#training-runs-for-the-8--128-model)).
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
the 5M budget (see [Making it hoverable](../README.md#making-it-hoverable)): at late
positions a single position of that model costs more than 5M counted
instantiations. With a budget reset before every block it now runs to the full
64-character context, also on repeated hovers. A 16-layer, 32-dimension model
(untrained, built to bisect the limit) does too.

## Training runs for the 8 × 128 model

Quantization-aware training switches from float to integer arithmetic once,
and that switch freezes the activation scales. Four long runs:

- **Switch at step 3,000, then 37,000 integer steps.** The learning rate stayed
  high for too long: the loss bottomed out at step 12,000 and then rose
  steadily (1.742 to 1.866 nats per token by step 16,000). Stopped.
- **Switch at step 20,000.** The float model got to 1.48 nats per token, the
  best yet, but the integer model calibrated from it never trained: 5.04 nats
  per token after 6,000 integer steps, barely better than guessing uniformly
  among 256 pieces (5.55). Stopped.
- **Switch at step 3,000, then 15,000 integer steps decaying to zero.** The
  default. 0.69 nats per character in training's own evaluation, 0.71 on the
  reference's fixed validation windows.
- **The same, with 30,000 integer steps.** The loss bottomed out at 1.73 nats
  per token around step 12,000, climbed to 1.85 by step 25,000, and only
  partly recovered as the learning rate reached zero (0.79 nats per character
  at the end). The likely culprit, a guess not tested, is the frozen scales:
  calibrated once at the switch, they fit the weights less and less as those
  keep moving.

## Ideas that were measured and dropped

| Idea                                                            | Result                                                                                                                 |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Distill from TinyStories-33M (a 33M-parameter model as teacher) | 1.1088 vs 1.1096–1.1105 nats/char without it: within noise. Building the teacher's targets took 97 minutes and 2.8 GB. |
| A 128-character context                                         | 3% better loss, but the third hover in a session returned `never`                                                      |
| A 256-character context                                         | 43% worse loss, and hovers failed even in a fresh session                                                              |
| A 512-piece vocabulary at 64 dimensions                         | 1.29 nats/char vs 1.15 for 256 pieces: too much vocabulary for a model that small                                      |

## Longer context on Colab

The 4-layer, 64-dimension model was also trained with 128- and 256-character
contexts on a Colab Tesla T4, using the same 14,000-step recipe and seed 1337.
Both runs completed; their checkpoints and full training logs are saved here.

| Context | Loss on shared targets (nats/char) | Checkpoint                                           |
| ------- | ---------------------------------- | ---------------------------------------------------- |
| 64      | 1.1064                             | Previous default (`git show 772a00c:model/qat.json`) |
| 128     | 1.0732                             | [Experimental](../model/context128.json)             |
| 256     | 1.5824                             | [Experimental](../model/context256.json)             |

`scripts/evaluate-context.ts` scores the same 4,096 target characters for all
three models: the last 64 predictions in each of 64 fixed validation windows.
Each model receives its own context length. This avoids comparing different
random validation batches. The 128-context checkpoint scores about 3% better;
the 256-context checkpoint scores about 43% worse. These are single training
runs on validation data, not a separate test set or evidence that context alone
caused the difference. [Evaluation outputs](../experiments/context-128.json) include
sampled and greedy text, checkpoint hashes and a hash of the normalized validation
text; [training logs](../experiments/context-training.json) include the training-script
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
cache limit found later (see [Making it hoverable](../README.md#making-it-hoverable)); it
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
outputs are [64](../experiments/context-64.json), [128](../experiments/context-128.json)
and [256](../experiments/context-256.json).

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
