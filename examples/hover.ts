// Open this file in VS Code or Cursor (with the tsrs settings in .vscode/) and
// hover the types below: the type checker runs the model and the tooltip
// shows what it wrote. No JavaScript runs.
//
// Each full story is 64 positions of an 8-layer transformer: about a minute
// per hover on an Apple Silicon Mac, with ~8 GB for the language server.
// Not part of `vp run examples`: the CLI caps a single statement at ~5M type
// instantiations, which one alias exceeds (see examples/story.ts and
// "Making it hoverable" in the README).
import type { Generate, Tokenize } from "../src/engine/index.ts";

/** Instant: the prompt split into the model's 256 vocabulary pieces. */
export type Tokens = Tokenize<"Once upon a time">;

/** A sampled story (temperature 0.6, noise offset 0). */
export type Text = Generate<"Once upon a time", 64>;

/** Argmax decoding instead of sampling. */
export type Greedy = Generate<"One day, a girl named Mia found", 64, "greedy">;

/** Change the fourth argument (0..63) for another reproducible sample. */
export type AnotherStory = Generate<"The little dog", 64, "sample", 3>;
