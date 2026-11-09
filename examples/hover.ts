// Open this file in VS Code or Cursor (with the tsrs settings in .vscode/) and
// hover `Text`: the type checker runs GPT-2 and the tooltip shows its output.
// Not part of `vp run examples`: the CLI caps a single statement at ~5M type
// instantiations, which this one alias exceeds (see examples/story.ts).
import type { Generate } from "../src/engine/index.ts";

export type Text = Generate<"Once upon a time", 24>;

/** Argmax decoding instead of seeded sampling. */
export type Greedy = Generate<"Once upon a time", 24, "greedy">;
