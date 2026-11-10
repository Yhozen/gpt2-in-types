// Compare checkpoints on the same 4,096 validation targets. Models receive their
// own context length; only the last 64 predictions of each of 64 windows count.
// Compile the selected QAT checkpoint first, then run with the same QAT path.
// Supports the 64/128/256-character context experiment; this is validation data,
// not a separate test split, and one run does not establish statistical significance.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import {
  forward,
  layerNorm,
  matvec,
  decode,
  encode,
  generate,
  type Program,
} from "../src/reference/int.ts";

const root = new URL("../", import.meta.url);
const read = (path: string) => readFileSync(new URL(path, root), "utf8");
const program = JSON.parse(read(process.env.PROGRAM ?? "model/program.json")) as Program;
const checkpoint = read(process.env.QAT ?? "model/qat.json");
const qat = JSON.parse(checkpoint);
if (![64, 128, 256].includes(program.config.block_size))
  throw new Error("expected context 64, 128, or 256");
if (
  JSON.stringify(program.config) !== JSON.stringify(qat.config) ||
  JSON.stringify(program.vocab) !== JSON.stringify(qat.vocab) ||
  JSON.stringify(program.head.w) !== JSON.stringify(qat.head.w)
)
  throw new Error("compile the selected QAT checkpoint first");
let text = read("model/tinystories-valid.txt");
for (const [a, b] of [
  ["<|endoftext|>", "\n"],
  ["“", '"'],
  ["”", '"'],
  ["’", "'"],
  ["‘", "'"],
  ["–", "-"],
  ["—", "-"],
  ["…", "..."],
])
  text = text.split(a!).join(b!);
text = text.replace(/[^a-zA-Z .,!?'"\-:;\n]/g, "");
const val = text.slice(Math.floor(text.length * 0.9));
const windows = 64,
  scored = 64,
  context = program.config.block_size;
let loss = 0;
const started = performance.now();
for (let w = 0; w < windows; w++) {
  // Same endpoints and final 64 prediction targets for every model.
  const end = 257 + Math.floor((w * (val.length - 258)) / (windows - 1));
  const ids = encode(program, val.slice(end - context - 1, end));
  const cache = program.layers.map(() => ({ k: [] as number[][][], v: [] as number[][][] }));
  for (let pos = 0; pos < context; pos++) {
    const x = forward(program, ids[pos]!, pos, cache);
    if (pos < context - scored) continue;
    const logits = matvec(program.head, layerNorm(x, program)).map(
      (v) => (v * qat.head.gamma) / 32,
    );
    const max = Math.max(...logits);
    loss +=
      max + Math.log(logits.reduce((s, l) => s + Math.exp(l - max), 0)) - logits[ids[pos + 1]!]!;
  }
}
const prompts = ["Once upon a time", "Tom and Lily went", "The little dog"];
const samples = prompts.flatMap((prompt) =>
  (["greedy", "sample"] as const).map((mode) => ({
    prompt,
    mode,
    text: decode(program, generate(program, encode(program, prompt), context, mode)),
  })),
);
const result = {
  checkpointSha256: createHash("sha256").update(checkpoint).digest("hex"),
  validationSha256: createHash("sha256").update(val).digest("hex"),
  config: program.config,
  context,
  windows,
  targets: windows * scored,
  loss: loss / (windows * scored),
  seconds: (performance.now() - started) / 1000,
  samples,
};

console.log(JSON.stringify(result, null, 2));
