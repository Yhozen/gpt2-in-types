// Regression check for editor sessions: hover several full-context generations
// one after another in ONE language-server session and compare each with the
// integer reference. Only the first hover of a session finds the model's
// tables cold; the later ones depend on the budget resets described at Refuel
// in src/engine/index.ts.
//
//   node scripts/check-hover.ts
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { decode, encode, generate, type Program } from "../src/reference/int.ts";
import { hoverAll } from "./hover.ts";

const ROOT = resolve(import.meta.dirname, "..");
const program = JSON.parse(readFileSync(resolve(ROOT, "model/program.json"), "utf8")) as Program;
const cases = [
  ["Once upon a time", "sample", 0],
  ["Tom and Lily went", "greedy", 0],
  ["The little dog", "sample", 0],
  ["Once upon a time", "sample", 1],
  ["Once upon a time", "sample", program.config.block_size - 1],
] as const;
const n = program.config.block_size; // more than fits: every case runs to the full context

const file = resolve(ROOT, ".run/check-hover.ts");
mkdirSync(resolve(ROOT, ".run"), { recursive: true });
writeFileSync(
  file,
  `import type { Generate } from "../src/engine/index.ts";\n\n${cases
    .map(
      ([prompt, mode, seed], i) =>
        `export type T${i} = Generate<${JSON.stringify(prompt)}, ${n}, "${mode}", ${seed}>;`,
    )
    .join("\n")}\n`,
);
const results = await hoverAll(
  file,
  cases.map((_, i) => `T${i}`),
);
let failed = false;
cases.forEach(([prompt, mode, seed], i) => {
  const expected = decode(program, generate(program, encode(program, prompt), n, mode, seed));
  const { text, seconds } = results[i]!;
  const ok = text === `type T${i} = ${JSON.stringify(expected)}`;
  failed ||= !ok;
  console.log(
    `${ok ? "ok  " : "FAIL"} hover ${i + 1} (${seconds.toFixed(1)}s): ${ok ? JSON.stringify(expected) : text}`,
  );
});
process.exitCode = failed ? 1 : 0;
