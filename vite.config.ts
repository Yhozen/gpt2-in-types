import { defineConfig } from "vite-plus";

/** Machine-generated sources: large literal tables, not worth formatting or linting. */
const generated = [
  "src/model/program.ts",
  "tests/types/engine.gen.ts",
  "examples/*.ts",
  "model/**",
];

export default defineConfig({
  staged: {
    "*": "vp check --fix",
    // Type-check the whole project (not just staged files) with tsrs.
    "*.{ts,mts,cts,tsx}": () => "tsrs --noEmit --fast",
  },
  lint: {
    ignorePatterns: generated,
    options: {
      // Keep type-aware lint rules, but leave type checking to tsrs
      // (`vp run typecheck`) so there's a single source of type errors.
      typeAware: true,
      typeCheck: false,
    },
  },
  fmt: {
    ignorePatterns: generated,
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
