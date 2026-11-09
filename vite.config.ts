import { defineConfig } from "vite-plus";

export default defineConfig({
  staged: {
    "*": "vp check --fix",
    // Type-check the whole project (not just staged files) with tsrs.
    "*.{ts,mts,cts,tsx}": () => "tsrs --noEmit --fast",
  },
  pack: {
    deps: {
      // tsdown <0.23 compatibility: resolve external dependency subpaths.
      // Remove to preserve subpath imports as written (the new default).
      // https://tsdown.dev/options/dependencies#deps-resolvedepsubpath
      resolveDepSubpath: true,
    },
    dts: {
      generator: "tsgo",
    },
    exports: true,
  },
  lint: {
    options: {
      // Keep type-aware lint rules, but leave type checking to tsrs
      // (`vp run typecheck`) so there's a single source of type errors.
      typeAware: true,
      typeCheck: false,
    },
  },
  fmt: {},
});
