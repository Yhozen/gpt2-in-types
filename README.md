# gpt2-in-types

GPT-2 implemented in the TypeScript type system.

## Toolchain

- [Vite+](https://viteplus.dev/) (`vp`) — package management, formatting (Oxfmt), linting (Oxlint), testing (Vitest), and bundling (tsdown).
- [tsrs](https://github.com/oxc-project/tsrs) — Rust port of the TypeScript 7 checker, used for type checking.
- Node.js 26 (see `.node-version`), pnpm (pinned via `devEngines`).

## Development

```bash
vp install         # install dependencies (also installs git hooks)
vp run typecheck   # type check with tsrs
vp run check       # format + lint (vp check) + type check (tsrs)
vp test            # run tests
vp pack            # build the library
```

A pre-commit hook (`.vite-hooks/pre-commit`) runs `vp staged`, which applies `vp check --fix` to staged files and type checks the project with tsrs.
