# Testing

> Last updated: 2026-08-25

## Current State

**No automated test suite.** There are zero `src/test` directories, no JUnit/TestNG dependencies, no CI workflow tracked (`.github/` is git-ignored), and no coverage tooling.

## What exists instead

Manual smoke-test drivers in `examples/` — plain classes with `main()` that print results to console:

| File | Scope | Size |
|------|-------|------|
| [examples/core/TestCore.java](../../examples/core/TestCore.java) | NDArray + linalg basics (det, inv, mean, std) | 32 lines |
| [examples/pandas/TestPandas.java](../../examples/pandas/TestPandas.java) | DataFrame/Series ops | 70 lines |
| `examples/plot/TestMatplotlib.java`, `TestSeaborn.java`, `TestConfusionMatrix.java` | plotting output | ~40 lines each |
| [examples/TestComprehensive.java](../../examples/TestComprehensive.java) | cross-module end-to-end exercise | 686 lines |
| `examples/NumJaClientExample.java` | downstream-consumer usage | — |

Run via [scripts/run_example.ps1](../../scripts/run_example.ps1) or `scripts/nj.ps1`. Verification is human eyeballing of console output / generated PNGs.

## Implications

- MODULES_INVENTORY.md claims "thoroughly tested" but this is manual/example-based only
- Regression safety relies on the comprehensive example not crashing
- If GSD TDD mode stays on, phases introducing logic changes will need to bootstrap a real test setup (plain `main()`-assert drivers or JUnit added to the build script — no Maven present, so a jar-on-classpath approach fits the existing build)

## Build verification

[scripts/build_check.ps1](../../scripts/build_check.ps1) — post-build sanity check script (closest thing to CI).
