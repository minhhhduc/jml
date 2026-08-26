---
phase: 01-baseline-benchmark
verified: 2026-08-26T20:30:00+07:00
status: passed
score: 8/8 success criteria met
gaps: []
---

# Phase 1: Restore Dev Environment & Baseline Benchmark — Verification Report

**Phase Goal:** Source từ nhánh dev chạy được, có benchmark harness đo baseline mọi core op — con số trước khi tối ưu.
**Verified:** 2026-08-26
**Status:** PASSED
**Overall Verdict:** **GO** — Phase 1 can be closed; proceed to Phase 2.

## Success Criteria Verdicts

### SC1 — `mvn test` pass trên nhánh dev — **MET**

- Ran `mvn clean test` (Maven 3.9.15) from repo root on branch `dev`.
- Exit code 0, `BUILD SUCCESS`, Reactor Summary: all 7 modules SUCCESS (numja-core, matplotlib, pandas-wrapper, seaborn, sklearn, bench).
- sklearn module: Tests run 6, Failures 0, Errors 0, Skipped 1 (the skipped one is the reference-only softmax test).

### SC2 — JMH harness chạy bằng một lệnh, đo core ops — **MET**

- `mvn -pl bench -am package -DskipTests` exit 0; `bench/target/benchmarks.jar` (5.3 MB shaded jar) built.
- Live smoke run from the jar: `java -jar bench/target/benchmarks.jar "CoreBench.matmul_NxN" -p n=256 -f 0 -wi 1 -i 1` printed `29.276 ms/op` — matching the recorded baseline (29.27).
- Coverage confirmed in source: CoreBench (matmul 256/1024, add/mul elementwise 10^6/10^7, sum/mean reduce), PandasBench (read_csv california, groupby titanic), SklearnBench (LinearRegression fit, KMeans iris) — exactly the ops BENCH-01 requires.
- Datasets referenced exist at `dist/datasets/`.

### SC3 — Baseline results ghi lại và commit — **MET**

- `.planning/phases/01-baseline-benchmark/BASELINE.md`: 12 benchmark results across Core/Pandas/Sklearn tables (requirement ≥8), plus machine metadata table (CPU i7-1255U, RAM 15.7 GB, JDK Temurin 25.0.3, OS Windows 11, Maven 3.9.15) and reproduce commands.
- Committed: last touched by commit `59343b8`; `git status --short` clean.

### SC4 — TornadoVM / Vector API / EJML version check — **MET**

- `.planning/phases/01-baseline-benchmark/VERSIONS.md` covers all three:
  - TornadoVM 5.2.0 (jdk21/jdk25 dual-track), GPU backend matrix, URLs to GitHub releases, confidence High.
  - Vector API: JEP 460 preview on JDK 24/25, GA target JDK 26, URLs, confidence High, plus concrete NumJa impact (`--release 11` → 21 decision for Phase 2).
  - EJML 0.45.0, artifact guidance, honest Medium confidence on the absence of a dedicated `ejml-mt` artifact.
- Remaining open sub-checks in VERSIONS.md are correctly deferred to Phases 2/3/5 where they are needed.

### Plan-level acceptance criteria (spot-checked)

| Check | Result |
|---|---|
| Golden JSON files in `bench/src/test/resources/golden/` | 4 files: matmul_256, sum_mean, softmax_extreme, linear_regression_iris |
| Each contains `tolerance_rel` | Verified by grep: 1e-12, 1e-13, 1e-12, 1e-09 respectively |
| `GoldenReferenceTest` exists with golden marker | Yes — `@Category(GoldenReferenceTest.Golden.class)` (JUnit4 category; REVIEW.md CR-02 doc fix applied) |
| GoldenReferenceTest runs green | `mvn -pl modules/sklearn -am test -Dtest=GoldenReferenceTest ...` exit 0: 5 tests run, 0 failures, 1 skipped |
| Durable accuracy report | `modules/sklearn/target/golden-report.txt` written after run; every line PASS within tolerance (e.g. matmul maxErr 1.985e-15 vs tol 1e-12, sum err 1.211e-14 vs tol 1e-13) |
| `bench/GOLDEN.md` documents regeneration | Yes — regenerate command, seed/LCG determinism contract, per-op tolerance rationale table |
| Three SUMMARY files exist | 01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md all present |

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| BENCH-01 | SATISFIED | One-command JMH harness covering NDArray matmul/elementwise/reduce, DataFrame groupby/read_csv, sklearn fit/predict |
| BENCH-02 | SATISFIED | BASELINE.md with 12 recorded baselines + machine metadata, committed |
| ACC-03 (early infra, fully closes in Phase 3) | ON TRACK | Golden harness operational, all current ops PASS within tolerance; Phase 3 owns hard gates |

## Known Caveats (documented, non-blocking)

1. BASELINE.md error bars exceed scores on some rows (hybrid P/E-core CPU, only 3 iterations). The document itself flags this and mandates multi-fork re-measurement when comparing in Phase 2. Acceptable for a first mốc.
2. Softmax golden test is `[reference-only]` (validates harness math, not a NumJa op — none exists yet), hence the 1 skip. Explicitly labeled in test name, javadoc, and report.
3. Benchmarks must run from repo root (CWD-relative dataset paths) — documented in BASELINE.md and REVIEW.md MJ-03 accepted deferral.

## Gaps Blocking Phase 2

None.

---

_Verifier: Claude (gsd-verifier)_
