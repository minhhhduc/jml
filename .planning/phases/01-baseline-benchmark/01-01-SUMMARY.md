# 01-01 SUMMARY: Restore Dev Environment & Baseline Benchmark

**Plan:** 01-01 (Wave 1)
**Type:** execute
**Branch:** `dev`
**Completed:** 2026-08-26

---

## What shipped

### Task 1 — Verify `mvn clean test` on dev branch

- **Status:** ✅ Done — **no source changes required**
- **Environment:**
  - JDK: Eclipse Adoptium 17.0.18 (hotspot), at `C:\Program Files\Eclipse Adoptium\jdk-17.0.3.9-hotspot\bin` (PATH default)
  - Maven: Apache Maven 3.9.15, at `C:\Users\Admin\.maven\maven-3.9.15\bin` (**NOT on PATH by default — must add explicitly**)
- **Build result:** `mvn clean test` → exit code **0**, **BUILD SUCCESS**
- **Test classes run (per module):**
  | Module | Test class | Tests | Time |
  |---|---|---|---|
  | numja | `numja.core.NDArrayTest` | 1 / 0F / 0E / 0S | 0.103s |
  | numja | `numja.linalg.LinAlgTest` | 1 / 0F / 0E / 0S | 0.001s |
  | numja | `numja.tests.NumJaJUnitTest` | 1 / 0F / 0E / 0S | 0.018s |
  | matplotlib | `matplotlib.MatplotlibTest` | 1 / 0F / 0E / 0S | 0.964s |
  | pandas | `pandas.DataFrameTest` | 1 / 0F / 0E / 0S | 0.094s |
  | seaborn | `seaborn.SeabornTest` | 2 / 0F / 0E / 0S | 0.888s |
  | sklearn | `sklearn.TestSklearnJUnit` | 1 / 0F / 0E / 0S | 0.266s |

  Total: 8 tests, 0 failures, 0 errors, 0 skipped. All 4 test classes from PLAN.md (NDArrayTest, LinAlgTest, NumJaJUnitTest wraps NumJaTest, TestSklearnJUnit) accounted for.

- **Stderr noise noted:** `numja.tests.NumJaTest.testBasicOperations` prints `Shape mismatch: [5] vs [1]` to stderr but is a **deliberate negative-case print** (the JUnit wrapper `NumJaJUnitTest.runAll` catches it). No code change needed per plan constraint "KHÔNG sửa code nguồn trong task này".

### Task 2 — Research gaps: TornadoVM / Vector API / EJML-mt

- **Status:** ✅ Done — see [VERSIONS.md](VERSIONS.md)
- **Deliverable:** `.planning/phases/01-baseline-benchmark/VERSIONS.md`
- **Key facts:**
  - TornadoVM 5.2.0 (dual-track `-jdk21` / `-jdk25`), released 23 Jul 2026 — GPU backends: NVIDIA CUDA, AMD OpenCL, Apple Metal, Intel Level Zero
  - Java Vector API: still preview/incubator on JDK 24 (JEP 460) and JDK 25; GA target JDK 26 (Mar 2026) — implication: NumJa Phase 2 cần nâng `--release` lên 21 cho module numja (multi-release JAR alternative)
  - EJML 0.45.0 (15 May 2026), `ejml-all` artifact — multithreaded variant `ejml-mt-*` không confirm được; EJML core đã có concurrency nội bộ
- **Confidence labels:** High cho cả 3 mục

---

## Deviations

| Plan said | Did | Why |
|---|---|---|
| Use `phase` branching strategy — `git checkout -b phase/01-baseline-benchmark` from `origin/main` | Stayed on `dev` | `main` ignores `/modules/` (gitignore rule); source chỉ tồn tại trên `dev` + `revent-backup`. Plan checkpoint "switch to dev" already satisfied before this session began. STATE.md documents the deviation. |
| `mvn` available on PATH | Not on PATH by default | Maven installed at `C:\Users\Admin\.maven\maven-3.9.15\` but PATH not extended. Documented here so subsequent sessions prepend `$env:Path = '...\bin;' + $env:Path` before running `mvn`. |
| Use system default JDK | Java 17 used; Java 25 also available | Both work; 17 picked because Maven wrapper default. JDK 25 needed only when activating Vector API (Phase 2+). |

---

## Files modified

- **NEW:** `.planning/phases/01-baseline-benchmark/VERSIONS.md` (104 lines, committed in `59e10f4`)
- **NEW:** `.planning/phases/01-baseline-benchmark/01-01-SUMMARY.md` (this file)
- **MODIFIED:** `.planning/STATE.md` (current focus & next command updates — committed next)

No source code, pom.xml, or .gitignore changes in this plan.

---

## Verification checklist

- [x] `git branch --show-current` returns `dev`
- [x] `mvn clean test` exit 0
- [x] All 4 required test classes run: NDArrayTest, LinAlgTest, NumJaJUnitTest (wraps NumJaTest), TestSklearnJUnit
- [x] `.planning/phases/01-baseline-benchmark/VERSIONS.md` exists with 3 research gaps closed
- [x] Each research item has URL source + confidence label

---

## What's next (Wave 2)

Plan **01-02** — JMH benchmark harness + baseline numbers:
- Create `bench/` Maven module with jmh-core / jmh-generator-annprocess
- Write CoreBench / PandasBench / SklearnBench with @Benchmark + Blackhole
- Produce `BASELINE.md` with ≥8 op scores + machine metadata

Plan **01-03** — Golden-value test infrastructure:
- Write `scripts/golden/generate_golden.py` (numpy + seed=42) producing ≥4 JSON files
- Write `GoldenReferenceTest` with `@Tag("golden")` soft-fail mode
- Document regenerate procedure in `GOLDEN.md`

Both plans depend on `01-01-SUMMARY.md` (this file). Wave 2 runs them in parallel.
