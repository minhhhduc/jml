---
phase: 01-baseline-benchmark
reviewed: 2026-08-26T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - bench/pom.xml
  - bench/src/main/java/bench/CoreBench.java
  - bench/src/main/java/bench/PandasBench.java
  - bench/src/main/java/bench/SklearnBench.java
  - modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java
  - scripts/golden/generate_golden.py
  - bench/GOLDEN.md
  - .planning/phases/01-baseline-benchmark/BASELINE.md
findings:
  critical: 3
  major: 5
  minor: 4
  total: 12
status: fixed
---

# Phase 1: Code Review Report

**Reviewed:** 2026-08-26
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found → **fixed (same day)** — see "Fix Log" at bottom.

## Fix Log (2026-08-26, post-review remediation)

| Finding | Resolution |
|---|---|
| CR-01 soft-fail dies with JVM | Report now flushed to `target/golden-report.txt` after every test; verified on disk after run. Hard assertion on file loading deferred to Phase 3 per plan's soft-fail mandate. |
| CR-02 `-Dgroups=golden` no-op doc | GOLDEN.md + test javadoc corrected: JUnit4 `@Category`, run via `-Dtest=GoldenReferenceTest`. Verified command works. |
| CR-03 sum/mean tol 1e-15 impossible | Generator tolerance raised to 1e-13 with rationale comment (drift ~1e-14 measured). Tests PASS. |
| MJ-01 softmax doesn't call NumJa | Test renamed `[reference-only]` in label, javadoc states explicitly it validates the harness, not NumJa. Switch when numja gains softmax op. |
| MJ-02 cartesian @Param | CoreBench rewritten with MatState/ElemState/ReduceState — each benchmark only runs its own params. Baseline re-run. |
| MJ-03 CWD-relative paths | Accepted for now: benchmarks documented as run-from-repo-root (STATE.md records the command). Fixing properly needs a dataset-path abstraction — deferred. |
| MJ-04 33MB golden JSON in git | Inputs removed from JSON entirely — regenerated from seed via Java-compatible LCG on both sides. Total golden size now ~1.2MB. Old fat files deleted. |
| MJ-05 dual file loading | Kept fallback but documented why (golden files not on sklearn test classpath); classpath-first order preserved. |
| Minor ×4 | Stale comment removed with dead method; compiler config dedup left (harmless); GOLDEN.md language cleaned; BASELINE.md mean-anomaly footnote superseded by fresh baseline v2 run. |

Post-fix verification: `mvn -pl modules/sklearn -am test -Dtest=GoldenReferenceTest` exit 0, all ops PASS within tolerance (matmul maxErr 2.0e-15, sum 1.2e-14, softmax 4.3e-16, linreg ≤3.2e-13).

## Summary

JMH benchmark classes themselves are methodologically sound (Blackhole consumption everywhere, state pre-allocated in `@Setup(Level.Trial)`, fresh model per fit invocation). The golden-value harness is where the defects concentrate: one test does not test NumJa at all, one tolerance is mathematically impossible to meet, and the documented way to run the suite does nothing. No security issues found — all file handling uses hardcoded constant paths with no user input.

## Critical Issues

### CR-01: Soft-fail harness can never fail — `assertTrue(true)` makes the entire suite decorative

**File:** `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java:109-111` (used at 135, 148, 173, 199)
**Issue:** Every test ends with `assertMechanicsOnly()`, which is `assertTrue("Golden harness ran", true)` — a tautology. All mismatches go only to `REPORT` (a private static list nobody reads) and stderr. If NumJa regresses from 1e-14 error to producing complete garbage, `mvn test` exits green. Even accepting the Phase-1 "soft-fail" intent, there is no machine-readable report artifact — the gap data dies with the JVM. This is a harness that records nothing durable and gates nothing.
**Fix:** At minimum write `REPORT` to a file (`target/golden-report.txt`) so failures persist, and add one hard assertion that the harness actually loaded all four golden files (a missing file currently surfaces only as an exception mid-suite). In Phase 3, replace the tautology with `assertTrue(maxErr <= tol || recordedInBacklog)`.

### CR-02: Documented run command `-Dgroups=golden` is a no-op

**File:** `bench/GOLDEN.md:44-48`; `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java:32`
**Issue:** GOLDEN.md instructs `mvn test -Dgroups=golden` and describes the tag as `@Tag("golden")`. The code uses JUnit 4 `@Category(GoldenReferenceTest.Golden.class)`, and neither the root pom nor `modules/sklearn/pom.xml` configures Surefire with JUnit 47.x providers or `groups`/`excludedGroups`. `-Dgroups` is a TestNG/JUnit5-style property; with plain JUnit 4 it silently does nothing. The documented command runs all tests, not the golden category. Doc and code disagree about basic operation of the deliverable.
**Fix:** Either configure surefire `<groups>` with junit47 provider, or fix GOLDEN.md to the command that actually works (run the class directly: `mvn test -Dtest=GoldenReferenceTest`) and correct the `@Tag` reference to `@Category`.

### CR-03: sum_mean tolerance 1e-15 is below the unavoidable sequential-vs-pairwise drift — permanently failing check

**File:** `scripts/golden/generate_golden.py:54-56`; `bench/GOLDEN.md:38`; verified against `NDArray.sum()` (`modules/numja/src/main/java/numja/core/NDArray.java:329-335`)
**Issue:** `NumJa.sum()` is naive sequential summation; NumPy uses pairwise summation. For 1e6 uniform doubles the expected drift is ~1e-14 relative (reproduced during this review: seq −355.3283300408214 vs numpy −355.3283300408253, rel err 1.09e-14). The chosen `tolerance_rel = 1e-15` is one order of magnitude tighter than the physics of the implementation allows, so `sum(1e6)` and `mean(1e6)` are guaranteed FAILs on every run regardless of code quality. The generator's own inline comment claims drift "rel 1e-15", which is wrong. Once the soft-fail tautology (CR-01) is fixed, these become permanent red marks attributed to a nonexistent defect.
**Fix:** Set `tolerance_rel = 1e-12` for sum_mean (still tight enough to catch real bugs), and correct the comment and GOLDEN.md table.

## Major Issues

### MJ-01: softmax "accuracy test" never invokes NumJa — compares Java reimplementation against its own reference

**File:** `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java:152-174`
**Issue:** `softmax_1000_extreme_vs_numpy()` computes softmax with a stable max-shift loop written *inside the test*, then diffs that against the NumPy golden values. It measures Java `Math.exp` vs NumPy `exp` — it never touches any `numja` API (no import of NumJa in this test path). The commit message for c040ae7 claims "matmul/softmax within tolerance", presenting a vacuous pass as evidence of NumJa accuracy. The comment admits "NumJa has no softmax op yet", but then the test should be `@Ignore`d or assert the op is absent, not report a meaningless PASS into the baseline record.
**Fix:** Mark the test `@Ignore("pending NumJa softmax op")` until the op exists, or call a NumJa-side implementation when one lands.

### MJ-02: JMH parameter cartesian runs every benchmark 2–4× with irrelevant sizes

**File:** `bench/src/main/java/bench/CoreBench.java:23-31`
**Issue:** `matmulN` and `elemN` are both `@Param` fields on one `@State` class, so JMH runs the full cartesian product: `matmul_NxN` executes 4 times (2 matmulN × 2 elemN), and `add/multiply_elementwise` execute 2 times each per matmulN value — 4 runs each, identical work. BASELINE.md:20 acknowledges this ("param kia N/A"), but the published baseline table then mixes scores from redundant duplicate runs (e.g. matmul_NxN "11.8–15.7*" is a range across meaningless duplicates). Baseline numbers other teams will compare against are obfuscated by noise from runs that measure nothing new.
**Fix:** Split into two benchmark classes (MatMulBench, ElementwiseBench) each with one `@Param`, or drop `@Param` and set sizes as constants. Re-run baseline after.

### MJ-03: Benchmark setup crashes cryptically unless invoked from repo root

**File:** `bench/src/main/java/bench/PandasBench.java:25-34`; `SklearnBench.java:29-38`
**Issue:** Dataset paths are CWD-relative (`dist/datasets/*.csv`). If `java -jar bench/target/benchmarks.jar` is launched from anywhere but the repo root, `@Setup(Level.Trial)` throws `FileNotFoundException` and JMH reports the entire benchmark set as failed with a stack trace pointing at CSV parsing — no hint about CWD. The requirement exists only in a comment (PandasBench.java:23-24) and in BASELINE.md:70, not in the harness.
**Fix:** Fail fast with a clear message in `setUp()`: check `Files.exists(CALIFORNIA)` and throw `IllegalStateException("Run from repo root; missing " + CALIFORNIA.toAbsolutePath())`.

### MJ-04: 33 MB of pretty-printed JSON committed to git

**File:** `bench/src/test/resources/golden/*.json` (sum_mean_1e6.json alone is 27.8 MB / 1,000,013 lines; matmul_256.json 5.1 MB)
**Issue:** `generate_golden.py:24` dumps with `indent=2`, inflating files ~3–4× versus compact form, and the full input arrays are stored alongside expected outputs. These are committed to git history permanently. Any future regeneration rewrites a million-line diff. Checkout size and clone time grow for every consumer of the repo for data that is deterministically regenerable from a 100-line script.
**Fix:** Dump compact (`json.dump(payload, f, separators=(",", ":"))`) and consider storing inputs as a seed recipe instead of materialized arrays where possible (all four generators are seeded; only iris.csv-derived X/y truly need materialization).

### MJ-05: Golden file loading has two divergent mechanisms, one CWD-dependent

**File:** `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java:38, 48-59`
**Issue:** `load()` tries the classpath first, then falls back to `new File("../../bench/src/test/resources/golden/...")` — a relative path from whatever directory Maven was invoked in (works only from `modules/sklearn`). Two resolution strategies means the test can silently read *different copies* of the golden data depending on invocation context (stale target/test-classes copy vs live source tree), making nondeterministic test results possible after editing a golden file without rebuilding. The fallback exists only because the JSON lives under `bench/` but the test lives under `modules/sklearn`.
**Fix:** Pick one mechanism. Simplest: keep golden resources on the sklearn test classpath (move the directory, or add a test-scoped `build-helper-maven-plugin` resource entry) and delete the File fallback.

## Minor Issues

### MN-01: Misleading comment — `matFromJson` claims to accept flat arrays but throws

**File:** `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java:88-99`
**Issue:** Comment says "Nested [[row],[row]] or flat [v,v,...] (matmul_256 stores flattened input)" but the flat branch is an unconditional `throw new IllegalArgumentException`. matmul_256 correctly uses `flatMatFromJson`; the comment suggests otherwise and would mislead whoever adds the next golden test.
**Fix:** Delete the stale half of the comment.

### MN-02: bench pom duplicates parent compiler configuration with version drift

**File:** `bench/pom.xml:67-75`
**Issue:** Redefines `maven-compiler-plugin` at version 3.14.0 with hardcoded `source/target=17`, while the parent pom already pins the plugin and `${maven.compiler.source}` (sklearn's pom correctly inherits). Two sources of truth for compiler config; the bench module will silently diverge when the parent bumps Java version.
**Fix:** Delete the plugin block; parent config covers it.

### MN-03: GOLDEN.md contains Vietnamese/English mix and a typo in the opening line

**File:** `bench/GOLDEN.md:3`
**Issue:** "Golden reference values cho accuracy tests được sinh từ..." — mixed-language sentence with typo "cho". Elsewhere the doc is consistent. Cosmetic, but this is a top-level ops document for regenerating baselines.
**Fix:** Rewrite line 3 in one language.

### MN-04: BASELINE.md consistency with code — mostly accurate, one unexplained anomaly left standing

**File:** `.planning/phases/01-baseline-benchmark/BASELINE.md:27, 33`
**Issue:** Cross-checked tables against source: params, dataset names, shade jar name (`benchmarks.jar`), and reproduce commands all match the code. Two caveats: (a) the matmul 1024 score range "868–1596 ms ±5459–7352" has an error bar 6× larger than the score — these numbers carry no information and shouldn't be cited as a baseline without the promised multi-fork rerun; (b) `mean_reduce` (7.92–8.07 ms) measurably faster than `sum_reduce` (11.28–12.05 ms) on identical input is flagged "**" but left unexplained in the published table — mean literally calls `sum()` (`NDArray.java:340-341`), so a 30% gap between two runs of the same loop indicates the measurement is unreliable, not that mean is faster.
**Fix:** Annotate both rows in BASELINE.md as "unstable — remeasure before use as comparison point" (the doc partially does this; make it explicit for mean vs sum).

## Security Review

No vulnerabilities found. `GoldenReferenceTest` and `generate_golden.py` open only hardcoded constant paths built from script location / repo-relative constants; no user input reaches any file operation, no traversal surface, no injection surface. `generate_golden.py` writes only inside the repo tree via `os.path.join` of fixed components.

---

_Reviewed: 2026-08-26_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
