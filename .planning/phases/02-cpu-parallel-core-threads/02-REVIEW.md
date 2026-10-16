---
phase: 02-cpu-parallel-core-threads
reviewed: 2026-08-27T12:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - modules/numja/src/main/java/numja/config/ThreadPoolConfig.java
  - modules/numja/src/test/java/com/numja/config/ThreadPoolConfigTest.java
  - modules/numja/src/main/java/numja/core/ParallelOps.java
  - modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java
  - modules/numja/src/main/java/numja/core/NDArray.java
  - modules/numja/src/test/java/com/numja/core/ParallelElementwiseTest.java
  - modules/numja/src/test/java/com/numja/core/ParallelReduceTest.java
  - modules/numja/src/test/java/com/numja/core/ParallelRegressionTest.java
  - bench/src/main/java/bench/CoreBench.java
findings:
  critical: 0
  warning: 5
  info: 6
  total: 11
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-08-27T12:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 2 wiring is sound. Threshold gating, FJP singleton, sub-threshold fallback, and per-op ForkJoin decomposition are correctly implemented. STRIDE mitigations (T-2-01 threshold gate, T-2-02 pool singleton, T-2-05 test-leak @After reset, T-2-NUM deterministic merge) verified present in code. Public API drift check: `NumJa.java` = 61 `public static` and `ArrayOps.java` = 34 `public static` — both match the locked v0.2.0 surface. JMH class-level annotations on `CoreBench.java` (lines 17-21) preserved — `@BenchmarkMode(AverageTime)` + `@OutputTimeUnit(MILLISECONDS)` + `@Warmup(5,1)` + `@Measurement(5,1)` + `@Fork(1)` unchanged.

Findings are concentrated in test-side naming and coverage, plus one defensive-coding gap in the parallel reduce tasks. No correctness, security, or numerical-stability defects in production code. The `multiply_elementwise` 10^7 1.42x flag is correctly deferred per context.

## Warnings

### WR-01: ParallelRegressionTest names assert "2x faster" but bound is `<= 1.30` (regression gate, not speedup)

**File:** `modules/numja/src/test/java/com/numja/core/ParallelRegressionTest.java:88-114, 168-191`
**Issue:** Three tests (`largeArray_isAtLeast2xFaster`, `reduceLarge_isAtLeast2xFaster`, `smallArray_under10PercentRegression_*`) have names that imply positive performance guarantees ("isAtLeast2xFaster", "under10PercentRegression") but actually assert a regression-only bound:
- `largeArray_isAtLeast2xFaster`: assertion `ratio <= 1.30` — allows parallel to be 30% slower than raw sequential (i.e., the test name is literally inverted: a "2x faster" test passes when parallel is *at most 1.30x slower*).
- `smallArray_under10PercentRegression_at10k`: bound is `1.50` (50% slower allowed), not 10%.
- `smallArray_under10PercentRegression_at100k`: bound is `3.00` (200% slower allowed), not 10%.

The class javadoc lines 23-39 documents the actual intent ("catches regressions where the parallel path silently degrades back to raw-loop parity or worse"), which contradicts the test names. Anyone reading the test names + JMH baseline (add 10^7 = 2.43x) will believe the JUnit test asserts ≥2x; it does not. Misleading green test on CI is a real risk if the parallel path silently degrades to 0.9x speedup — the test still passes.
**Fix:** Rename to reflect regression-gate semantics:
- `largeArray_isAtLeast2xFaster` → `largeArray_parallelDoesNotRegressMoreThan30Percent`
- `reduceLarge_isAtLeast2xFaster` → `reduceLarge_parallelDoesNotRegressMoreThan30Percent`
- `smallArray_under10PercentRegression_at10k` → `smallArray_at10k_thresholdGatePreserved` (bound 1.50)
- `smallArray_under10PercentRegression_at100k` → `smallArray_at100k_thresholdBoundaryStable` (bound 3.00)
Or, if the project wants the JUnit gate to actually assert user-visible speedup, swap the bound to a ratio-based assertion backed by an absolute ms budget derived from `02-BASELINE-AFTER.md` median. The class javadoc should be updated in lockstep.

### WR-02: ParallelReduceTest.sum_forcedSequential_matchesParallel tests only sequential determinism, not cross-path equivalence

**File:** `modules/numja/src/test/java/com/numja/core/ParallelReduceTest.java:96-102`
**Issue:** The test sets `setThresholdForTesting(Integer.MAX_VALUE)` (sequential path) then calls `ParallelOps.sum(data)` twice and asserts `first == second`. Both calls take the sequential branch — so the test does NOT verify "forcedSequential matches Parallel". It only verifies that the sequential sum is deterministic. The test name is wrong and the test does not assert what it claims.
**Fix:** Either (a) mirror `ParallelElementwiseTest.add_forcedSequential_matchesParallel` (lines 95-120) — run sequential, reset, run parallel, assert equality; or (b) rename to `sum_isDeterministic_sequentialPath` and add a separate `sum_parallelMatchesSequential_aboveThreshold` test that pairs the two paths.

### WR-03: ParallelOpsTest missing @After reset for testThresholdOverride leak hygiene

**File:** `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java:1-94`
**Issue:** This test class does NOT call `ParallelOps.resetThresholdForTesting()` in `@After`, and currently does not call `setThresholdForTesting()` either — but it imports `ParallelOps` and other tests in the suite do mutate the static `testThresholdOverride`. If a future test (or a refactor) adds `setThresholdForTesting` here and forgets to reset, the override leaks across test classes (T-2-05 mitigation partially weakened). The other four test classes in scope all have `@After reset()` — this one is the gap.
**Fix:** Add the standard hygiene block:
```java
@After
public void reset() {
    ParallelOps.resetThresholdForTesting();
}
```

### WR-04: ParallelElementwiseTest coverage gaps for parallel-path ops

**File:** `modules/numja/src/test/java/com/numja/core/ParallelElementwiseTest.java:1-139`
**Issue:** Only `add`, `multiply`, `exp`, and `add(double)` are cross-path tested above THRESHOLD. The following parallel-gated paths in `NDArray.java` have NO above-threshold correctness test:
- `subtract(NDArray)` (line 189-202) — `(x, y) -> x - y`
- `divide(NDArray)` (line 246-261) — `(x, y) -> x / y`
- `power(double)` (line 273-285) — `d -> Math.pow(d, exponent)`
- `abs()` (line 290-301) — `Math::abs`
- `sqrt()` (line 306-317) — `Math::sqrt`
- `log()` (line 338-349) — `Math::log`
- `sin()` (line 354-365) — `Math::sin`
- `cos()` (line 370-381) — `Math::cos`
- `tan()` (line 386-397) — `Math::tan`
- `multiply(double)` (line 232-241) — `(x, s) -> x * s`

The shared `ParallelOps.elementwiseBinary` / `elementwiseUnary` / `scalarBinary` are tested by `add`/`multiply`/`exp`/`scalarAdd`, but the wiring in each `NDArray` method (shape check, `result.setTo(data)` for in-place cases like `power`/`add(double)` line 175-182, threshold gate, return value) is per-method and not exercised for these ops. RESEARCH §Pitfall 5 explicitly warns about numerical drift in tree-parallel paths; `divide` near zero and `power` with negative bases are most at risk.
**Fix:** Add at minimum `subtract_aboveThreshold_matchesSequential` and `divide_aboveThreshold_matchesSequential` tests. For `power`/`sqrt`/`log`/`sin`/`cos`/`tan`/`abs`, a single `unary_aboveThreshold_matchesSequential` parameterized over the operator is sufficient.

### WR-05: MinTask.best / MaxTask.best default-initialised to 0.0 — defensive-coding gap

**File:** `modules/numja/src/main/java/numja/core/ParallelOps.java:236, 262`
**Issue:** `MinTask.best` and `MaxTask.best` are non-final `double` fields with Java default initialiser `0.0`. They are only written inside `compute()`. In practice, the parent task reads `left.best`/`right.best` only after `invokeAll(left, right)` returns, which establishes happens-before and guarantees the children have written their values. **However**, if a future refactor (or a fork into a counting/sentinel pattern) ever exposes `best` to a code path that doesn't go through the leaf, the parent would silently use `0.0` — for `MinTask` over an all-positive array that's silently wrong. The pre-leaf reads are guarded by `invokeAll` today, but the field-level invariant "best is initialised before parent reads it" is not defended at the field level.
**Fix:** Either (a) make the field `final` and pass the initial value into the constructor (`new MinTask(data, lo, hi, Double.MAX_VALUE)`), or (b) add a `assert best != 0.0 || ... ;` invariant. Option (a) is the lazier fix and matches the existing `SumTask.total` pattern's intent (leaf writes, parent reads after join — but `total` is correctly initialised to `0.0` for sum, which is the correct identity; `best = 0.0` is wrong for min/max). ponytail: this is a small defensive-coding gap, not a current bug — fix when the next refactor touches the reduce task classes.

## Info

### IN-01: Static `testThresholdOverride` shared across test classes — informational only

**File:** `modules/numja/src/main/java/numja/core/ParallelOps.java:29`
**Issue:** `static volatile int testThresholdOverride = -1;` is mutable global state. JUnit 4 + Surefire default forkMode runs all classes in one JVM sequentially. Tests within a single class run sequentially too, so `@After reset()` is sufficient today. If Surefire config ever flips to forkMode=always or parallel test execution, this becomes a real leak vector.
**Fix:** None required for Phase 2. Document the assumption in `ParallelOps.java` javadoc ("single-JVM test execution required; uses static volatile state").

### IN-02: ParallelOpsTest.elementwiseBinary_matchesSequential checks only 3 indices out of 1M

**File:** `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java:52-54`
**Issue:** `for (int idx : new int[]{0, n / 2, n - 1})` — verifies 3 of 1_000_000 indices. Low coverage for the parallel path; a race condition that corrupts indices 1..n/2-1 would slip past.
**Fix:** For random-uniform `[0,1)` data, elementwise is per-index independent so 3 indices give strong statistical confidence; this is acceptable. If paranoid, add `n / 4`, `3 * n / 4` for 5 samples. Skip if the JMH gate is the binding check.

### IN-03: NDArray.prod() is intentionally out of Phase 2 scope (unchanged sequential loop)

**File:** `modules/numja/src/main/java/numja/core/NDArray.java:458-464`
**Issue:** `prod()` still uses a sequential loop and is not gated through `ParallelOps`. RESEARCH §Requirements lists only `add/mul/exp/sum/mean/min/max` in scope. `prod()` is a reduce (multiplicative) — left out per scope, not a defect.
**Fix:** None for Phase 2. If Phase 3+ adds product, gate it the same way as `sum`.

### IN-04: Class-level JMH annotations on CoreBench preserved

**File:** `bench/src/main/java/bench/CoreBench.java:17-21`
**Issue:** Verified present and unmodified: `@BenchmarkMode(Mode.AverageTime)` + `@OutputTimeUnit(TimeUnit.MILLISECONDS)` + `@Warmup(iterations=5, time=1)` + `@Measurement(iterations=5, time=1)` + `@Fork(1)`. Matches Phase 1 baseline config per `02-PATTERNS.md` line 333. Confirms RESEARCH §Pitfall 3 ("Do NOT modify class-level annotations") was respected.
**Fix:** None — confirmation only.

### IN-05: NaN/Inf edge cases not exercised in any Phase 2 test

**File:** `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java` and parallel-elementwise/reduce test files
**Issue:** All test data uses `Random.nextDouble()` which returns `[0.0, 1.0)` — no NaN, no -0.0, no infinity, no denormals. Pre-existing `NDArray.min/max` sequential path has known NaN behavior (returns `Double.MAX_VALUE` for NaN inputs because `NaN < x` is always false), and the parallel path inherits the same behavior (deterministic, identical to sequential). Not a Phase 2 regression but a coverage gap.
**Fix:** None required for Phase 2 acceptance. Add a `NaN_propagatesCorrectly_inMinMax` test if a future bug report surfaces.

### IN-06: Floating-point tree-reduce merge order documented but not stress-tested for pathological inputs

**File:** `modules/numja/src/main/java/numja/core/ParallelOps.java:206-230` (`SumTask`)
**Issue:** RESEARCH §Pitfall 5 commits to relErr ≤1e-13 for tree-reduce vs sequential. The current test data (`Random.nextDouble()` in `[0,1)`) gives magnitudes ~0.5 and partial sums ~500_000 — well-behaved. A pathological input (large-magnitude values mixed with tiny values, or `[1e15, 1.0, -1e15, ...]`) could exceed 1e-13 cancellation error. The merge order (`left.total + right.total`, left-to-right) is correct and deterministic; only the magnitude of accumulated rounding error is unverified at the extremes.
**Fix:** None for Phase 2 (test data is representative of real workloads). Add a `sum_pathologicalCancellation` test if Kahan summation is deferred to Phase 3+.

---

_Reviewed: 2026-08-27T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
