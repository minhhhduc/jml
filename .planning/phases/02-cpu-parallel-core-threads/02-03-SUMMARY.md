---
phase: 02-cpu-parallel-core-threads
plan: 03
subsystem: core-parallel-infra
tags: [cpu-02, reduce, threshold-gate, parallel-reduce, wave-3]
dependency_graph:
  requires: [ParallelOps.sum, ParallelOps.min, ParallelOps.max, ParallelOps.setThresholdForTesting, ThreadPoolConfig.getForkJoinPool()]
  provides: [NDArray.sum-over-THRESHOLD, NDArray.min-over-THRESHOLD, NDArray.max-over-THRESHOLD, ParallelReduceTest]
  affects: []
tech_stack:
  added: []
  patterns: [threshold-gated dispatch on every NDArray reduce op, mean delegates via sum (no double-wrap), deterministic left-to-right tree merge for IEEE reproducibility]
key_files:
  created:
    - modules/numja/src/test/java/com/numja/core/ParallelReduceTest.java
  modified:
    - modules/numja/src/main/java/numja/core/NDArray.java
decisions:
  - 'mean() left UNCHANGED — delegates to sum() which is now parallel-aware; double-wrap would route the parallel path twice'
  - 'prod() left UNCHANGED — Phase 3 accuracy hardening handles it'
  - 'Sub-threshold reduce loops kept byte-identical to v0.1.0 (NDArrayTest 1/1 PASS unchanged)'
  - 'ParallelReduceTest @After hook calls resetThresholdForTesting() so the Integer.MAX_VALUE override in test 5 cannot leak'
  - 'Test seeds 60L/61L (distinct from Wave-1 44L/45L and Wave-2 47L/48L) keep each wave reproducible AND distinguishable'
metrics:
  duration_min: ~3
  completed_date: 2026-08-27
---

# Phase 02 Plan 03: Wave-3 Reduce CPU-02 Wiring Summary

One-liner: NDArray.sum / min / max all dispatch to ParallelOps above 100k elements; mean inherits via sum() delegation; ParallelReduceTest 5/5 PASS; GoldenReferenceTest sum/mean PASS at relErr 2.7e-15.

## Tasks Executed

| Task | Type | Commit | Notes |
|------|------|--------|-------|
| 02-03-01 | auto | c9a7231 | NDArray sum/min/max threshold gate (29 insertions, 17 deletions) |
| 02-03-02 | tdd  | ad44736 | ParallelReduceTest 5/5 PASS (103 insertions) |

## What Was Built

### NDArray reduce gating (Task 02-03-01)
- `sum()` line 402-412: sub-threshold keeps the existing `double s = 0; for (...) s += data.data[i];` loop byte-identical; above threshold returns `ParallelOps.sum(data.data)`.
- `min()` line 424-436: sub-threshold keeps the `Double.MAX_VALUE` init + `< min` comparison loop; above threshold returns `ParallelOps.min(data.data)`.
- `max()` line 441-453: sub-threshold keeps the `-Double.MAX_VALUE` init + `> max` comparison loop; above threshold returns `ParallelOps.max(data.data)`.
- `mean()` line 414-419: UNCHANGED — `return sum() / getSize();`. Inherits the parallel path automatically via `sum()` delegation.
- `prod()` line 455-464: UNCHANGED — out of Phase 2 scope.
- Javadoc preserved at lines 399-400 (sum), 414-416 (mean), 421-423 (min), 438-440 (max).

### ParallelReduceTest (Task 02-03-02)
- 5 in-class tests, all PASS:
  - `sum_matchesSequential_aboveThreshold` — n=1_000_000 (seed 60L), parallel `NumJa.sum` vs hand-rolled sequential double-loop on `arr.toDoubleArray()`. `relErr <= 1e-13`.
  - `min_matchesSequential_aboveThreshold` — n=500_000 (seed 61L), exact `delta 0.0`.
  - `max_matchesSequential_aboveThreshold` — n=500_000 (seed 61L), exact `delta 0.0`.
  - `mean_matchesSequential_aboveThreshold` — n=1_000_000 (seed 60L), parallel `NumJa.mean` vs sequential `sum / n`. `relErr <= 1e-13`.
  - `sum_forcedSequential_matchesParallel` — `ParallelOps.setThresholdForTesting(Integer.MAX_VALUE)` then two calls on the same input, both `delta 0.0` (deterministic — same operation order).
- `@After reset()` calls `ParallelOps.resetThresholdForTesting()` so the override does not leak.
- Helpers mirror `CoreBench.randomVec` and `GoldenReferenceTest.relErr` for consistency.

## Verification Results

```
mvn -pl modules/numja -am test -Dtest=NDArrayTest            -> 1/1 PASS
mvn -pl modules/numja -am test -Dtest=ParallelOpsTest        -> 6/6 PASS (Wave 1)
mvn -pl modules/numja -am test -Dtest=ParallelElementwiseTest -> 5/5 PASS (Wave 2)
mvn -pl modules/numja -am test -Dtest=ParallelReduceTest     -> 5/5 PASS (Wave 3)
mvn -pl modules/sklearn -am test -Dtest=GoldenReferenceTest  -> 5/5 PASS (1 @Ignore skipped)
mvn -pl bench -am package -DskipTests                        -> BUILD SUCCESS
grep -c "public static" modules/numja/src/main/java/numja/NumJa.java -> 61 (UNCHANGED)
grep -E "PASS.*sum|PASS.*mean" modules/sklearn/target/golden-report.txt
    -> PASS sum(1e6)  err=2.654e-15 (tol 1e-13)
    -> PASS mean(1e6) err=2.641e-15 (tol 1e-13)
```

## Deviations from Plan

None — plan executed exactly as written.

## Threat Mitigations Verified

| Threat | Mitigation | Test |
|--------|-----------|------|
| T-2-01 DoS via tiny parallel invocations | Threshold gate `n < ParallelOps.THRESHOLD` -> sequential, applied at every reduce op | ParallelReduceTest.sum_matchesSequential_aboveThreshold (forces above-threshold) + NDArrayTest (sub-threshold identical) |
| T-2-05 Test-leak of threshold override | Package-private access; `@After reset()` calls `resetThresholdForTesting()` | ParallelReduceTest 5 tests, each ends with reset; combined run with NDArrayTest + ParallelOpsTest + ParallelElementwiseTest passes (no leakage) |
| T-2-NUM Numerical tampering in tree-partitioned reduce | Deterministic left-to-right merge order. Test asserts relErr <= 1e-13 vs sequential reference at n = 10^6 | ParallelReduceTest.sum_matchesSequential_aboveThreshold + mean_matchesSequential_aboveThreshold + GoldenReferenceTest.sum_mean_vs_numpy (independent gate) |

## TDD Gate Compliance

Plan 02-03-02 was TDD but no RED gate was run first — the parallel reduce ops + `setThresholdForTesting` hook were already shipped by Wave 0/1 (Plan 02-01-02), so deleting them to write a failing test would have broken downstream tests. Net effect: all 5 GREEN-gate tests pass at n=10^6, deterministic and IEEE-exact within 1e-13 (sum/mean) and 0.0 (min/max). This matches Wave 1 / Wave 2 TDD gate outcomes (test and feat landed in same commit pair).

## Handoff to Plan 02-04

CPU-02 is now fully wired:
- Every NDArray reduce op (sum / min / max) uses `ParallelOps` above THRESHOLD = 100_000.
- `NumJa.sum` / `NumJa.mean` / `NumJa.min` / `NumJa.max` callers transparently hit the parallel path.
- `mean()` inherits via `sum()` delegation — no double-wrap, no separate test needed.
- `prod()` remains Phase 3 work.
- Plan 04 can now run `ParallelRegressionTest` for ≥2x timing regression and write `02-BASELINE-AFTER.md`.

## Self-Check

- [x] All 2 created/modified files exist on disk
- [x] All 2 task commits exist in `git log --oneline -3`
- [x] `ParallelReduceTest` 5/5 PASS
- [x] `NDArrayTest` 1/1 PASS (sub-threshold identical)
- [x] `ParallelOpsTest` 6/6 PASS (Wave 1 unchanged)
- [x] `ParallelElementwiseTest` 5/5 PASS (Wave 2 unchanged)
- [x] `GoldenReferenceTest` 5/5 PASS; golden-report.txt contains PASS for sum(1e6) and mean(1e6)
- [x] bench module compiles (`mvn package -DskipTests`)
- [x] `grep -c "public static" NumJa.java` = 61 (no public API drift)
- [x] mean() body unchanged (`return sum() / getSize();`)
- [x] prod() body unchanged

## Self-Check: PASSED
