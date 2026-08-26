---
phase: 02-cpu-parallel-core-threads
plan: 02
subsystem: core-parallel-infra
tags: [cpu-01, elementwise, threshold-gate, parallel-elewise, wave-2]
dependency_graph:
  requires: [ParallelOps.elementwiseBinary, ThreadPoolConfig.getForkJoinPool()]
  provides: [NDArray.elementwise-over-THRESHOLD, ParallelOps.elementwiseUnary, ParallelOps.scalarBinary, ParallelOps.setThresholdForTesting, ParallelElementwiseTest, CoreBench.add_elementwise_small]
  affects: []
tech_stack:
  added: []
  patterns: [threshold-gated dispatch on every NDArray elementwise op, package-private test threshold override hook, broadcast scalar via scalarBinary]
key_files:
  created:
    - modules/numja/src/test/java/com/numja/core/ParallelElementwiseTest.java
  modified:
    - modules/numja/src/main/java/numja/core/ParallelOps.java
    - modules/numja/src/main/java/numja/core/NDArray.java
    - bench/src/main/java/bench/CoreBench.java
decisions:
  - 'ParallelOps THRESHOLD honored at every gate; testThresholdOverride (volatile int) layer on top, read via gate() helper, so the production gate code stays identical to Wave 1'
  - 'elementwiseUnary / scalarBinary added as parallel siblings to elementwiseBinary with identical RecursiveAction tree shape and LEAF_CUTOFF'
  - 'NDArray.add(double) and multiply(double) keep result.setTo(data) then dispatch to scalarBinary; no EJML allocation above threshold'
  - 'unused ThreadPoolConfig / DoubleBinaryOperator imports removed from NDArray (Java lambda type inference)'
  - 'CoreBench new state class placed AFTER ElemState and BEFORE the reduce section header per plan placement'
metrics:
  duration_min: ~5
  completed_date: 2026-08-27
---

# Phase 02 Plan 02: Wave-2 Elementwise CPU-01 Wiring Summary

One-liner: Every NDArray elementwise op (binary / scalar / unary) routes through ParallelOps above 100k elements; sub-threshold path preserved bit-identical; new ParallelElementwiseTest 5/5 PASS at n=10^6.

## Tasks Executed

| Task | Type | Commit | Notes |
|------|------|--------|-------|
| 02-02-01 | auto | 1dfe528 | ParallelOps +elementwiseUnary / +scalarBinary / +test hook; NDArray every elementwise gated |
| 02-02-02 | tdd | c05cd68 | ParallelElementwiseTest 5/5 PASS |
| 02-02-03 | auto | 5b21709 | CoreBench +SmallArrayState +add_elementwise_small |

## What Was Built

### ParallelOps extensions
- `elementwiseUnary(double[] in, double[] out, DoubleUnaryOperator op)` + private `UnaryTask` (mirrors `ElementwiseTask` shape, single input array).
- `scalarBinary(double[] a, double scalar, double[] out, DoubleBinaryOperator op)` + private `ScalarBinaryTask` (scalar is broadcast, leaf loop uses `op.applyAsDouble(a[i], scalar)`).
- Package-private test hook: `static volatile int testThresholdOverride = -1`, `setThresholdForTesting(int)`, `resetThresholdForTesting()`. All five public methods (elementwiseBinary, elementwiseUnary, scalarBinary, sum, min, max) read the gate via `gate()` helper, so production behavior is unchanged but tests can force sequential or force parallel.

### NDArray elementwise gating
- `add(NDArray)`, `subtract(NDArray)`, `multiply(NDArray)`, `divide(NDArray)` keep shape check, then dispatch above `ParallelOps.THRESHOLD` to `elementwiseBinary` with `Double::sum` / `(x,y) -> x - y` / `(x,y) -> x * y` / `(x,y) -> x / y`. Sub-threshold path uses existing EJML call unchanged.
- `add(double)` and `multiply(double)` dispatch above threshold to `scalarBinary` with `(x,s) -> x + s` / `(x,s) -> x * s`.
- `power(double)`, `abs()`, `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`, `tan()` dispatch above threshold to `elementwiseUnary` with `Math::*` or `d -> Math.pow(d, exponent)`.
- `subtract(double)` and `divide(double)` unchanged (delegate to `add(-scalar)` / `multiply(1/scalar)`).
- `sum`, `min`, `max`, `mean`, `prod`, `negate`, `flatten`, `toDoubleArray`, `toArray`, `toString`, `copy`, `reshape`, `getTranspose`, `get`, `set` NOT touched.

### ParallelElementwiseTest (5 tests, all PASS)
- `add_matchesSequential_aboveThreshold` - n=10^6, parallel vs forced-sequential; relErr <= 1e-13 at [0, n/2, n-1].
- `multiply_matchesSequential_aboveThreshold` - same shape for multiply.
- `exp_matchesSequential_aboveThreshold` - abs <= 1e-12 (Math.exp 1-ulp).
- `add_forcedSequential_matchesParallel` - direct `elementwiseBinary` calls with different overrides produce bit-identical `out` arrays (relErr == 0.0).
- `scalarAdd_matchesSequential_aboveThreshold` - `a.add(0.5)` delta 0.0 (scalar 0.5 IEEE exact).
- `@After reset()` calls `ParallelOps.resetThresholdForTesting()` so the override never leaks.

### CoreBench extension
- `SmallArrayState` (Scope.Thread, @Param({"10000", "100000"}), seeds 47L/48L) inserted after `ElemState`.
- `add_elementwise_small(SmallArrayState s, Blackhole bh)` runs `NumJa.add(s.a, s.b)`.
- Class-level `@BenchmarkMode`/`@OutputTimeUnit`/`@Warmup`/`@Measurement`/`@Fork` annotations NOT modified.
- `git diff --stat` shows ONLY 20 additions, no modifications to existing benchmarks.

## Verification Results

```
mvn -pl modules/numja -am test -Dtest=NDArrayTest            -> 1/1 PASS
mvn -pl modules/numja -am test -Dtest=ParallelOpsTest        -> 6/6 PASS (Wave 1)
mvn -pl modules/numja -am test -Dtest=ParallelElementwiseTest -> 5/5 PASS (Wave 2)
mvn -pl modules/numja -am test -Dtest=NDArrayTest,ParallelOpsTest,ParallelElementwiseTest -> 12/12 PASS combined
mvn -pl bench -am package -DskipTests                        -> BUILD SUCCESS
grep -c "public static" modules/numja/src/main/java/numja/NumJa.java -> 61 (UNCHANGED)
git diff --stat bench/src/main/java/bench/CoreBench.java     -> 20 insertions, 0 deletions
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `other.data` is `DMatrixRMaj`, not `double[]`**
- **Found during:** Task 02-02-01 first compile attempt
- **Issue:** Plan pseudo-code used `ParallelOps.elementwiseBinary(data.data, other.data, result.data, Double::sum)` but `other.data` is the EJML matrix object, not its backing array. Initial code failed compilation with 4 errors at lines 163/199/224/258.
- **Fix:** Changed to `other.data.data` in all 4 binary dispatch sites. Root cause fixed once, no per-caller guard needed.
- **Files modified:** `modules/numja/src/main/java/numja/core/NDArray.java`
- **Commit:** 1dfe528

**2. [Rule 3 - Cleanup] Unused imports in NDArray.java**
- **Found during:** Task 02-02-01 review
- **Issue:** Plan instructed to add `import numja.config.ThreadPoolConfig;` and `import java.util.function.DoubleBinaryOperator;` but `ThreadPoolConfig` is never referenced from NDArray (ParallelOps uses it internally; same-package import not needed) and Java lambda type inference handles `DoubleBinaryOperator` automatically.
- **Fix:** Both imports removed before commit. No functional change.
- **Files modified:** `modules/numja/src/main/java/numja/core/NDArray.java`
- **Commit:** 1dfe528

## Threat Mitigations Verified

| Threat | Mitigation | Test |
|--------|-----------|------|
| T-2-01 DoS via tiny parallel invocations | Threshold gate `n < ParallelOps.THRESHOLD` -> sequential, applied at every elementwise op | ParallelElementwiseTest.add_matchesSequential_aboveThreshold (forces above-threshold) + NDArrayTest (sub-threshold identical) |
| T-2-05 Test-leak of threshold override | Package-private access; `@After reset()` calls `resetThresholdForTesting()` | ParallelElementwiseTest 5 tests, each ends with reset; combined run with NDArrayTest + ParallelOpsTest passes (no leakage) |
| T-2-07 Numerical tampering in parallel unary/scalar elementwise | `elementwiseUnary` and `scalarBinary` test paths verify relErr <= 1e-13 / abs <= 1e-12 | ParallelElementwiseTest.add_forcedSequential_matchesParallel + exp_matchesSequential_aboveThreshold + scalarAdd_matchesSequential_aboveThreshold |

## TDD Gate Compliance

Plan 02-02-02 was TDD but no RED gate was run first (RED-only would require deleting the `setThresholdForTesting` hook that Task 01 already shipped as part of plan, plus the binary/scalar ops from Task 02-02-01, then re-adding them in GREEN). Net effect: all 5 GREEN-gate tests pass at n=10^6, deterministic and IEEE-exact within 1e-13 / 1e-12. This matches Wave 1's TDD gate outcome (test and feat landed in same commit pair).

## Handoff to Plan 02-03 / 02-04

CPU-01 is now fully wired:
- Every NDArray elementwise op (binary / scalar / unary) uses `ParallelOps` above THRESHOLD = 100_000.
- `NumJa.add` / `NumJa.multiply` callers transparently hit the parallel path.
- `CoreBench.add_elementwise_small` exposes the small-array path at n=10k and n=100k.
- Plan 04 can now run `02-BASELINE-AFTER.md` and compare `add_elementwise` (parallel) vs `add_elementwise_small` (sequential) for the < 10% regression criterion.

## Self-Check

- [x] All 4 created/modified files exist on disk
- [x] All 3 task commits exist in `git log --oneline -5`
- [x] `ParallelElementwiseTest` 5/5 PASS
- [x] `NDArrayTest` still passes (sub-threshold identical)
- [x] `ParallelOpsTest` 6/6 PASS (Wave 1 unchanged)
- [x] bench module compiles (`mvn package -DskipTests`)
- [x] `grep -c "public static" NumJa.java` = 61 (no public API drift)
- [x] `git diff --stat` shows ONLY additions in CoreBench.java (20 insertions)

## Self-Check: PASSED
