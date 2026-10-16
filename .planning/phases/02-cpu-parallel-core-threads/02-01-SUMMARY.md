---
phase: 02-cpu-parallel-core-threads
plan: 01
subsystem: core-parallel-infra
tags: [cpu-01, cpu-02, forkjoin, parallel-ops, wave-0, threshold-gate]
dependency_graph:
  requires: []
  provides: [ThreadPoolConfig.getForkJoinPool(), ParallelOps.elementwiseBinary/sum/min/max]
  affects: []
tech_stack:
  added: []
  patterns: [ForkJoinPool singleton via double-checked locking, RecursiveAction tree with leaf-cutoff, threshold-gated dispatch]
key_files:
  created:
    - modules/numja/src/main/java/numja/core/ParallelOps.java
    - modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java
    - modules/numja/src/test/java/com/numja/config/ThreadPoolConfigTest.java
  modified:
    - modules/numja/src/main/java/numja/config/ThreadPoolConfig.java
decisions:
  - THRESHOLD = 100_000 (matches research User Constraints)
  - LEAF_CUTOFF = 16_384 (package-private tuning knob, per research Open Q#1)
  - Double-checked locking on volatile ForkJoinPool field for lazy init
  - Deterministic left-to-right merge in SumTask for relErr stability
  - setThreads does NOT recreate pool (matches existing setThreads semantics; documented limitation)
metrics:
  duration_min: ~6
  completed_date: 2026-08-27
---

# Phase 02 Plan 01: Wave-0 Parallel Infrastructure Summary

One-liner: Lazy ForkJoinPool singleton on ThreadPoolConfig + threshold-gated ParallelOps utility (elementwise + sum/min/max) — Wave-0 infra for CPU-01/CPU-02 wiring, no public API drift.

## Tasks Executed

| Task | Type | Commit | Notes |
|------|------|--------|-------|
| 02-01-01 | tdd | fcddc97 (RED) + 4bce08d (GREEN) | ThreadPoolConfig FJP singleton + 3-test regression suite |
| 02-01-02 | tdd | 2b90397 (RED) + b835d61 (GREEN) | ParallelOps utility + 6-test threshold + correctness suite |

## What Was Built

### ThreadPoolConfig.getForkJoinPool()
- Added `import java.util.concurrent.ForkJoinPool`, `private volatile ForkJoinPool forkJoinPool;`, and a package-private `public ForkJoinPool getForkJoinPool()` using double-checked locking (mirrors the lazy-init shape of the existing `getInstance()`).
- Pool is sized to `currentThreads` (already 60% of cores via `calculateOptimalThreads()`).
- `setThreads()` does NOT recreate the pool — documented as a known limitation matching existing `setThreads` semantics.
- Mitigates T-2-02 (pool exhaustion): one pool for JVM lifetime.

### numja.core.ParallelOps
- `public static final int THRESHOLD = 100_000;` and package-private `static final int LEAF_CUTOFF = 16_384;`.
- Public surface (no NDArray body changes): `elementwiseBinary(double[], double[], double[], DoubleBinaryOperator)`, `sum(double[])`, `min(double[])`, `max(double[])`.
- Private `RecursiveAction` inner classes: `ElementwiseTask`, `SumTask`, `MinTask`, `MaxTask`. Each splits via `(lo + hi) >>> 1` (overflow-safe), leaves run a sequential loop over LEAF_CUTOFF-sized chunks, and parents merge deterministically left-to-right (sum) or via `Math.min`/`Math.max` (min/max).
- Sub-threshold branch stays sequential — guarantees the existing single-threaded path is preserved for small arrays and zero overhead for those callers.

## Tests Added

### ThreadPoolConfigTest (3 tests, all pass)
- `getForkJoinPool_isSingleton` — assertSame on two successive calls.
- `getForkJoinPool_sizedToCurrentThreads` — `getParallelism() == getCurrentThreads()`.
- `getForkJoinPool_survivesSetThreads` — `setThreads(1)` does NOT throw; pool identity preserved; `resetToDefaults()` in `@After`.

### ParallelOpsTest (6 tests, all pass)
- `THRESHOLD_is100k` — asserts constant value.
- `belowThreshold_usesSequential` — n=THRESHOLD-1; output equals elementwise sum.
- `elementwiseBinary_matchesSequential` — n=1_000_000; checks `[0, mid, last]` indices within delta 1e-9.
- `sum_matchesSequential` — n=1_000_000; relErr <= 1e-13 (matches STATE.md / GoldenReferenceTest tolerance).
- `min_matchesSequential` / `max_matchesSequential` — n=500_000; exact match (delta 0.0) since min/max are deterministic IEEE ops.

## Verification Results

```
mvn -pl modules/numja -am test -Dtest=ThreadPoolConfigTest  -> 3/3 PASS
mvn -pl modules/numja -am test -Dtest=ParallelOpsTest       -> 6/6 PASS
mvn -pl modules/numja -am test -Dtest=NDArrayTest           -> 1/1 PASS (no regression)
grep -c "public static" modules/numja/src/main/java/numja/NumJa.java         -> 61 (unchanged)
grep -c "public static" modules/numja/src/main/java/numja/core/ArrayOps.java -> 34 (unchanged)
```

No public API drift (RESEARCH Pitfall 7 confirmed). `ParallelOps` is the only new public class; its public methods are intentionally package-internal helpers — they will be called from `NDArray` in Plans 02-03 and remain invisible to library consumers.

## Deviations from Plan

None — plan executed exactly as written.

## Threat Mitigations Verified

| Threat | Mitigation | Test |
|--------|-----------|------|
| T-2-01 DoS via tiny parallel invocations | Threshold gate `n < THRESHOLD` -> sequential | ParallelOpsTest.belowThreshold_usesSequential |
| T-2-02 ForkJoinPool exhaustion | Singleton via getForkJoinPool() + double-checked locking | ThreadPoolConfigTest.getForkJoinPool_isSingleton |
| T-2-NUM Numerical stability of parallel sum | Deterministic left-to-right tree merge | ParallelOpsTest.sum_matchesSequential (relErr <= 1e-13) |

## TDD Gate Compliance

```
fcddc97  test(02-01-01): add failing test for ForkJoinPool singleton accessor   (RED gate)
4bce08d  feat(02-01-01): add getForkJoinPool() singleton to ThreadPoolConfig    (GREEN gate)
2b90397  test(02-01-02): add failing tests for ParallelOps                       (RED gate)
b835d61  feat(02-01-02): add ParallelOps ForkJoin utility                       (GREEN gate)
```

Both RED gates ran and produced `cannot find symbol` compile failures (verified) before GREEN implementation landed.

## Handoff to Plans 02-02 / 02-03

CPU-01 / CPU-02 can now directly call:
- `ParallelOps.elementwiseBinary(a, b, out, Double::sum)` — elementwise add/sub/mul/div
- `ParallelOps.sum(data)` / `min` / `max` — reduce paths
- `ThreadPoolConfig.getInstance().getForkJoinPool()` — direct pool access if needed

No further FJP plumbing required in downstream plans.

## Self-Check

- [x] All 4 created/modified files exist on disk
- [x] All 4 commits exist in `git log --oneline -5`
- [x] Both new test classes pass (ThreadPoolConfigTest 3/3, ParallelOpsTest 6/6)
- [x] NDArrayTest still passes (no regression)
- [x] Public static counts unchanged in NumJa.java (61) and ArrayOps.java (34)

## Self-Check: PASSED
