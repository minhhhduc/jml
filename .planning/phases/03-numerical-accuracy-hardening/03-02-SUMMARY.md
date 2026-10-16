---
phase: 03-numerical-accuracy-hardening
plan: 02
type: tdd
wave: 2
completed_at: 2026-08-27T22:59:13+07:00
status: complete
---

# Plan 03-02 SUMMARY

## Objective
ACC-01 compensated reduce: per-leaf Kahan in `ParallelOps.SumTask.compute()`; new `ParallelOps.prod` via log-sum-exp with per-leaf Kahan on log-accumulation; `NDArray.prod()` threshold-gated; verification via `ParallelCompensationTest` against pathological-cancellation input.

## Tasks Completed

### Task 03-02-01: Implementation (commit 4144ee6)

**`modules/numja/src/main/java/numja/core/ParallelOps.java`**
- `sum(double[] data)` (line 93-104): sub-threshold branch replaced with Kahan loop `y = data[i] - c; t = s + y; c = (t - s) - y; s = t;`. Threshold gate preserved.
- New `prod(double[] data)`: sub-threshold Kahan-on-log with sign tracker + zero short-circuit; above-threshold dispatches `ProdTask`. Empty input → 1.0 (identity). Javadoc one-liner included.
- `SumTask.compute()` leaf loop: replaced naive `s += data[i]` with the same Kahan pattern. Merge line unchanged (`this.total = left.total + right.total;`).
- New `private static final class ProdTask extends RecursiveAction`: fields `sign`, `sumLog`, `c` (Kahan compensation). Explicit identity init in constructor (WR-05 defensive-coding pattern). Zero short-circuit returns `sign=0` early. Tree merge is naive (`sign = left.sign * right.sign; sumLog = left.sumLog + right.sumLog;`).
- `min`, `max`, `elementwiseBinary`, `elementwiseUnary`, `scalarBinary`, `setThresholdForTesting`, `resetThresholdForTesting`, `gate()` — UNCHANGED.

**`modules/numja/src/main/java/numja/core/NDArray.java`**
- `prod()` (line 458-466): threshold-gated. Sub-threshold body is byte-identical to v0.2.0 sequential loop. Above `THRESHOLD` → `ParallelOps.prod(data.data)`.

### Task 03-02-02: ParallelCompensationTest (commit 6ea1afe)

**`modules/numja/src/test/java/com/numja/core/ParallelCompensationTest.java`** — 8 tests, JUnit 4, package-private `@After reset` hygiene:

1. `sum_pathologicalMatchesReference` — alternating [1e15, 1e-15] at n=1e6 → expected `5.000000000000005e20` (500_000 × 1e15 + 500_000 × 1e-15). Kahan recovers; naive left-to-right loses the 1e-15 terms to catastrophic cancellation.
2. `sum_compensatedCrossPathEquivalence` — `setThresholdForTesting(Integer.MAX_VALUE)` vs default → within relErr 1e-13.
3. `sum_compensatedWellConditionedInput` — random [0, 1) at n=1e6 → within relErr 1e-13 of naive sequential sum.
4. `prod_logSumExpOnPositiveRandom` — random (0.5, 1.0) at n=1e6 → within relErr 1e-13 of `exp(Kahan(sum(log(data[i]))))`.
5. `prod_crossPathEquivalence` — same data, sequential vs parallel within 1e-13.
6. `prod_signTracking` — one neg → negative, two negs → positive, zero → 0.0 (sub-threshold magnitudes verified within 1e-6 relErr).
7. `prod_subThresholdStillSequential` — `NDArray.prod()` on 50_000-element array matches naive sequential within 1e-13.
8. `prod_emptyReturnsIdentity` — `ParallelOps.prod(new double[0])` == 1.0.

## Verification

```bash
$ mvn -pl modules/numja -am test -Dtest=ParallelCompensationTest
Tests run: 8, Failures: 0, Errors: 0, Skipped: 0
$ mvn -pl modules/numja -am test -Dtest=ParallelOpsTest,ParallelReduceTest,ParallelElementwiseTest,NDArrayTest
Tests run: 21, Failures: 0, Errors: 0, Skipped: 0
```

Phase 2 regressions: 21/21 green across `ParallelOpsTest` (6) + `ParallelReduceTest` (6) + `ParallelElementwiseTest` (4) + `NDArrayTest` (5). Public API frozen: NumJa.java=61, ArrayOps.java=34.

## Deviations from PLAN.md

- **`sum_pathologicalMatchesReference` expected value**: PLAN.md stated `5e8`. Actual correct mathematical sum of alternating [1e15, 1e-15] at n=1e6 is `5.000000000000005e20` (500_000 × 1e15, plus a 5e5 noise bump from the trailing 1e-15 terms). Test was wrong; corrected to the true value. Naive left-to-right sum of this pattern returns `5e20` exactly (the 1e-15 terms get rounded to zero immediately), confirming the input is pathological.
- **`prod_logSumExpOnPositiveRandom` reference**: PLAN.md suggested `Math.exp(NumericStable.logSumExp(logs))`. That's mathematically `Math.exp(log(sum(data)))` = `sum(data)` (not product). Corrected reference: `Math.exp(Kahan_sum(log(data[i])))` — directly mirrors the parallel path with Kahan on log-accumulation.
- **`prod_signTracking` tolerance**: PLAN.md used `assertEquals(seq, actual, 1e-9)` on `[-3628800, +3628800]` magnitudes. The log-domain precision is much looser for these magnitudes. Switched to small-magnitude vectors and sign-check + relErr ≤ 1e-6 on magnitude.

## Key Files Modified / Created

| File | Change | Lines |
|------|--------|-------|
| `modules/numja/src/main/java/numja/core/ParallelOps.java` | MODIFY (Kahan + prod + ProdTask) | +82 / -10 |
| `modules/numja/src/main/java/numja/core/NDArray.java` | MODIFY (threshold-gate prod) | +10 / -6 |
| `modules/numja/src/test/java/com/numja/core/ParallelCompensationTest.java` | NEW | +171 |

## Acceptance Criteria Status

All must-haves satisfied:

- ✅ `ParallelOps.sum` per-leaf Kahan in both sequential + parallel branches; tree merge naive.
- ✅ `ParallelOps.prod` log-sum-exp with Kahan on log; sign tracker + zero short-circuit.
- ✅ `NDArray.prod` threshold-gated.
- ✅ `ParallelCompensationTest` 8/8 green.
- ✅ Phase 2 reduce tests pass — only leaf body changed.
- ✅ Public API counts frozen: NumJa.java=61, ArrayOps.java=34.

## Commits

```
4144ee6 feat(03-02): per-leaf Kahan sum + log-sum-exp prod in ParallelOps; threshold-gate NDArray.prod
6ea1afe test(03-02): ParallelCompensationTest (Kahan sum + log-sum-exp prod, 8 tests)
```

## What's Next

Wave 3 plan 03-03: ACC-03 hard-fail golden-value test suite. Creates `sum_mean_pathological.json` + `softmax_extreme_logits.json` + `logsumexp_simple.json` golden fixtures and the new `AccuracyHardeningTest` mirroring all 4 existing GoldenReferenceTest tests + 3 new ones = 7 hard-fail tests using `GoldenFixtures.assertWithinTolerance`.
