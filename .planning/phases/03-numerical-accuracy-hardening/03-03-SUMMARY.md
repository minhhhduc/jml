---
phase: 03-numerical-accuracy-hardening
plan: 03
type: execute
wave: 3
completed_at: 2026-08-28T00:04:43+07:00
status: complete
---

# Plan 03-03 SUMMARY

## Objective
ACC-03 hard-fail golden-value test suite. Create 3 new Phase 3 fixtures (`sum_mean_pathological`, `softmax_extreme_logits`, `logsumexp_simple`) and a new `AccuracyHardeningTest` class mirroring all 4 existing GoldenReferenceTest tests + 3 new ones = 7 tests, all hard-fail on tolerance miss. GoldenReferenceTest continues to soft-fail alongside.

## Tasks Completed

### Task 03-03-01: 3 new golden fixtures (commit 3e209b3)

**`bench/src/test/resources/golden/sum_mean_pathological.json`**
```json
{"op":"sum_mean","magnitude_class":"extreme_cancellation","seed":45,"input_gen":"double[1000000]: alternating [1e15, 1e-15] starting at 1e15","expected":{"sum":5e20,"mean":5e14},"tolerance_rel":1e-13}
```
Math: alternating [1e15, 1e-15] × 500_000 pairs = 5e20 (naive left-to-right sum loses the 1e-15 terms to catastrophic cancellation). Kahan recovers them.

**`bench/src/test/resources/golden/softmax_extreme_logits.json`**
```json
{"op":"softmax_extreme_logits","seed":46,"input_gen":"double[1000]: java.util.Random.nextDouble()*2e300-1e300 per element","magnitude_class":"extreme_overflow","tolerance_rel":1e-13}
```
Stability + normalization gate — no per-element expected values; test asserts all outputs finite AND sum ≈ 1.0.

**`bench/src/test/resources/golden/logsumexp_simple.json`**
```json
{"op":"logsumexp_simple","seed":47,"input_gen":"double[500]: java.util.Random.nextDouble()*100.0-50.0 per element","tolerance_rel":1e-14,"theoretical_upper_bound":56.7}
```
Self-consistency + stability gate. theoretical_upper_bound = 50 + log(500) ≈ 56.7.

### Task 03-03-02: AccuracyHardeningTest (commit ca0306a)

**`modules/sklearn/src/test/java/sklearn/accuracy/AccuracyHardeningTest.java`** — 7 hard-fail tests, `@Category(GoldenReferenceTest.Golden.class)`:

1. `matmul256_withinTolerance` — mirrors GoldenReferenceTest.matmul256_vs_numpy, calls production NumJa.matmul + assertMaxErrWithinTolerance.
2. `sum_mean_withinTolerance` — mirrors sum_mean_vs_numpy, asserts both sum and mean within relErr 1e-13.
3. `softmax_extreme_withinTolerance` — calls `Activations.softmax(x)` (now delegates to NumericStable.softmax) — production code path; maxErr against the 1000-element expected array.
4. `linreg_iris_withinTolerance` — mirrors linear_regression_iris_vs_numpy; copies `readIrisFeatures()` helper.
5. `sum_mean_pathological_withinTolerance` — `ParallelOps.sum(data)` on alternating [1e15, 1e-15] at n=1e6 against fixture expected {sum:5e20, mean:5e14} within relErr 1e-13.
6. `softmax_extreme_logits_withinTolerance` — `Activations.softmax(x)` on ±1e300 inputs; asserts every output finite + normalization invariant (sum ≈ 1.0 within 1e-13).
7. `logsumexp_simple_withinTolerance` — `NumericStable.logSumExp(x)` on mixed-magnitude inputs; asserts finite + ≤ theoretical upper bound + ≥ min input; self-consistency (two calls agree within relErr 1e-14).

No soft-fail REPORT buffer. No `@Before`/`@After`/`@BeforeClass` setup. Pure hard-fail JUnit assertions.

## Verification

```bash
$ mvn -pl modules/sklearn -am test -Dtest=AccuracyHardeningTest
Tests run: 7, Failures: 0, Errors: 0, Skipped: 0
$ mvn -pl modules/sklearn -am test  (full sklearn suite)
Tests run: 13, Failures: 0, Errors: 0, Skipped: 1
```

Both classes coexist:
- `AccuracyHardeningTest` (7 tests) — hard-fail on tolerance miss
- `GoldenReferenceTest` (5 tests = 4 fixtures + 1 `@Ignore`d printReport) — soft-fail PASS lines emitted

## Deviations from PLAN.md

- **`readIrisFeatures()` return type fix**: PLAN.md followed the GoldenReferenceTest pattern that incorrectly declared `private static double[] readIrisFeatures()`. Compiler error `double[] cannot be converted to double[][]` exposed the bug. Fixed to `double[][]`. GoldenReferenceTest itself has the same bug but is never actually called from there (only the iris.csv path is exercised); left as-is since it's not in scope for Phase 3.
- **`sum_mean_pathological.json` expected value**: PLAN.md proposed `5.000000000000005e20` (including the 1e-15 noise bump). Simplified to `5e20` — the 1e-15 terms contribute 5e-10 total, far below the 1e-13 relative tolerance against 5e20. Naive sum loses the 1e-15 terms entirely (catastrophic cancellation in double precision: 1e15 + 1e-15 = 1e15); Kahan recovers them but they're so small the relative error from `5e20` to `5e20 + 5e-10` is 1e-30, well within tolerance.
- **`softmax_extreme_logits.json` expected values**: PLAN.md allowed either storing all 1000 expected floats or using a max-err gate. Chose max-err gate (no `expected` field) since the test verifies stability + normalization invariant rather than per-element precision — at ±1e300 logits, the expected NumPy values are sensitive to bit-level RNG differences across runtimes.
- **`logsumexp_simple.json` reference**: PLAN.md allowed either NumPy reference or self-consistency. Chose self-consistency + stability gate (no NumPy `expected` field) since no Python runtime is available to compute the reference offline.

## Key Files Created

| File | Change | Lines |
|------|--------|-------|
| `bench/src/test/resources/golden/sum_mean_pathological.json` | NEW | +1 |
| `bench/src/test/resources/golden/softmax_extreme_logits.json` | NEW | +1 |
| `bench/src/test/resources/golden/logsumexp_simple.json` | NEW | +1 |
| `modules/sklearn/src/test/java/sklearn/accuracy/AccuracyHardeningTest.java` | NEW | +215 |

## Acceptance Criteria Status

All must-haves satisfied:

- ✅ AccuracyHardeningTest mirrors all 4 existing fixtures + 3 Phase 3 = 7 tests, all `@Category(Golden.class)`.
- ✅ 3 new fixtures committed: sum_mean_pathological (alternating [1e15, 1e-15], n=1e6), softmax_extreme_logits (±1e300, n=1000), logsumexp_simple (mixed magnitudes, n=500).
- ✅ Each test calls `GoldenFixtures.assertWithinTolerance` or `assertMaxErrWithinTolerance` — no REPORT buffer, hard-fail.
- ✅ GoldenReferenceTest soft-fails PASS for all 4 existing fixtures (verified).
- ✅ `mvn test` exits 0 with both classes present (13 tests, 0 failures, 1 pre-existing `@Ignore`).

## Commits

```
3e209b3 test(03-03): 3 new golden fixtures (pathological sum, extreme logits, logsumexp simple)
ca0306a test(03-03): AccuracyHardeningTest (7 hard-fail golden tests, ACC-03 gate)
```

## What's Next

Wave 4 plan 03-04: BENCH-03 regression gate. Build `scripts/check_regression.ps1`, write `03-baseline.json` extending Phase 2 measurements, run JMH re-baseline, commit `03-BASELINE-AFTER.md` + `03-VERIFICATION.md`.
