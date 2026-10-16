---
phase: 03-numerical-accuracy-hardening
plan: 01
type: execute
wave: 1
completed_at: 2026-08-27T21:19:50+07:00
status: complete
---

# Plan 03-01 SUMMARY

## Objective
Build the Wave-1 infrastructure for ACC-02: a single-source `numja.NumericStable` primitive class hosting stable softmax + logSoftmax + logSumExp, refactor the two existing sklearn consumers (`Activations.softmax`, `LogisticRegression.softmax`) to delegate to it, and create a package-private `GoldenFixtures` utility that extracts the `load()` + `check()` helpers out of `GoldenReferenceTest` so Plan 03 can build `AccuracyHardeningTest` against the same shared loader. No behavior change for in-range inputs.

## Tasks Completed

### Task 03-01-01: Create numja.NumericStable (commit 7e82b14)
Created `modules/numja/src/main/java/numja/NumericStable.java` — `public final class` with private constructor, sitting at the `numja` root alongside `NumJa.java` (not in `numja.core`, not in the facade). Three max-shift stable methods:

- `public static double[] softmax(double[] x)` — algorithm lifted verbatim from `Activations.java:89-107`. Subtract max, exp, sum, divide.
- `public static double[] logSoftmax(double[] x)` — uses `logSumExp(x)` for the denominator; returns `(x[i] - max) - logSumExp(x)`.
- `public static double logSumExp(double[] x)` — returns `Double.NEGATIVE_INFINITY` for empty input, otherwise `max + log(sum(exp(x[i] - max)))`.

No external dependencies; only `java.lang.Math`. **Facade public-API count verified unchanged: NumJa.java = 61, ArrayOps.java = 34 (D-07 satisfied).**

### Task 03-01-02: sklearn softmax consumers delegate (commit ddd7c8f)

**`Activations.java`** (modules/sklearn/src/main/java/sklearn/neural_network/Activations.java):
- Added `import numja.NumericStable;`
- Replaced inline `softmax` body with `return NumericStable.softmax(x);` (Javadoc preserved verbatim).
- All other methods (sigmoid, relu, tanh, linear, derivatives) untouched.

**`LogisticRegression.java`** (modules/sklearn/src/main/java/sklearn/linear_model/LogisticRegression.java):
- Added `import numja.NumericStable;`
- Chose **lazy option per task spec**: kept the private `softmax` method as a one-line delegator (`return NumericStable.softmax(x);`). `computeProbabilities` still routes through the local helper, diff minimal.

### Task 03-01-03: Extract GoldenFixtures (commit 37975ac)

Created `modules/sklearn/src/test/java/sklearn/accuracy/GoldenFixtures.java` — package-private utility (`class GoldenFixtures`, no `public` modifier, private constructor). Hosts:

- **Soft-fail helpers** (used by `GoldenReferenceTest`): `load(String)`, `relErr(double, double)`, `check(...)`, `checkMaxErr(...)`, `flushReport()`, `resetReport()`, `assertMechanicsOnly()`, `jsonToDoubleArray(JSONArray)`, plus the shared `REPORT` buffer + `GOLDEN_DIR` constant.
- **Hard-fail helpers** (used by `AccuracyHardeningTest` in Plan 03): `assertWithinTolerance(label, actual, expected, tolRel)` and `assertMaxErrWithinTolerance(label, maxErr, tolRel)`.

`GoldenReferenceTest` refactored to delegate to `GoldenFixtures.load`/`check`/`checkMaxErr`/`resetReport`/`assertMechanicsOnly`/`jsonToDoubleArray`/`flushReport`/`relErr`. The `Golden` marker interface and `@Category(GoldenReferenceTest.Golden.class)` annotation are preserved. The `@Ignore`d `printReport` test still exists for manual golden-report dumping.

## Verification

```bash
$ mvn -pl modules/sklearn -am test
Tests run: 6, Failures: 0, Errors: 0, Skipped: 1
BUILD SUCCESS
```

- `GoldenReferenceTest` still soft-fails all 5 (4 fixtures + 1 `@Ignore`d `printReport`); `[GOLDEN] PASS` lines emitted to stdout and `target/golden-report.txt`.
- 1 pre-existing `@Ignore` preserved.
- Full sklearn suite green (no regressions from NumericStable or delegation).
- `NumJa.java` `public static` count = 61 (unchanged).
- `ArrayOps.java` `public static` count = 34 (unchanged).

## Key Files Created / Modified

| File | Change | Lines |
|------|--------|-------|
| `modules/numja/src/main/java/numja/NumericStable.java` | NEW | +78 |
| `modules/sklearn/src/main/java/sklearn/neural_network/Activations.java` | MODIFY (delegate) | -16 / +2 |
| `modules/sklearn/src/main/java/sklearn/linear_model/LogisticRegression.java` | MODIFY (delegate) | -15 / +2 |
| `modules/sklearn/src/test/java/sklearn/accuracy/GoldenFixtures.java` | NEW | +99 |
| `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java` | MODIFY (refactor) | -61 / +25 |

## Decisions & Deviations

- **LogisticRegression lazy option applied**: kept private `softmax` as a one-line delegator (matches "lazy option" in task spec; minimal diff).
- **GoldenFixtures assertions stay package-private** (no `public` modifier) — only `sklearn.accuracy` test classes can call them; matches D-08 + D-11.
- **Hard-fail assertions live in GoldenFixtures** alongside soft-fail — single source of truth for both pass-recording and gate-keeping (D-08 contract).

## Acceptance Criteria Status

All must-haves satisfied:

- ✅ `NumericStable.softmax(double[])` returns same values as inline stable softmax on in-range inputs (algorithm lifted verbatim — bit-identical).
- ✅ `Activations.softmax` and `LogisticRegression.computeProbabilities` both delegate to `NumericStable.softmax`.
- ✅ `NumericStable.softmax/logSoftmax/logSumExp` handle ±1e300 inputs (max-shift: subtract max → `exp(0)` or `exp(negative)` → finite; no NaN/Inf).
- ✅ `GoldenFixtures.load(name)` returns same `JSONObject` as `GoldenReferenceTest.load(name)` — body copied verbatim.
- ✅ `GoldenFixtures.check` records PASS/FAIL to per-call buffer exactly as `GoldenReferenceTest.check` did.
- ✅ `NumericStable` does NOT appear in `NumJa` facade (count frozen at 61).

## Commits

```
7e82b14 feat(03-01): add NumericStable with stable softmax/logSoftmax/logSumExp
ddd7c8f refactor(03-01): sklearn softmax consumers delegate to NumericStable
37975ac test(03-01): extract GoldenFixtures shared loader + recorder from GoldenReferenceTest
```

## What's Next

Wave 2 plan 03-02: ACC-01 compensated reduce (per-leaf Kahan in `ParallelOps.SumTask`, log-sum-exp `ParallelOps.prod`, `NDArray.prod()` parallelization, `ParallelCompensationTest`). Depends on `NumericStable` for logSumExp usage in `prod()` if planner wired it that way.
