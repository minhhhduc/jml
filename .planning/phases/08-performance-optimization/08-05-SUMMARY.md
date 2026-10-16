# 08-05 Summary — NDArray.copyTo2D bulk extract, migrate hot callers

## What changed
- Added additive `NDArray.copyTo2D(double[][] out)` (`modules/numja/.../NDArray.java:589-608`):
  caller allocates `out[rows][cols]`, callee bulk-copies the row-major `data.data`
  buffer row by row via `System.arraycopy`. Rejects 1D arrays and buffer shape
  mismatches with `IllegalArgumentException`. No existing public method touched.
- Migrated plain-copy `extract2DArray` bodies (allocate + `copyTo2D`, signatures
  and call sites untouched, zero `arr.get(i,j)` loops remaining in these methods):
  - `LogisticRegression.java:221-229` (covers fit/predict/predictProba extracts)
  - `StandardScaler.java:136-144`
  - `KNeighbors.java:209-217`
  - `PCA.java:235-243`
- Files left out: none — all four listed extracts were plain copies with no extra
  logic (no scaling/masking). `KMeans.java` untouched per scope fence (owned by 08-04).

## Verification
- Task 1 targeted suite: `mvn -pl modules/sklearn -am test
  -Dtest=TestSklearnJUnit,TestSklearn,GaussianNBPartialFitTest,PandasPipelineTest,
  GoldenReferenceTest,AccuracyHardeningTest` → BUILD SUCCESS (22 run, 0 fail, 1 skipped);
  `copyTo2D` + `System.arraycopy` presence asserted via python.
- Task 2 spot-check (`C:\Users\Admin\AppData\Local\Temp\opencode\SpotCheck0805.java`,
  never in repo): copyTo2D ≡ toArray bit-for-bit; 1D + shape-mismatch guards throw;
  same-config LogisticRegression fits bit-identical weights (train acc 1.0);
  StandardScaler output mean ~3.5e-17; KNeighbors k=1 self-match on all 60 points.
  ALL SPOT CHECKS PASSED.
- Full suites: `mvn -pl modules/sklearn -am test` → BUILD SUCCESS all 4 modules
  (numja-core incl. ParallelRegressionTest, pandas-wrapper 27 tests, sklearn 22 tests,
  0 failures/errors). `mvn -pl modules/pandas test` → BUILD SUCCESS (27 tests).
  Note: one `ParallelRegressionTest.smallArray_at10k_thresholdGatePreserved` timing
  flake tripped on the first full run (parallel slower than sequential — env noise,
  unrelated to this change); reran the class alone → 4/4 pass, then the full reactor
  went green on retry.
- Diff hygiene: only the 5 listed files changed; `git diff` contains zero
  `-    public` lines (purely additive); KMeans untouched.

## Commit
`feat(08-05): NDArray.copyTo2D bulk extract, migrate hot callers`
