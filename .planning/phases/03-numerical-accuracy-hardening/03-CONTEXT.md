# Phase 3: Numerical Accuracy Hardening - Context

**Gathered:** 2026-08-27
**Status:** Ready for planning

## Phase Boundary

Deliver stable numerics for large reduce (Kahan/compensation), stable softmax/log primitives, golden-value accuracy gates (hard-fail), and a perf+accuracy regression gate. Public API stays frozen at v0.2.0 (NumJa.java = 61 public static, ArrayOps.java = 34 public static) — all changes are internal additions, delegation refactors, or test additions. Scope covers ACC-01 (compensated reduce), ACC-02 (stable softmax/log), ACC-03 (golden suite hard-fail), BENCH-03 (regression gate).

## Implementation Decisions

### Kahan placement (ACC-01)
- **D-01:** Per-leaf Kahan summation only in `ParallelOps.SumTask.compute()`. Leaf loop accumulates running sum `s` with compensation `c` via the standard `y = data[i] - c; t = s + y; c = (t - s) - y; s = t;` pattern. Tree merge stays naive `this.total = left.total + right.total` — tree structure unchanged from Phase 2. Per-leaf cost ~5% vs naive, error bound O(log n * eps).
- **D-02:** Compensate `ParallelOps.prod()` via log-sum-exp: `sign * exp(sum(log(|x[i]))))` where the log-accumulation uses per-leaf Kahan (same pattern as D-01) and the sign is tracked by parity of negative inputs. Merge at tree level stays naive `left.total + right.total`. Handles mixed magnitudes + negatives cleanly; final `exp` is a single amplification step.
- **D-03:** `sum()` / `mean()` / `prod()` callers in NDArray stay unchanged (signatures same). Sum/mean already route through `ParallelOps.sum` (Phase 2). `NDArray.prod()` is currently a sequential loop (Phase 2 IN-03) — Phase 3 owns both the parallelization and the compensation in one wave.

### Softmax surface (ACC-02)
- **D-04:** Create `numja.NumericStable` (public final class, package `numja`, NOT in NumJa facade). Private constructor; static methods `softmax(double[])`, `logSoftmax(double[])`, `logSumExp(double[])`. All three are max-shift stable (subtract max before any `Math.exp`).
- **D-05:** Extract existing max-shift stable logic from `sklearn.neural_network.Activations.softmax` and `sklearn.linear_model.LogisticRegression.softmax` into `NumericStable.softmax`. Both consumers replace their inline implementation with `NumericStable.softmax(x)`. Zero behavior change for in-range inputs (already stable); single source of truth going forward.
- **D-06:** `logSoftmax` and `logSumExp` are new primitives. No current sklearn consumer needs them today, but ACC-02 success criterion mentions "log stable" — covering both forms locks the requirement. Phase 4+ groupby aggregations will reuse `logSumExp`.
- **D-07:** Public API delta: `NumJa.java` still 61 public static (no changes). `ArrayOps.java` still 34 public static (no changes). `NumericStable` is reachable by class import but not added to the `import static numja.NumJa.*` facade.

### Golden suite semantics (ACC-03)
- **D-08:** Create `AccuracyHardeningTest` (hard-fail) in `modules/sklearn/src/test/java/sklearn/accuracy/`. Uses the same `load()` + `check()` helpers as `GoldenReferenceTest` — extract helpers to a package-private `GoldenFixtures` utility to avoid duplication. Each test calls `assertTrue(err <= tol)` directly — no REPORT buffer.
- **D-09:** `AccuracyHardeningTest` mirrors all 4 existing tests (`matmul256_withinTolerance`, `sum_mean_withinTolerance`, `softmax_extreme_withinTolerance`, `linreg_iris_withinTolerance`) PLUS 3 Phase 3 additions (`sum_mean_pathological_withinTolerance`, `softmax_extreme_logits_withinTolerance`, `logsumexp_simple_withinTolerance`) = 7 tests total. Annotation `@Category(Golden.class)` for filter parity.
- **D-10:** `GoldenReferenceTest` keeps its soft-fail REPORT-buffer behavior — diagnostic only, human-eyeballed via `target/golden-report.txt`. Both classes coexist: `mvn test` runs both; `GoldenReferenceTest` never throws, `AccuracyHardeningTest` always throws on tolerance miss.
- **D-11:** Three new golden fixtures committed alongside Phase 3 changes:
  - `bench/src/test/resources/golden/sum_mean_pathological.json` — `magnitude_class: extreme_cancellation`, alternating `[1e15, 1e-15]` pattern, n=10⁶, compare against `math.fsum` reference
  - `bench/src/test/resources/golden/softmax_extreme_logits.json` — values in `[-1e300, 1e300]`, n=1000, called via `Activations.softmax`
  - `bench/src/test/resources/golden/logsumexp_simple.json` — mixed magnitudes, n=500, called via `NumericStable.logSumExp`

### Regression gate placement (BENCH-03)
- **D-12:** New `scripts/check_regression.ps1` PowerShell script. Sequence: build JMH jar → run with `-rf csv -rff target/jmh-phase3.csv` → parse CSV → diff each `(benchmark, score, error)` against `03-baseline.json` thresholds → exit non-zero on perf regression > 15% or accuracy > tolerance. Manual invocation by developer; matches existing `build_core.ps1` / `run_example.ps1` pattern.
- **D-13:** `.planning/phases/03-numerical-accuracy-hardening/03-baseline.json` extends Phase 2's measurement set with Phase 3 additions. Phase 2 numbers (add_elementwise 10⁷ = 36.86ms, sum_reduce 10⁷ = 10.33ms, etc.) carry forward with their existing tolerance windows. New entries:
  - `softmax_extreme_logits` — `max_rel_err: 1e-13`
  - `logsumexp_simple` — `max_rel_err: 1e-14`
  - `sum_pathological_cancellation` — `max_rel_err: 1e-13`
- **D-14:** Closed-source repo (modules/ git-ignored, no CI workflow exists). The "chạy được" success criterion is satisfied by manual `pwsh scripts/check_regression.ps1` invocation. No failsafe plugin, no GitHub Actions — keeping the script thin matches the project's distribution model.

### Claude's Discretion
- Exact threshold values inside per-test `tolerance_rel` (the JSON schema is locked; the numeric value is per-fixture and follows the Phase 1 baseline pattern of `~10x machine epsilon`)
- Whether to refactor `GoldenReferenceTest` to call the same shared `GoldenFixtures` helper, or leave its `load/check` inline (D-08 says "extract" — the helpers exist already in GoldenReferenceTest, so refactor is forced)
- Whether `sum()` op in `ParallelOps` calls into a `compensatedSum()` method or inlines the leaf loop (D-01 says "per-leaf" — placement is locked, method extraction is internal style)

### Folded Todos
None — `cross_reference_todos` found no phase-3-matching pending todos.

## Canonical References

Downstream agents MUST read these before planning or implementing.

### Phase boundary + scope
- `.planning/ROADMAP.md` §Phase 3 — success criteria, plan structure, results section to update
- `.planning/REQUIREMENTS.md` §ACC-01, §ACC-02, §ACC-03, §BENCH-03 — formal requirement text
- `.planning/PROJECT.md` §Active requirements (PERF-05, PERF-07) — milestone-level framing

### Prior phase artifacts (load-bearing context)
- `.planning/phases/02-cpu-parallel-core-threads/02-BASELINE-AFTER.md` — Phase 2 perf numbers that `03-baseline.json` extends
- `.planning/phases/02-cpu-parallel-core-threads/02-REVIEW.md` — IN-06 (`sum` pathological inputs, deferred to Phase 3), IN-03 (`prod()` out of Phase 2 scope), WR-05 (defensive-coding pattern for new task classes)
- `.planning/phases/02-cpu-parallel-core-threads/02-VERIFICATION.md` — 22/22 must-haves proof pattern Phase 3 inherits
- `.planning/phases/01-baseline-benchmark/BASELINE.md` — Phase 1 baseline numbers that feed Phase 3's golden fixtures
- `.planning/research/SUMMARY.md` §Pitfall 5 (numerics + parallelism) and §Stack (Kahan summation, max-shift softmax)

### Codebase architecture (carry-forward decisions)
- `.planning/codebase/ARCHITECTURE.md` — modular layered library, numja = core, sklearn depends on numja + pandas
- `.planning/codebase/CONCERNS.md` §1 (modules/ git-ignored — work on `dev` branch only), §2 (no automated tests pre-Phase 1 — Phase 1+ owns the test suite)
- `.planning/codebase/STACK.md` — JDK 11+ target, EJML 0.43.1 vendored, no Maven in main build (but `pom.xml` exists on `dev` for test execution)

### Reference implementations
- `modules/numja/src/main/java/numja/core/ParallelOps.java` — Phase 2 task classes, threshold gate pattern, double-checked locking singleton
- `modules/numja/src/main/java/numja/core/NDArray.java` — every elementwise op already routed through ParallelOps; `prod()` still sequential
- `modules/sklearn/src/main/java/sklearn/neural_network/Activations.java:89` — existing max-shift stable softmax
- `modules/sklearn/src/main/java/sklearn/linear_model/LogisticRegression.java:170` — existing max-shift stable softmax (private method)
- `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java` — soft-fail pattern, `load()` + `check()` helpers, REPORT buffer
- `bench/src/test/resources/golden/*.json` — existing fixtures (matmul_256, sum_mean, softmax_extreme, linear_regression_iris) that AccuracyHardeningTest mirrors

## Existing Code Insights

### Reusable Assets
- **`ParallelOps.elementwiseBinary` / `sum` / `min` / `max`** (Phase 2) — pattern applies directly to D-01's per-leaf Kahan and D-02's per-leaf log-Kahan. Tree structure of `SumTask` already splits at LEAF_CUTOFF=16_384; the compensated loop replaces the uncompensated leaf block.
- **`ThreadPoolConfig.getInstance().getForkJoinPool()`** (Phase 2) — singleton FJP sized to 60% cores. New `compensatedSum` / `compensatedProd` / `compensatedMean` use the same dispatch.
- **`GoldenReferenceTest.load()` + `check()`** — soft-fail fixture loader + helper. Extract to `GoldenFixtures` utility so `AccuracyHardeningTest` shares the same loader; the assertion is the only difference.
- **WR-05 defensive-coding pattern** — explicit identity init in MinTask/MaxTask (`best = Double.MAX_VALUE` / `-Double.MAX_VALUE`). Same lens applies to Kahan leaf init (compensation `c = 0.0` is correct identity for Kahan, but document it).

### Established Patterns
- **Per-method threshold gate** — every NDArray op checks `n < THRESHOLD` (100_000) before going FJP. Phase 3 inherits this; compensated versions replace the uncompensated leaf block but keep the threshold gate.
- **Package-private testing hooks** (`setThresholdForTesting` / `resetThresholdForTesting`) with `@After reset()` in test classes — same pattern applies if Phase 3 needs to override leaf size for stress testing.
- **Single-JVM test execution** — `static volatile int testThresholdOverride = -1;` relies on Surefire single-fork. New test hooks follow the same model.
- **JMH class-level annotations** (`@Fork(1)` + `@Warmup(5,1)` + `@Measurement(5,1)`) preserved through Phase 2 — Phase 3 BENCH-03 doesn't add JMH benchmarks, only re-runs the existing ones via `check_regression.ps1`.

### Integration Points
- **NDArray.sum() / mean() / prod()** — already call into ParallelOps (sum) or sequential loop (mean delegates to sum, prod is sequential). Phase 3 changes the leaf loop body, not the call sites.
- **Activations.softmax + LogisticRegression.softmax** — Phase 3 changes both call sites to delegate to NumericStable.softmax; the public signatures of Activations and LogisticRegression stay unchanged (no breaking change).
- **NumJa.java facade** — no entry added. NumericStable.softmax is reachable via direct class import only.
- **GoldenReferenceTest** — soft-fail preserved; AccuracyHardeningTest added alongside.
- **scripts/build_core.ps1** — already builds JMH jar via `mvn -pl bench -am package`. check_regression.ps1 calls the same target.

## Specific Ideas

- **Free-text user note during Softmax wiring question**: "tận dụng được code thì tốt" (reuse what we have) — confirmed via extracting existing stable softmax into NumericStable rather than writing fresh.
- **Per-leaf Kahan ~5% overhead** is an estimate — Phase 3 plan should run a quick JMH micro-benchmark on `sum_reduce 10^7` after the change to verify the actual overhead is within the 15% regression threshold (D-13). If actual overhead > 15%, push back to researcher/planner.
- **`math.fsum` is the Python reference for `sum_mean_pathological.json`**. Python's `math.fsum` uses full pairwise compensation (Shewchuk algorithm) — that's the strict reference for "what's correct". Per-leaf Kahan should come close enough for tolerance 1e-13 on the worst realistic inputs.

## Deferred Ideas

- **Vector API for hot loops (CPU-03)** — JDK 21+ Vector API still preview per VERSIONS.md; deferred by Phase 2, still deferred. Re-evaluate at end of milestone v0.3.0 if matmul/elementwise SIMD wins materialise.
- **EJML mt- variant for matmul/elementwise** — exists in EJML 0.43.x as `ejml-ddense-mt` artifact. Vendor decision deferred to Phase 5 (ComputeBackend abstraction); CPU threads via ParallelOps is the v0.3.0 path.
- **ComputeBackend abstraction (HW-01/02/03)** — Phase 5. Phase 3's NumericStable is a primitive utility, not a backend dispatch.
- **`multiply_elementwise` 10⁷ 1.42x ceiling** — flagged in 02-BASELINE-AFTER.md as EJML SIMD ceiling on i7-1255U. Phase 3 doesn't reopen it; Phase 5 (ComputeBackend POC) can route elementwise through EJML mt- variant.
- **`prod()` NaN handling semantics** — Phase 3 Kahan variant treats NaN inputs the same as the existing sequential prod (NaN in → NaN out, multiplication parity). Documented, no change.
- **Vector API for Kahan inner loop** — possible optimization for Phase 3+. Kahan itself is bandwidth-bound on the loop; SIMD would help leaf throughput. Deferred to Phase 5/6 if at all.

---

*Phase: 3-Numerical Accuracy Hardening*
*Context gathered: 2026-08-27*
