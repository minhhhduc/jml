---
gsd_state_version: 1.0
milestone: v0.3.0
milestone_name: milestone
status: unknown
last_updated: "2026-08-27T13:30:00.000Z"
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
  percent: 33
---

# STATE: NumJa Performance & Scalability

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-25)

**Core value:** Xử lý dữ liệu lớn nhanh hơn và chính xác hơn mà API không thay đổi
**Current focus:** Phase 3 — Numerical Accuracy Hardening [CONTEXT GATHERED]

## Current Position

- **Milestone:** v0.3.0 Performance & Scalability (first)
- **Phase:** 1 of 6 — **✅ CLOSED (verified GO 2026-08-26)** — plans 01-01..01-03 + review fixes (`59343b8`) + VERIFICATION.md
- **Phase 2:** **✅ CLOSED (verified GO 2026-08-27)** — plans 02-01..02-04 + code-review fixes (5 WRs in `28e7352..1d05d30`) + VERIFICATION.md + 02-BASELINE-AFTER.md. CPU-01 (add 10^7 = 2.43x, multiply 10^7 = 1.42x FLAGGED) + CPU-02 (sum 10^7 = 2.63x, mean 10^7 = 5.45x) PASS; GoldenReferenceTest sum/mean err ~2.6e-15. CPU-03 deferred.
- **Phase 3:** **CONTEXT GATHERED (2026-08-27)** — `f153e71` (03-CONTEXT.md + 03-DISCUSSION-LOG.md). Decisions: per-leaf Kahan + log-sum-exp prod compensation (ACC-01); `numja.NumericStable` public-not-in-facade softmax/logSoftmax/logSumExp extraction (ACC-02); `AccuracyHardeningTest` hard-fail mirroring 4 existing + 3 new fixtures (ACC-03); `scripts/check_regression.ps1` + `03-baseline.json` extending Phase 2 (BENCH-03).
- **Branch:** `dev`
- **Next command:** `/gsd-plan-phase 3`

## Key Context for Future Sessions

- Source KHÔNG có trên nhánh main (gitignore /modules/) — luôn làm việc trên `dev`
- Codebase map: `.planning/codebase/` (7 docs)
- Research summary: `.planning/research/SUMMARY.md` + **VERIFIED VERSIONS tại `.planning/phases/01-baseline-benchmark/VERSIONS.md`**
- Baseline numbers: `.planning/phases/01-baseline-benchmark/BASELINE.md` — matmul 1024²~1s/op, elementwise 10⁷ ~40ms, reduce ~11ms
- Phase 2 after-numbers: `.planning/phases/02-cpu-parallel-core-threads/02-BASELINE-AFTER.md` — add elementwise 10^7 = 36.86ms (2.43x), sum_reduce 10^7 = 10.33ms (2.63x), mean_reduce 10^7 = 7.84ms (5.45x). multiply_elementwise 10^7 = 69.73ms (1.42x — EJML SIMD ceiling, Phase 3 follow-up).
- Parallel infra: `ThreadPoolConfig.getInstance().getForkJoinPool()` (singleton, sized to 60% cores) + `numja.core.ParallelOps` (THRESHOLD=100_000, LEAF_CUTOFF=16_384, RecursiveAction tree). Test threshold hook `setThresholdForTesting`/`resetThresholdForTesting` is package-private. Production code is bit-identical below THRESHOLD.
- Accuracy gaps: **không còn gap blocking** — golden tests all PASS: matmul 2.0e-15, sum/mean ~2.6e-15 (tol 1e-13), softmax ref-only 4.3e-16, linreg ≤3.2e-13. Kahan summation + tree-reduce improvements deferred to Phase 3 (ACC-01).
- Code review: Phase 1 REVIEW.md (12 findings fixed in `59343b8`) + Phase 2 REVIEW.md (0 critical / 5 WR / 6 info — all WRs fixed in `28e7352..1d05d30`)
- Build: Maven 3.9.15 tại `C:\Users\Admin\.maven\maven-3.9.15\bin` (**không trên PATH** — prefix `$env:Path = '...\bin;' + $env:Path`)
- Benchmarks: `mvn -pl bench -am package -DskipTests` rồi `java -jar bench/target/benchmarks.jar` từ repo root
- Golden tests: `mvn -pl modules/sklearn -am test "-Dtest=GoldenReferenceTest" "-Dsurefire.failIfNoSpecifiedTests=false"`
- Tests: full suite xanh (35 tests, 0 failures, 1 pre-existing @Ignore on GoldenReferenceTest). JUnit 4. Phase 2 added 28 new tests across 6 classes (ThreadPoolConfigTest, ParallelOpsTest, ParallelElementwiseTest, ParallelReduceTest, ParallelRegressionTest).
- API công khai bị đóng băng (v0.2.0) — mọi tối ưu là internal. NumJa.java = 61 public static, ArrayOps.java = 34 public static (both unchanged through Phase 2).

## Decisions Log

See PROJECT.md ## Key Decisions.

## Blockers

- None known.
