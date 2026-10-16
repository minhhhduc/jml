---
gsd_state_version: "1.0"
milestone: v0.3.0
current_phase: 08
current_phase_name: clean-hot-paths
status: active
stopped_at: Phase 8 re-scoped to intermediate reduction 2026-10-15
last_updated: "2026-10-02T09:32:45.211Z"
state_head: ebb4f55072dbce95a5bb1332d1209a667c79e77d
progress:
  total_phases: 9
  completed_phases: 7
  total_plans: 20
  completed_plans: 20
milestone_name: milestone
---

# STATE: NumJa Performance & Scalability

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-25)

**Core value:** Xử lý dữ liệu lớn nhanh hơn và chính xác hơn mà API không thay đổi
**Current focus:** Phase 08 Clean Hot Paths — re-scoped 2026-10-15 per user (clean code + cut intermediate steps); GPU HAL integration deferred

## Current Position

Phase: 08 (performance-optimization) — PLANNING

- **Milestone:** v0.3.0 Performance & Scalability
- **Phase:** Phases 1–7 all CLOSED. Phase 6 DROPPED 2026-10-02 per user instruction (skipped; requirements USE-01 + CPU-03 left unmapped). Phase 8 is the active phase.
- **Phase 5:** **✅ CLOSED (2026-09-13, via Phase 7 evidence)** — HW-01/HW-02 complete (`ComputeBackend` + `CpuThreadBackend` + `BackendSelector`, 3/3 tests green). HW-03 closed via Phase 7's measured GO, NOT the original TornadoVM path: TornadoVM POC scaffolded in `bench/tornado-poc` but unrun on GPU (CUDA 13 driver breakage; issue #710 workaround documented); no `05-GPU-POC-RESULTS.md` exists — Phase 7 superseded evidence collection. GPU production integration deferred pending the GO caveats below.
- **Phase 6:** **⏭️ DROPPED (2026-10-02, per user instruction)** — Usability Polish & Release skipped; USE-01 + CPU-03 unmapped. See ROADMAP.md.
- **Phase 7:** **✅ DONE (2026-09-13, measured GO)** — JCuda 12.6 + cuBLAS Dgemm on Colab T4 at N=4096: `speedup_ratio=122.75`, `transfer_pct=17.33%`, `cpu_vs_gpu_frob_rel_err=9.85e-17`, `cuda_alloc_delta_mib=384.0` (in-process, = expected). GO scope caveats: 2-vCPU Colab CPU baseline, only N=4096 measured (dispatch threshold unmeasured), T4 not a deployment target. Production modules remain JCuda-free; no HAL integration yet.
- **Branch:** `dev`

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
- Tests: full suite xanh (90 tests, 0 failures, 1 pre-existing @Ignore on GoldenReferenceTest). JUnit 4. Phase 4 added 35 tests across 6 classes (ChunkedReadOptionsTest 6 + CsvChunkReaderTest 7 + RunningGroupAggregatorTest 8 + PandasStreamingTest 5 + GaussianNBPartialFitTest 5 + PandasPipelineTest 4). Phase 3 added 15 tests (ParallelCompensationTest 8, AccuracyHardeningTest 7). Phase 2 added 28 new tests across 6 classes (ThreadPoolConfigTest, ParallelOpsTest, ParallelElementwiseTest, ParallelReduceTest, ParallelRegressionTest).
- API công khai bị đóng băng (v0.2.0) — mọi tối ưu là internal. NumJa.java = 61 public static, ArrayOps.java = 34 public static (both unchanged). Pandas.java = 11 (+1 read_csv_streaming, Phase 4). GaussianNB.java = +2 partial_fit + finalize_fit (Phase 4).

## Decisions Log

See PROJECT.md ## Key Decisions.

## Blockers

- None known.

## Session

**Last session:** 2026-10-02T09:32:43.697Z
**Stopped at:** Phase 8 context gathered
**Resume file:** .planning/phases/08-performance-optimization/08-CONTEXT.md
