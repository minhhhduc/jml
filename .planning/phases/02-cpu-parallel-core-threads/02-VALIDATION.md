---
phase: 2
slug: cpu-parallel-core-threads
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-08-27
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `02-RESEARCH.md` Validation Architecture section.

---

## Test Infrastructure
| Property | Value |
|----------|-------|
| **Framework** | JUnit 4 (existing in `modules/numja/pom.xml:37`) + JMH 1.37 (existing in `bench/pom.xml:22`) |
| **Config file** | `pom.xml` per module; surefire via `mvn -pl modules/X -am test -Dtest=ClassName` |
| **Quick run command** | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest -DfailIfNoTests=false` |
| **Full suite command** | `mvn test` from repo root (8+5 tests per STATE.md) |
| **Estimated runtime** | ~60 seconds for ParallelRegressionTest; ~5–10 min for full JMH run |

---

## Sampling Rate

- **After every task commit:** Run `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest -DfailIfNoTests=false`
- **After every plan wave:** Run full `mvn test` from repo root + targeted JMH subset (`-wi 3 -i 3 -w 1s -r 1s` for new small + large array benchmarks)
- **Before `/gsd:verify-work`:** Full suite green AND `02-BASELINE-AFTER.md` recorded with ≥2x speedup on ≥10⁶ elements and <10% regression on ≤100k elements
- **Max feedback latency:** ~60 seconds (ParallelRegressionTest only) / ~10 min (full JMH)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | CPU-01 | T-2-01 (DoS) | Threshold gate prevents tiny-parallel DoS | unit | `mvn -pl modules/numja -am test -Dtest=ParallelOpsTest` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | CPU-01 | T-2-02 (PoolExhaustion) | Singleton FJP reused, no per-call allocation | unit | `mvn -pl modules/numja -am test -Dtest=ThreadPoolConfigTest#getForkJoinPool_isSingleton` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | CPU-01 | — | Elementwise binary ops (add/mul/sub/div) parallel path correctness | unit | `mvn -pl modules/numja -am test -Dtest=ParallelElementwiseTest` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | CPU-01 | — | Large array ≥2x speedup vs sequential | unit + bench | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest#largeArray_isAtLeast2xFaster` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 1 | CPU-01 | — | Small array (10k, 100k) <10% regression | unit | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest#smallArray_under10PercentRegression` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | CPU-02 | — | Reduce ops (sum/mean/min/max) parallel path correctness | unit | `mvn -pl modules/sklearn -am test -Dtest=GoldenReferenceTest` | ✅ existing | ⬜ pending |
| 02-03-02 | 03 | 2 | CPU-02 | — | Reduce numerical stability within relErr ≤1e-13 vs sequential | unit (reuses golden) | `mvn -pl modules/sklearn -am test -Dtest=GoldenReferenceTest` | ✅ existing | ⬜ pending |
| 02-03-03 | 03 | 2 | CPU-02 | — | Reduce ops ≥2x speedup large array | unit + bench | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest#reduceLarge_isAtLeast2xFaster` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 3 | (no req) | — | Full suite still passes — no regression in existing 8+5 tests | integration | `mvn test` | ✅ existing | ⬜ pending |
| 02-04-02 | 04 | 3 | (no req) | — | Fresh JMH baseline committed with ≥2x on ≥10⁶, <10% on ≤100k | bench | `java -jar bench/target/benchmarks.jar -f 1 -wi 3 -i 3 -w 1s -r 1s` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `modules/numja/src/main/java/numja/core/ParallelOps.java` — ForkJoin elementwise + sum/min/max
- [ ] `modules/numja/src/main/java/numja/config/ThreadPoolConfig.java` — add `getForkJoinPool()` singleton accessor
- [ ] `modules/numja/src/test/java/numja/core/ParallelOpsTest.java` — stubs: threshold gate, singleton pool, sequential fallback below THRESHOLD
- [ ] `modules/numja/src/test/java/numja/core/ParallelElementwiseTest.java` — elementwise correctness add/mul/sub/div
- [ ] `modules/numja/src/test/java/numja/core/ParallelRegressionTest.java` — JUnit assertions for ≥2x and <10% regression
- [ ] `bench/src/main/java/bench/CoreBench.java` — extend with `SmallArrayState` (10k, 100k) + `add_elementwise_small` benchmark
- [ ] `02-BASELINE-AFTER.md` — fresh JMH numbers proving success criteria, methodology section on hybrid-core variance

*If Wave 0 incomplete, plans will fall back to "existing infrastructure covers" (which is false for CPU-01/02 — there are NO automated tests for parallelism yet).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Hybrid P/E-core speedup variance on i7-1255U | CPU-01 | Benchmark variance dominated by Windows scheduler + thread migration; can't automate "same power profile as baseline run" | Run same-session baseline + Phase 2 JMH at `-f 3 -wi 5 -i 5`, take median. Document methodology in `02-BASELINE-AFTER.md`. |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s (ParallelRegressionTest only) / <10min (full JMH)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
