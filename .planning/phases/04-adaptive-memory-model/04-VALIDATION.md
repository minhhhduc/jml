---
phase: 04
slug: adaptive-memory-model
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-28
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 4.13.2 (locked from Phase 1/2/3 — NOT JUnit 5) |
| **Config file** | `pom.xml` per module (no surefire overrides needed) |
| **Quick run command** | `mvn -pl modules/pandas -am test "-Dtest=PandasStreamingTest" "-Dsurefire.failIfNoSpecifiedTests=false"` |
| **Full suite command** | `mvn test` from repo root |
| **Estimated runtime** | ~90 seconds (current full suite ~55 tests; Phase 4 adds ~25 tests across 5 new test classes) |

---

## Sampling Rate

- **After every task commit:** `mvn -pl modules/pandas -am test "-Dtest=PandasStreamingTest"` (streaming-specific changes) **or** `mvn -pl modules/sklearn -am test "-Dtest=GaussianNBPartialFitTest,PandasPipelineTest"` (model changes).
- **After every plan wave:** `mvn test` from repo root (full suite, ensure no regression in existing 55 tests).
- **Before `/gsd:verify-work`:** Full suite green + Phase 3 regression gate still passes (`powershell -ExecutionPolicy Bypass -File scripts/check_regression.ps1`) + cross-path equivalence verified.
- **Max feedback latency:** ~30 seconds for module-scoped test, ~90 seconds for full suite.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-W0-01 | 01 | 1 | MEM-01/02 | STRIDE-T405 | chunkRows validation; reject pathologically large values | unit | `mvn -pl modules/pandas -am test -Dtest=ChunkedReadOptionsTest` | ❌ W0 | ⬜ pending |
| 04-W0-02 | 01 | 1 | MEM-01/02 | STRIDE-T401/T403 | canonicalize path; AutoCloseable release on early break | unit | `mvn -pl modules/pandas -am test -Dtest=CsvChunkReaderTest` | ❌ W0 | ⬜ pending |
| 04-W0-03 | 02 | 1 | MEM-03 | — | sum/mean/count/min/max matches in-memory within `1e-9 * num_groups` | unit (cross-path) | `mvn -pl modules/pandas -am test -Dtest=RunningGroupAggregatorTest` | ❌ W0 | ⬜ pending |
| 04-W0-04 | 02 | 1 | MEM-03 | STRIDE-T406 | GaussianNB partial_fit accumulation consistent | unit (cross-path) | `mvn -pl modules/sklearn -am test -Dtest=GaussianNBPartialFitTest` | ❌ W0 | ⬜ pending |
| 04-W0-05 | 03 | 1 | USE-02 | — | 4-line caller pattern works; missing path → IAE | integration | `mvn -pl modules/sklearn -am test -Dtest=PandasPipelineTest` | ❌ W0 | ⬜ pending |
| 04-W0-06 | 01 | 1 | MEM-01/02/03 | STRIDE-T401/T402 | small file unchanged, large file chunks, formula cell neutralization | integration (cross-path) | `mvn -pl modules/pandas -am test -Dtest=PandasStreamingTest -DargLine="-Xmx128m"` | ❌ W0 | ⬜ pending |
| 04-W0-07 | 02 | 1 | MEM-01 | — | `grep -c "public static DataFrame read_csv"` Pandas.java unchanged | spot-check | shell grep | n/a | ⬜ pending |
| 04-W0-08 | 02 | 1 | API-2 | — | NumJa.java public static = 61 (frozen) | spot-check | `grep -c "public static" NumJa.java` | n/a | ⬜ pending |
| 04-W0-09 | 02 | 1 | API-3 | — | ArrayOps.java public static = 34 (frozen) | spot-check | `grep -c "public static" ArrayOps.java` | n/a | ⬜ pending |
| 04-W0-10 | 03 | 1 | REGRESSION | — | full suite green | integration | `mvn test` | n/a | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `modules/pandas/src/main/java/pandas/internal/ChunkedReadOptions.java` — NEW (chunkRows validation, defaults from maxMemory)
- [ ] `modules/pandas/src/main/java/pandas/internal/CsvChunkReader.java` — NEW (implements Iterator<DataFrame>, AutoCloseable; path canonicalization STRIDE-T401)
- [ ] `modules/pandas/src/main/java/pandas/internal/RunningGroupAggregator.java` — NEW (per-group sum/sumSq/count/min/max; WR-05 defensive identity init)
- [ ] `modules/pandas/src/main/java/pandas/Pandas.java` — additive: `+1 public read_csv_streaming(String, ChunkedReadOptions)` (existing 2 methods byte-identical)
- [ ] `modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java` — additive: `+1 public partial_fit(NDArray, int[])` + private `init()` refactor
- [ ] `modules/sklearn/src/main/java/sklearn/pipeline/PandasPipeline.java` — NEW public class (fluent builder: load().chunk().partialFit(model).run())
- [ ] `modules/pandas/src/test/java/com/pandas/internal/CsvChunkReaderTest.java` — NEW (≥6 tests: chunkCountMatchesCeil, close_releasesFileHandle, pathTraversal_canonicalized, etc.)
- [ ] `modules/pandas/src/test/java/com/pandas/internal/RunningGroupAggregatorTest.java` — NEW (≥6 tests: sum_matchesInMemory, mean_matchesInMemory, std_matchesInMemory, NaN_skip, chunkBoundaryAggregation)
- [ ] `modules/pandas/src/test/java/com/pandas/PandasStreamingTest.java` — NEW (≥4 tests: smallFile_inMemoryPathUnchanged, largeFile_streamingFitsInSmallHeap -DargLine="-Xmx128m", csvFormulaCells_neutralized, chunkBoundary_correct)
- [ ] `modules/sklearn/src/test/java/sklearn/naive_bayes/GaussianNBPartialFitTest.java` — NEW (≥4 tests: twoPartialFits_equalOneFit, singlePartialFit_equalsFit, init_resetsState, init_idempotentAfterFirstCall)
- [ ] `modules/sklearn/src/test/java/sklearn/pipeline/PandasPipelineTest.java` — NEW (≥3 tests: tenLineCallerPattern, missingPath_throwsIAE, modelPredictsAfterRun)

*If all gaps filled: framework infra complete; implementation can begin.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Streaming behavior on truly >RAM dataset (>15GB) | MEM-02 | Cannot allocate 15GB on a 15.7GB dev machine | Synthetic 500MB CSV with `-Xmx128m` is the proxy; document as "verified at 500MB scale, scaling argued by algorithm" |
| JMH regression gate still catches Phase 4 streaming overhead | MEM-01/02 | Performance characteristic — gate is bench infrastructure | After Phase 4 plans land, run `powershell -ExecutionPolicy Bypass -File scripts/check_regression.ps1` — gate must stay green against `03-baseline.json` (Phase 4 streaming benchmarks added to `04-baseline.json` in Plan 04) |

*If none: "All other phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (10 test stubs/files above)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s for module-scoped test, < 90s for full suite
- [ ] `nyquist_compliant: true` set in frontmatter (after Wave 0 fills)

**Approval:** pending
