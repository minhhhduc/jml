# Phase 8: Performance Optimization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-02
**Phase:** 8-performance-optimization
**Areas discussed:** domain, cpu_strategy, cpu_03, gpu_integration, measurements, acceptance, constraints

---

## Domain (what Phase 8 optimizes)

| Option | Description | Selected |
|--------|-------------|----------|
| CPU throughput completion | Close the `multiply_elementwise` 1.42x ceiling Phase 2 flagged, re-verify CPU-03 Vector API against current JDK status, sweep pandas/sklearn hot paths still at Phase 1 numbers. Pure-internal, API frozen, no native deps. | |
| Close Phase 7 GO caveats + HAL integration | Measure the 3 unestablished Phase 7 prerequisites (10-core CPU baseline, N-sweep threshold, local/Windows GPU), then wire JCuda gemm into the idle `BackendSelector` behind a size threshold. Biggest win, but reverses Phase 7 D-03 (production modules JCuda-free). | |
| Measurement-first, defer integration | Run only the missing measurements, record a GO/NO-GO, defer HAL wiring to Phase 10+. Zero production code changes — but Phase 8 ships no optimization a user feels. | |
| Both CPU + GPU | Phase 8 spans both: finish CPU throughput AND close the Phase 7 caveats with HAL integration. Broadest value, roughly doubles the phase, mixes a pure-internal track with a native-dependency track — the planner will likely recommend a split. | ✓ |

**User's choice:** Both CPU + GPU
**Notes:** The phase goal was undefined coming in (`Goal: [To be planned]`, `Requirements: TBD`), so the domain itself was the first gray area. Every v1 optimization requirement was already closed, so the alternatives were materially different phases rather than variants of one. The user's standing framing for the phase was "optimize code"; the choice made it concrete.

---

## CPU ceiling strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Re-measure, then cheapest fix | Re-measure `multiply_elementwise` locally first. Optimize only if 1.42x still reproduces. Try the cheapest fix (FJP leaf-size tuning, or EJML delegation above threshold) and accept the result even if under 2x — the ceiling may be hardware-side, as Phase 2 concluded. | ✓ |
| Commit to 2x, Vector API if needed | Commit to ≥2x; if hand-tuned FJP/EJML can't reach it, adopt the Java Vector API — requiring `--release 11→21` for `modules/numja` and re-verifying the Java 11 compat claim the README advertises. | |
| Defer multiply, sweep pandas/sklearn | Don't chase multiply. Re-verify CPU-03 and sweep pandas/sklearn hot paths (`read_csv_california` 105ms, `linear_regression_fit` 20.7ms) for bigger aggregate wins. Document 1.42x as a hardware ceiling. | |

**User's choice:** Re-measure, then cheapest fix
**Notes:** Phase 2's own root-cause analysis was that EJML's `CommonOps_DDRM.multiply` is more aggressively SIMD-vectorized than the raw `double[]` loop, and that! the parallel path hits a memory-bandwidth ceiling on 80MB arrays. The user chose not to chase a target that may be unreachable without a compile-target change.

---

## CPU-03 Vector API

| Option | Description | Selected |
|--------|-------------|----------|
| Re-verify first, then decide | Have the researcher verify current JDK status before any decision. If JDK 26 (Mar 2026 target) or JDK 27 made Vector API non-preview, adopt with runtime version detection + pure-Java fallback; if still preview, keep it deferred. | ✓ |
| Keep deferred, no SIMD | Keep the deferral — VERSIONS.md's finding stands; the API changes between previews and adopting it raises the compile target, risking the Java 11 compat claim. Phase 8 stays pure scalar + FJP + EJML. | |
| Adopt now, verify later | Assume the API is usable now and plan adoption directly (runtime version detection, multi-release jar or raised `--release`). Skips the verification gate; risks planning against a false premise. | |

**User's choice:** Re-verify first, then decide
**Notes:** The deferral rationale is 7+ months old ("JDK 25 still preview", recorded 2026-08-26 against a JDK 26 GA target of Mar 2026). Both options that pre-judged the outcome were declined — the decision goes to the researcher with live version verification, and the plan branches on the finding.

---

## GPU integration depth

| Option | Description | Selected |
|--------|-------------|----------|
| Opt-in classpath, default CPU | `JcudaBackend` implements `ComputeBackend`, registered in `BackendSelector`'s allowlist, matmul auto-dispatches above a size threshold — but the JCuda dependency stays out of `modules/numja`. The backend loads only if JCuda jars are on the classpath (reflection + `ClassNotFoundException` → CPU fallback). Zero-install preserved; GPU is opt-in by adding jars. | ✓ |
| Vendor JCuda into `dist/libs/` | Same as above, but JCuda native jars vendored into `dist/libs/` so every user gets the GPU path automatically when a device is present. Matches the zero-install promise, but every offline user downloads CUDA native libs they may never use — and Phase 7 D-03 explicitly forbade vendoring. | |
| System-property opt-in only | GPU path enabled by an explicit system property (like `BackendSelector`'s existing `-Dnumja.backend.allow=true` gate), not automatic. Most conservative — no caller ever gets GPU behavior by surprise. But it defeats HW-02's auto-dispatch requirement, which is what Phase 5 built the HAL for. | |

**User's choice:** Opt-in classpath, default CPU
**Notes:** This is the option that keeps Phase 7 D-03's intent (production modules stay JCuda-free) while still delivering the integration Phase 7 D-01 deferred. The mechanism makes the default path provably unchanged — no JCuda on classpath means the backend never loads.

---

## Missing measurements

| Option | Description | Selected |
|--------|-------------|----------|
| Dev machine only | All three caveats measured on the dev machine (10-core i7-1255U, Windows): proper CPU baseline, N-sweep (e.g. 512/1024/2048/4096/8192) for the real dispatch threshold, local GPU run if a device is present. Numbers come from the same box users develop on — directly comparable to Phase 2 JMH baselines. | |
| Dev machine + Colab cross-check | Dev machine for CPU baseline + threshold sweep, plus re-run the Colab notebook for a Linux/GPU cross-check against the 122.75x number. More evidence, but Colab is not a deployment target (Phase 7's own caveat) and re-running it costs a session. | ✓ |
| Dev machine, GPU_ABSENT fallback | Local measurements first; if this Windows box has no CUDA device, record the GPU_ABSENT negative result and ship Phase 8 as CPU-only — `JcudaBackend` ships as an opt-in code path with no local perf evidence. Honest but risks an unintegrated backend. | |

**User's choice:** Dev machine + Colab cross-check
**Notes:** The 122.75x headline rests on a 2-vCPU Colab container CPU baseline — a materially different machine from the 10-core dev box the regression suite runs on. The cross-check exists so the GO number isn't the only evidence, while Colab stays evidence-only and not a deployment target.

---

## Acceptance bar

| Option | Description | Selected |
|--------|-------------|----------|
| Any measurable improvement | Phase 8 must show a measurable improvement somewhere — `multiply_elementwise` improves on its Phase 2 number if it reproduces, a pandas/sklearn hot path improves on its Phase 1 number, **or** GPU matmul beats CPU above the measured threshold — whichever the measurements actually support. No target pre-committed. | ✓ |
| Pre-committed numeric targets | Pre-commit: multiply ≥2x, GPU matmul ≥2x over the 10-core baseline, `read_csv` and `linear_regression_fit` improve. Highest ambition, but binds Phase 8 to numbers it may not reach — Phase 2's multiply ceiling was root-caused as hardware-side. | |
| Evidence-only, no improvement required | The deliverable is measured evidence, not speedups: close the 3 GO caveats, re-verify CPU-03 with a recorded verdict, document whatever the numbers show. Safest, but Phase 8 may ship no optimization. | |

**User's choice:** Any measurable improvement
**Notes:** The middle path between target-chasing and pure measurement. It still requires the phase to *improve* something measurable, but leaves which thing to the evidence — consistent with D-02's and D-03's evidence-first stance.

---

## Constraints

| Option | Description | Selected |
|--------|-------------|----------|
| Public API frozen | `NumJa.java` = 61 and `ArrayOps.java` = 34 public statics byte-identical (frozen since Phase 2). All optimization internal. | |
| No regression (tests + gate) | The 90-test suite stays green, and `scripts/check_regression.ps1` against `04-baseline.json` (50% tolerance) must not regress. | |
| Java 11 compat re-verified if touched | If CPU-03 is adopted and requires `--release 11→21` for `modules/numja`, the README's "Java 11 through 25+" claim must be re-verified or corrected — no silent compatibility break. | |
| Colab re-run stays code-only | Phase 7's code-only constraint still applies to the Colab re-run: no manual JAR downloads, no `mvn install:install-file`, no committed binaries. Maven Central coordinates resolve them. | ✓ |

**User's choice:** Colab re-run stays code-only
**Notes:** Multi-select. The user explicitly selected only the Colab code-only constraint. The other three were **not** selected here because they are already locked by earlier phases and carried into CONTEXT.md as "Carried-forward constraints (locked in earlier phases, NOT re-decided)" — re-asking would have been redundant given the scope guardrail's rule against re-asking decided questions. The API freeze, the regression contract, and the compat-verification requirement all hold regardless.

---

## Claude's Discretion

- Exact N-sweep values beyond the example set (512/1024/2048/4096/8192) — tuned to where the crossover actually appears
- Which specific pandas/sklearn hot paths are worth attacking after the sweep
- The `multiply_elementwise` fix mechanism (FJP leaf-size vs EJML delegation) — D-02 fixes the policy, not the mechanism
- JCuda classpath-detection mechanism (reflection vs ServiceLoader) — D-04 fixes only the requirement that the default path is JCuda-free
- Where the measured dispatch threshold lands — Phase 5 stubbed `THRESHOLD_GPU = 4_096_000` as a heuristic; Phase 8 replaces it with a measured value

## Deferred Ideas

- Non-gemm GPU ops (elementwise, reduce on GPU) — Phase 7 D-02: "only after matmul integration proves value"
- Vendoring JCuda native jars into `dist/libs/` for true zero-install GPU — Phase 7 deferred; D-04 keeps it deferred (classpath-optional instead)
- Distributed/cluster execution, lazy evaluation + fluent query API, production-ready GPU for the full op set, new ML models — all v2 requirements per REQUIREMENTS.md
- USE-01 (shape/dtype mismatch error messages) — dropped with Phase 6; still unmapped and not re-assigned here
