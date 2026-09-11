# Phase 7: GPU Fallback POC via JCuda/cuBLAS - Context

**Gathered:** 2026-09-12
**Status:** Ready for planning
**Conditional:** Execute ONLY if Phase 5 (TornadoVM POC on Colab T4) concludes NO-GO. If Phase 5 achieves GO, this phase becomes an optional follow-up or is removed.

## Phase Boundary

Second GPU POC for the same GEMM benchmark, using JCuda 12.6.0 + cuBLAS `Dgemm` instead of TornadoVM. Isolated benchmark module + Colab notebook; production modules (`modules/numja`, `modules/sklearn`, `modules/pandas`) gain zero JCuda dependency in this phase. Integration into the HAL (`JcudaBackend` implementing `ComputeBackend`) is explicitly a LATER phase, gated on this POC passing both gates. Public API (NumJa.java=61, ArrayOps.java=34) unchanged.

## Implementation Decisions

### Scope (user decision)
- **D-01:** POC first, integration later. Phase 7 delivers ONLY the isolated benchmark (bench module + Colab notebook + measured results + go/no-go conclusion). Wiring `JcudaBackend` into `BackendSelector`/`ComputeBackend` is a separate future phase, executed only if this POC passes both gates. Rationale: production modules must stay stdlib/EJML-only until value is proven; matches the TornadoVM precedent (tornado-poc is also an isolated bench module).
- **D-02:** "Chuyển hết phần tính toán bằng GPU sang JCuda" is the END GOAL across phases, not Phase 7 scope. Phase 7 proves the library works; a later phase does the HAL integration if gates pass.

### Dependency acquisition (user decision)
- **D-03:** Notebook downloads JCuda jars at runtime (jcuda.org release matching CUDA 12.6 — 12.6.0 line), then `mvn install:install-file` into the local Maven repo on Colab — same pattern as the TornadoVM SDK jars in `notebooks/colab-gpu-poc.ipynb` Step 1. Nothing vendored into `dist/libs/`, no Windows build-pipeline changes, no new committed binaries.

### Bench module shape (user decision)
- **D-04:** `bench/jcuda-poc` clones the `bench/tornado-poc` structure: same `GemmBench` key=value output format (`cpu_baseline_ms`, `gpu_ms`, `transfer_ms`, `speedup_ratio`, `transfer_pct`, `device`, `result`, `verdict`, `env`, `jdk`, `hardware`, `size`, `cpu_vs_gpu_frob_rel_err`), same seeds (0xC0FFEE, 0xBADF00D), same N=4096, same CPU baseline path. Direct cross-comparison with the TornadoVM numbers from Phase 5.
- **D-05:** New dedicated notebook `notebooks/colab-jcuda-poc.ipynb` (Colab GPU/T4 metadata, %%bash cells, T4 preflight, nvidia-smi telemetry sampling, hard result=OK check — mirroring the fixed TornadoVM notebook). The existing `notebooks/colab-gpu-poc.ipynb` is NOT modified; its TornadoVM outputs remain the Phase 5 evidence. Note: avoid `set -o pipefail` with `| head` (SIGPIPE aborts cell) and verify real asset URLs/extensions before wget — two bugs already hit in the TornadoVM notebook.

### Gates (carried from Phase 5 rubric)
- **D-06:** Numerical: full-matrix Frobenius relative error `cpu_vs_gpu_frob_rel_err <= 1e-9` (expected ~1e-12 for double GEMM N=4096). Fail → `result=NUMERIC_MISMATCH`, kernel/parameter bug — fix before any perf claim.
- **D-07:** Performance: `speedup_ratio >= 2.0` AND `transfer_pct < 50.0` → GO. Both gates pass → later HAL-integration phase is justified. Either fails → NO-GO.
- **D-08:** cuBLAS `cublasDgemm` is column-major; NumJa arrays are row-major. The bench must handle the transpose correctly (transpose A/B, swap operands: C_row = A_row·B_row ⇔ C_col^T = B_col^T·A_col^T) and the Frobenius check will catch a wrong formulation. This is the known correctness trap of the POC.

### Failure policy (user decision)
- **D-09:** If JCuda/cuBLAS also fails on Colab T4 (native init error, no device, etc.): document the failure evidence, conclude GPU production backend NO-GO, close the phase, and DEFER GPU to a future milestone. No third backend (JavaCPP, ND4J) tried in this phase — avoids an endless POC chain.

## Claude's Discretion

- Exact JCuda artifact set (jcuda + jcublas + native classifier jars) — researcher verifies what the 12.6.0 release ships.
- Warmup iterations, timing methodology inside GemmBench (mirror tornado-poc's).
- Notebook cell layout / env persistence mechanics (mirror the fixed tornado notebook's `/tmp/poc-env.sh` pattern).

## Deferred Ideas

- `JcudaBackend implements ComputeBackend` + `BackendSelector` allowlist entry ("jcuda") + auto-dispatch — future phase, gated on this POC passing D-06 + D-07.
- Covering more ops (elementwise, reduce) on GPU — only after matmul integration proves value.
- Vendoring JCuda native jars into `dist/libs/` for offline users — only relevant if integration ships.

## Canonical References

Downstream agents MUST read these before planning or implementing.

### Phase boundary + scope
- `.planning/ROADMAP.md` §Phase 7 — goal, success criteria, conditionality
- `.planning/REQUIREMENTS.md` §GPU-04 — conditional fallback requirement
- `.planning/STATE.md` — Phase 5 status, phase 7 addition note

### Phase 5 evidence (the baseline being compared against)
- `notebooks/colab-gpu-poc.ipynb` — TornadoVM POC notebook (setup pattern, telemetry, gates, the pipefail/asset-URL bugs to avoid)
- `bench/tornado-poc/pom.xml` — module structure, shade + ServicesResourceTransformer, dependency versions
- `bench/tornado-poc/src/main/java/bench/tornadopoc/GemmBench.java` — key=value output format, seeds, Frobenius check, CPU baseline path (the template D-04 clones)
- `docs/COLAB-RECIPE.md` — historical Colab recipe (older, direct-java references; use notebook as source of truth)

### HAL integration points (context only — NOT Phase 7 scope)
- `modules/numja/src/main/java/numja/core/ComputeBackend.java` — the interface a future JcudaBackend would implement
- `modules/numja/src/main/java/numja/core/BackendSelector.java` — allowlist + opt-in model the future phase would extend
- `modules/numja/src/main/java/numja/core/CpuThreadBackend.java` — CPU fallback that must never regress

### Project constraints
- `.planning/PROJECT.md` §Constraints — public API frozen, stdlib-JVM-only for production, no breaking change
- `.planning/codebase/STACK.md` — JDK/release levels, EJML vendored, build model
- `.planning/codebase/CONCERNS.md` — modules/ git-ignored, closed-source release model

### External (researcher verifies exact versions/URLs)
- JCuda downloads: http://www.jcuda.org/downloads/downloads.html — native JARs bundled since 0.8.0RC
- jcuda/jcuda + jcuda/jcublas GitHub repos — 12.6.0 release line matches the CUDA 12.6 runtime the Colab T4 workaround already installs

## Existing Code Insights

### Reusable Assets
- `bench/tornado-poc/` — the entire module is the cloning template (pom shade config, GemmBench harness, output labels)
- `notebooks/colab-gpu-poc.ipynb` Step 1 — SDK download + `mvn install:install-file` + env persistence pattern; Step 2 — device preflight, nvidia-smi telemetry, hard success enforcement
- CUDA 12.6 runtime install block (cuda-keyring + cuda-toolkit-12-6) already proven in the TornadoVM notebook — JCuda 12.6.0 binds to the same runtime

### Known Landmines (from the TornadoVM debugging)
- `set -euo pipefail` + `nvidia-smi | head` = SIGPIPE abort — use `set -eu`
- `wget -q` hides 404s on wrong asset URLs — verify real filename/extension from the release page first
- Each `%%bash` cell is a fresh shell — persist env via `/tmp/poc-env.sh`
- Broad `catch (Throwable)` in device resolution hides the real native error — surface stderr, don't swallow
- Shaded JAR must keep `META-INF/services` (ServicesResourceTransformer) — relevant if JCuda bench also shades
