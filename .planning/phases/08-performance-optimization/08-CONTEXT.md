# Phase 8: Clean Hot Paths - Context

> **RE-SCOPED 2026-10-15 per user:** GPU HAL integration (Track 2) deferred. This phase is now
> clean code + cut intermediate steps on measured hot paths (OPT-01/OPT-02). The Both-CPU+GPU
> scope below (D-01) is superseded; GPU items move to backlog. Phase 9 covers repo-wide cleanup (MNT-01).

**Gathered:** 2026-10-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 8 closes the two open performance threads left by Phases 2 and 7, and is the phase that makes the idle HAL do real work.

**Track 1 — CPU throughput completion (pure-internal, no native deps):**
- Re-measure and attempt to lift the `multiply_elementwise` 10⁷ = **1.42x** ceiling that Phase 2 flagged (target was ≥2x) and recommended investigating "before Phase 3 work" — never picked up.
- Resolve **CPU-03** (Java Vector API), the only unmapped v1 requirement, whose deferral rationale ("JDK 25 still preview") is 7+ months old.
- Sweep the **pandas/sklearn hot paths** Phase 2 never touched — `read_csv_california` (105.5ms) and `linear_regression_fit` (20.67ms) are still Phase 1-era numbers.

**Track 2 — Phase 7 GO-caveat closure + HAL integration (opt-in native path):**
- Measure the three prerequisites Phase 7's GO explicitly left unestablished: (1) the 10-core dev-machine CPU baseline (its 122.75x was against a 2-vCPU Colab container), (2) an N-sweep to locate the real GPU dispatch threshold (only N=4096 was measured), (3) local/Windows GPU validation (T4 is not a deployment target).
- Wire `JcudaBackend` into the existing idle HAL (`ComputeBackend` / `BackendSelector`) behind a size threshold, with CPU fallback — the item Phase 7 D-01 explicitly deferred to "a future phase, gated on this POC passing D-06 + D-07". Both gates passed (frob 9.85e-17, speedup 122.75x, transfer 17.33%), so the deferral is satisfied.

**Out of scope:** new ML models, breaking API changes, distributed execution, UI/dashboard — all per PROJECT.md Out of Scope. Non-gemm GPU ops stay deferred per Phase 7 D-02 ("only after matmul integration proves value").

</domain>

<decisions>
## Implementation Decisions

### Scope
- **D-01:** Phase 8 spans **both** tracks — CPU throughput completion **and** Phase 7 GO-caveat closure with JCuda gemm HAL integration. Rationale: user selected "Both CPU + GPU" when offered the materially-different scope options (CPU-only / GPU-only / measurement-only / both). — **Reversibility:** one-way — ROADMAP.md currently lists this phase as `Goal: [To be planned]`; committing to this scope means rewriting the phase goal and adding new requirement IDs to REQUIREMENTS.md, which the coverage gate then checks against. Undoing means re-stubbing both files mid-planning.

### CPU throughput
- **D-02:** Re-measure `multiply_elementwise` on this machine first; optimize only if the 1.42x still reproduces. Then try the **cheapest** fix (FJP leaf-size tuning, or delegating to EJML above threshold) and accept the result even if it stays under 2x — Phase 2 concluded the ceiling may be hardware-side (EJML SIMD advantage + 80MB memory-bandwidth ceiling on the i7-1255U). Honest reporting over target-chasing.
- **D-03:** **Re-verify Vector API's JDK status before deciding** on CPU-03. The deferral rationale is 7+ months old — JDK 26's GA target was Mar 2026, today is Oct 2026. If non-preview: adopt with runtime version detection + pure-Java fallback. If still preview: keep deferred and record the verdict. Evidence-driven, not assumption-driven.

### GPU integration
- **D-04:** `JcudaBackend` implements `ComputeBackend`, registered in `BackendSelector`'s allowlist, with matmul auto-dispatching above a size threshold. **The JCuda dependency stays OUT of `modules/numja`** — the backend loads only if JCuda jars are on the classpath (reflection + `ClassNotFoundException` → CPU fallback). Zero-install is preserved for offline `dist/` users; GPU is opt-in by adding the jars. — **Reversibility:** one-way — this reverses Phase 7 D-03's "production modules stay JCuda-free" and adds a native-dependency path to the offline distribution contract. Undoing it after release means shipped users have a backend that silently no-ops; mitigation is the classpath-optional load so the default path never changes.
- **D-05:** Missing measurements come from the **dev machine + Colab cross-check**: 10-core i7-1255U (Windows) for the proper CPU baseline and the N-sweep threshold curve (e.g. 512/1024/2048/4096/8192), plus a Colab re-run for a Linux/GPU cross-check against the 122.75x number. Colab is evidence only, not a deployment target (Phase 7 caveat).
- **D-06:** **Acceptance bar = any measurable improvement.** Phase 8 must show a measurable improvement somewhere — `multiply_elementwise` improves on its Phase 2 number if it reproduces, a pandas/sklearn hot path improves on its Phase 1 number, **or** GPU matmul beats CPU above the measured threshold — whichever the measurements actually support. No target is pre-committed; the win is whatever the evidence shows. Rationale: pre-committing to numbers binds the phase to results it may not reach — Phase 2's multiply ceiling was root-caused as hardware-side.
- **D-07:** **The Colab re-run stays code-only** (Phase 7 D-03 carry-forward): no manual JAR downloads, no `mvn install:install-file`, no committed binaries. Maven Central coordinates (`org.jcuda:jcuda:12.6.0`, `org.jcuda:jcublas:12.6.0`) resolve them.

### Carried-forward constraints (locked in earlier phases, NOT re-decided)
- **Public API frozen (v0.2.0):** `NumJa.java` = 61 public statics, `ArrayOps.java` = 34 — byte-identical. Every optimization is internal. Locked since Phase 2, asserted in every phase since.
- **Regression contract:** the 90-test suite stays green, and `scripts/check_regression.ps1` against `04-baseline.json` (50% tolerance, carried from Phase 3) must not regress.

### Claude's Discretion
- Exact N-sweep values beyond the example set (512/1024/2048/4096/8192) — researcher/planner tune to where the crossover actually appears.
- Which specific pandas/sklearn hot paths are worth attacking after the sweep — determined by which are still at Phase 1 numbers.
- The `multiply_elementwise` fix mechanism (FJP leaf-size vs EJML delegation) — D-02 fixes the *policy* (cheapest first, accept the result), not the mechanism.
- JCuda classpath-detection mechanism (reflection vs ServiceLoader) — implementation detail for the planner; D-04 fixes only the requirement that the default path is JCuda-free.
- Where the measured dispatch threshold lands — Phase 5 stubbed `THRESHOLD_GPU = 4_096_000` as a heuristic; Phase 8 replaces it with a measured value.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase goal + requirements (must be written this phase)
- `.planning/ROADMAP.md` §Phase 8 — currently `Goal: [To be planned]`, `Requirements: TBD`; the planner needs the goal and success criteria filled in
- `.planning/REQUIREMENTS.md` — CPU-03 is the only unmapped v1 requirement; new IDs for Phase 8's scope get added here
- `.planning/STATE.md` — current phase position + branch (`dev`) + parallel infra summary

### Phase 7 evidence (the GO being extended)
- `.planning/phases/07-gpu-fallback-poc-via-jcuda-cublas/07-RESULTS.md` — measured GO: `speedup_ratio=122.75`, `transfer_pct=17.33`, `cpu_vs_gpu_frob_rel_err=9.85e-17`; the three "Not yet established" caveats Phase 8 must close
- `.planning/phases/07-gpu-fallback-poc-via-jcuda-cublas/07-CONTEXT.md` — D-01 (POC first, integration later — the deferral Phase 8 consumes), D-03 (code-only Colab), D-06/D-07 (the gates), D-08 (the row/column-major transpose trap)
- `bench/jcuda-poc/` — the POC whose `GemmBench` becomes the integration's reference implementation
- `notebooks/colab-jcuda-poc.ipynb` — the Colab runbook for the D-05 cross-check

### Phase 2 evidence (the CPU ceiling)
- `.planning/phases/02-cpu-parallel-core-threads/02-BASELINE-AFTER.md` — `multiply_elementwise` 10⁷ = 1.42x FLAG with root cause; the "investigate before Phase 3" recommendation Phase 8 picks up
- `.planning/phases/02-cpu-parallel-core-threads/02-RESEARCH.md` — FJP threshold/leaf-size methodology

### Phase 5 HAL (the integration target)
- `.planning/phases/05-hardware-abstraction-layer-gpu-poc/05-RESEARCH.md` — `ComputeBackend` interface shape, `BackendSelector` dispatch rules, `THRESHOLD_GPU = 4_096_000` heuristic, threat model HW-01/HW-02, risk register R1–R5
- `modules/numja/src/main/java/numja/core/ComputeBackend.java` — interface `JcudaBackend` implements
- `modules/numja/src/main/java/numja/core/BackendSelector.java` — allowlist + opt-in model being extended
- `modules/numja/src/main/java/numja/core/CpuThreadBackend.java` — CPU fallback that must never regress

### CPU-03 / Vector API status
- `.planning/phases/01-baseline-benchmark/VERSIONS.md` §2 — the "JDK 25 vẫn preview" finding D-03 re-verifies, plus the `--release 11 → 21` tradeoff if adopted

### Regression gate + benchmarks
- `scripts/check_regression.ps1` — the gate Phase 8 must not regress
- `.planning/phases/03-numerical-accuracy-hardening/03-baseline.json` — origin of the 50% tolerance
- `.planning/phases/04-adaptive-memory-model/04-baseline.json` — current baseline file the gate reads (10 perf + 3 accuracy)
- `bench/` — JMH harness (`mvn -pl bench -am package -DskipTests` → `java -jar bench/target/benchmarks.jar`)

### Project constraints
- `.planning/PROJECT.md` §Constraints + §Out of Scope — API frozen, stdlib-JVM-only for production, no breaking change
- `.planning/codebase/STACK.md` — JDK/release levels, vendored deps, `dist/libs/` composition
- `.planning/codebase/CONCERNS.md` — `modules/` git-ignored (source only on `dev`)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`ComputeBackend` + `CpuThreadBackend` + `BackendSelector`** (Phase 5) — the HAL is already built and idle, with a stubbed `THRESHOLD_GPU = 4_096_000`. Phase 8's GPU work is extending it, not building it. `CpuThreadBackend` delegates 1:1 to `ParallelOps`, so the default path is bit-identical — that property is what makes D-04's classpath-optional load safe.
- **`ParallelOps`** — FJP-based, `THRESHOLD=100_000`, `LEAF_CUTOFF=16_384`, Kahan per-leaf. `LEAF_CUTOFF` is the first tuning knob for D-02's multiply fix; there is a package-private `setThresholdForTesting`/`resetThresholdForTesting` test hook.
- **`ThreadPoolConfig.getInstance().getForkJoinPool()`** — singleton pool sized to 60% of cores.
- **`bench/jcuda-poc/GemmBench`** — already-shipped cuBLAS Dgemm reference: key=value output, fixed seeds (0xC0FFEE, 0xBADF00D), Frobenius check, in-process `cudaMemGetInfo` device-memory evidence. The integration ports this logic onto `ComputeBackend.matmul`.
- **`scripts/check_regression.ps1`** — median-of-3 JMH aggregation against `04-baseline.json`; extended in Phase 4 to parse `Param:*` columns and strip class names (CoreBench/PandasBench/SklearnBench).
- **Golden-value test infrastructure** — `GoldenReferenceTest` vs NumPy (tolerance 1e-13), `AccuracyHardeningTest` (7 hard-fail fixtures), `GoldenFixtures` shared utility.

### Established Patterns
- **Threshold-gated dispatch** — `n < THRESHOLD` stays sequential; the gate is a structural property asserted by `ParallelRegressionTest`, not just a performance knob. Any new dispatch (GPU included) follows this shape.
- **Cross-path equivalence testing** — every parallel path is proved equivalent to its sequential/in-memory counterpart within tolerance (Phase 2 reduce, Phase 3 Kahan, Phase 4 partial-fit). GPU matmul needs the same: golden values, not just timing.
- **Additive-only API changes** — every phase since v0.2.0 froze the surface has shipped additions as *sibling* methods (`read_csv_streaming`, `partial_fit`/`finalize_fit`), never modifications.
- **ASVS L1 + STRIDE threat model** in every PLAN.md `<threat_model>` block (`security_enforcement: true`, `disposition_policy: block on high`) — Phase 4 cross-cutting constraint, carried forward.
- **Offline distribution contract** — `dist/` jars + `dist/libs/` vendored deps are the zero-install promise. D-04 keeps this intact by making JCuda classpath-optional.

### Integration Points
- **`BackendSelector` allowlist + `-Dnumja.backend.allow=true` gate** (Phase 5 threat model HW-01) — where `jcuda` registers; the security model already exists, it needs an entry.
- **`ArrayOps.dot` → `BackendSelector.get().matmul(...)`** — the matmul wiring point Phase 5 identified but left to the integration phase; `ArrayOps.java:213-217` has the inline EJML `CommonOps_DDRM.mult` it replaces.
- **`modules/numja/pom.xml`** — must stay stdlib-only for D-04 to hold; the JCuda coordinates live only in `bench/jcuda-poc/pom.xml` (and the Colab notebook's env).
- **`--release 17`** (Phase 4) — the compile level any Vector API adoption must reconcile with; VERSIONS.md's `--release 11 → 21` tradeoff is the D-03 follow-on.

### Known Landmines
- **Row/column-major transpose** (Phase 7 D-08) — cuBLAS `cublasDgemm` is column-major, NumJa arrays row-major: `C_row = A_row·B_row ⇔ C_col^T = B_col^T·A_col^T`. The Frobenius check catches a wrong formulation; a layout error shows `frob_rel_err` ~O(1), not a small drift.
- **Hybrid P/E-core variance** — JMH 99.9% CI on this machine runs ±6–555ms/op, often comparable to the score. Phase 2's methodology (single fork, same power profile, median-of-N) is what makes comparisons meaningful; Phase 8 must inherit it or its numbers aren't comparable to the baselines.
- **`modules/` is git-ignored** — source exists only on `dev`; the repo is currently on `gsd/phase-07-gpu-fallback-poc-via-jcuda-cublas`. Execution must happen on `dev` or source edits don't land.
- **Windows GPU runtime unverified** — no Windows CUDA run exists in the project; the local leg of D-05 may discover the dev box has no device, in which case the GPU_ABSENT negative result is the honest outcome.

</code_context>

<specifics>
## Specific Ideas

- The user's framing for this phase was simply "optimize code". The discussion refined that into D-01's two-track scope, but the intent is: Phase 8 should be the phase where already-measured opportunities finally get acted on, not where new capability is invented. Every item in scope traces to a number an earlier phase recorded and did not chase.
- Phase 7's own deferral language — "gated on this POC passing D-06 + D-07" — is the explicit hook Phase 8's GPU track consumes. Both gates have measured passes, so this is a deferral being *kept*, not a decision being reopened.
- Two candidate scopes the user saw and **declined**: CPU-only (Phase 8 narrowed to the multiply ceiling + CPU-03 + pandas/sklearn sweep) and measurement-only (no production code changes). The choice of "Both" is the scope anchor; if the planner finds the phase exceeds context budget, splitting should preserve both tracks rather than dropping one.

</specifics>

<deferred>
## Deferred Ideas

- **Non-gemm GPU ops** (elementwise, reduce on GPU) — Phase 7 D-02: "only after matmul integration proves value". Stays deferred until the matmul integration ships and shows a real win.
- **Vendoring JCuda native jars into `dist/libs/`** for true zero-install GPU — Phase 7 deferred this; D-04 keeps it deferred (classpath-optional instead). Revisit only if a later milestone decides the GPU path is a default, not an opt-in.
- **Distributed/cluster execution, lazy evaluation + fluent query API, production-ready GPU for the full op set, new ML models** — all v2 requirements per REQUIREMENTS.md, unchanged.
- **USE-01 (shape/dtype mismatch error messages)** — dropped with Phase 6; still unmapped and not re-assigned here.

</deferred>

---

*Phase: 8-performance-optimization*
*Context gathered: 2026-10-02*
