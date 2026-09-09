# Phase 5: Hardware Abstraction Layer & GPU POC - Research

**Researched:** 2026-08-28
**Domain:** Compute backend abstraction, GPU offload POC
**Confidence:** MEDIUM (TornadoVM version HIGH; dispatch threshold LOW until measured)

## Summary

Phase 5 inserts a `ComputeBackend` abstraction between `ArrayOps`/`NDArray` hot paths and the existing FJP-based `ParallelOps`, so a future GPU implementation can be swapped in without changing public API. Default backend wraps `ParallelOps` (zero behavior change). A separate `bench/tornado-poc/` module hosts the TornadoVM 5.2.0-jdk21 GEMM POC — kept out of `modules/numja/pom.xml` per the stdlib-only interface constraint.

Primary recommendation: ship `ComputeBackend` + `CpuThreadBackend` + `BackendSelector` first (no public API change, 3-4 tests), wire dispatch into 5-10 hot paths second (default-path regression must stay green), run the GPU POC last as a separate module that informs a go/no-go decision doc.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Backend abstraction interface | API / Backend | — | `ComputeBackend` lives in `numja.core`; called from same module's hot paths |
| CPU backend impl | API / Backend | — | `CpuThreadBackend` wraps `ParallelOps` 1:1, in-process |
| Dispatch decision | API / Backend | — | Size threshold check in hot path; no I/O, no service boundary |
| GPU POC execution | External (TornadoVM process) | — | POC is a separate Maven module with own pom + JNI/native deps |
| Driver selection | External (OS + TornadoVM config) | — | `-Dtornado.device=...` system property, set by caller before POC runs |

## Codebase Reconnaissance

**Existing parallel infra (do not break):**
- `numja.core.ParallelOps` — FJP-based, `THRESHOLD=100_000`, `LEAF_CUTOFF=16_384`, Kahan per-leaf, volatile `testThresholdOverride` for tests.
- `numja.config.ThreadPoolConfig.getInstance().getForkJoinPool()` — singleton pool, 60% cores.
- Public-API-free: callers see no new entry points from Phase 2.

**Public API surface to preserve (frozen):**
- `NumJa.java` = 61 public static methods (from context).
- `ArrayOps.java` = 34 public static methods (verified by Read: lines 1-314, 34 public statics counted).
- `NDArray` class — instance methods called by `ArrayOps` (`add`, `subtract`, `multiply`, `divide`, `sum`, `mean`, `min`, `max`, `prod`, etc.).
- Constraints: add new methods, do not modify signatures; no new public deps in `modules/numja/pom.xml`.

**Existing regression gate (must remain green):**
- `modules/numja/src/test/java/com/numja/core/ParallelRegressionTest.java` — 4 timing tests using median-of-51, FJP-vs-raw ratio gates. Any HAL dispatch that touches the parallel path must not inflate these ratios.

**MEM-02 ceiling (for HW-02 DoS mitigation):**
- `pandas.internal.ChunkedReadOptions.MAX_CHUNK_ROWS = 100_000` — hard row ceiling for hostile input. Reuse the same scale for backend dispatch DoS guard.

## Architecture Recommendation

### Interface shape: `ComputeBackend` with op-level methods

```java
public interface ComputeBackend {
    void elementwiseBinary(double[] a, double[] b, double[] out, DoubleBinaryOperator op);
    void elementwiseUnary(double[] in, double[] out, DoubleUnaryOperator op);
    void scalarBinary(double[] a, double scalar, double[] out, DoubleBinaryOperator op);
    double sum(double[] data);
    double prod(double[] data);
    double min(double[] data);
    double max(double[] data);
    void matmul(double[] a, int aRows, int aCols, double[] b, int bRows, int bCols, double[] out);
}
```

**Justification (ponytail-minimal):**
- Op-level methods (not strategy-per-op classes) match the shape already in `ParallelOps`. No over-engineering.
- `matmul` is the only new shape; current code uses `EJML CommonOps_DDRM.mult` inline in `ArrayOps.dot` (line 213-217). Moving it onto the interface is what makes GPU offload possible — but the EJML delegation stays in `CpuThreadBackend.matmul` for the default path (zero behavior change).
- Mirrors NumPy `__array_function__` ergonomics (one dispatcher, many backends) without its dynamic-dispatch over-engineering — static interface dispatch is faster and JIT-friendly.
- Stdlib-only: `DoubleBinaryOperator`/`DoubleUnaryOperator` are `java.util.function`. No new deps.

### `CpuThreadBackend` impl

- Singleton: `CpuThreadBackend.getInstance()` (lazy-init holder).
- Each method delegates to the matching `ParallelOps` static (e.g., `elementwiseBinary` → `ParallelOps.elementwiseBinary`, `sum` → `ParallelOps.sum`, `matmul` → current EJML inline body).
- **Zero behavior change:** identical numerics, identical threshold gate, identical FJP path. This is the assertion `ParallelRegressionTest` enforces.

### Dispatch: `BackendSelector` + size-threshold gate

```java
public final class BackendSelector {
    private static final int THRESHOLD_GPU = 4_096_000; // ~4096^2
    private static volatile ComputeBackend active = CpuThreadBackend.getInstance();
    // setBackend() gated by system property allowlist (see Threat Model HW-01)
}
```

**Dispatch rules:**
- Default: `CpuThreadBackend.getInstance()` for every op. No overhead beyond one virtual call (JIT inlines).
- GPU path is POC-only — never auto-promoted in Phase 5. Threshold constant exists, dispatch code is stubbed but inactive.
- `THRESHOLD_GPU = 4_096_000` = ~4096² matmul. Matches `MAX_CHUNK_ROWS × 4096` from `pandas.internal.ChunkedReadOptions`. Below threshold = CPU default always.

**Why stub, not wire GPU now:** Phase 5 = POC. Wiring GPU into the hot path before the POC proves ≥2x speedup is premature optimization — the `bench/tornado-poc/` module's job is to earn that wiring.

## Library Analysis

### TornadoVM 5.2.0-jdk21

- **Verified live** in `.planning/phases/01-baseline-benchmark/VERSIONS.md` (2026-08-26).
- Dual-track: `v5.2.0-jdk21` and `v5.2.0-jdk25` (released 23 Jul 2026).
- Compile target `--release 17` in `modules/numja` is compatible with TornadoVM's runtime requirements (TornadoVM JIT runs as a JVM-level tool, doesn't require re-compiling user code at a higher source level).
- Backends: NVIDIA CUDA (Linux + Windows amd64), AMD OpenCL, Apple Metal, Intel Level Zero / SPIR-V.
- POC dep allowed **only** in `bench/tornado-poc/pom.xml` — `modules/numja/pom.xml` stays stdlib-only.

### POC module structure

```
bench/
└── tornado-poc/
    ├── pom.xml                          # TornadoVM 5.2.0-jdk21 dep, separate module
    └── src/main/java/bench/tornadopoc/
        └── GemmBench.java               # 4096² matmul, JMH or simple timed loop
```

- POM parent: same root pom, new `<module>bench/tornado-poc</module>` entry. Root pom adds the module, `modules/numja` POM unchanged.
- Benchmark target: GEMM at 4096². POC success = GPU ≥ 2x CPU wall time AND transfer < 50% of wall time.
- Output: `docs/05-GPU-POC-RESULTS.md` (go/no-go decision doc), referenced by Phase 6+ if POC succeeds.

### Alternatives considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TornadoVM | Aparapi | Less maintained, narrower GPU support. TornadoVM wins on ecosystem maturity. |
| TornadoVM | JCuda | JNI binding — loses zero-install. Defeats the "pure Java POC" goal. |
| Op-level interface | Strategy-per-op class hierarchy | More files, no perf gain. Interface with default methods is the minimum. |
| Singleton `BackendSelector` | Per-call `ComputeBackend` parameter | Threads the backend through every call site — breaks 5-10 hot paths instead of one dispatch site. |

## API Surface Impact Table

All signatures byte-identical. No public API additions.

| Public method (caller) | Current internal route | New internal route (via HAL) | Notes |
|------------------------|------------------------|------------------------------|-------|
| `ArrayOps.add` | `NDArray.add` → EJML | unchanged (Phase 5 wiring deferred) | Phase 5-02 candidate |
| `ArrayOps.multiply` | `NDArray.multiply` → EJML | unchanged | Phase 5-02 candidate |
| `ArrayOps.dot` | inline `CommonOps_DDRM.mult` (line 213-217) | `BackendSelector.get().matmul(...)` | Phase 5-02 first wiring — gemm is the POC target |
| `ArrayOps.sum` | `NDArray.sum` → ? | `BackendSelector.get().sum(...)` | Phase 5-02 candidate |
| `ArrayOps.mean` | `NDArray.mean` → sum/N | unchanged | deferred — derives from sum |
| `ArrayOps.min` | `NDArray.min` → ? | `BackendSelector.get().min(...)` | Phase 5-02 candidate |
| `ArrayOps.max` | `NDArray.max` → ? | `BackendSelector.get().max(...)` | Phase 5-02 candidate |
| `ArrayOps.prod` | `NDArray.prod` → ? | `BackendSelector.get().prod(...)` | Phase 5-02 candidate |
| `ArrayOps.std` / `var` | inline (line 236-247) | unchanged | deferred — derives from mean |
| `ArrayOps.zeros/ones/empty/eye/arange/linspace/full` | pure construction | unchanged | out of scope (no compute) |

**Phase 5-02 scope:** wire 5-10 hot paths above. Default-path regression (`ParallelRegressionTest`) must remain green — `CpuThreadBackend` is bit-identical to current behavior.

## Performance Budget

| Metric | Budget | How verified |
|--------|--------|--------------|
| Dispatch overhead (interface call vs direct static) | < 5% wall time at n ≥ 10⁶ | JMH microbench in POC module, compare `BackendSelector.get().sum()` vs `ParallelOps.sum()` |
| POC GEMM speedup | GPU ≥ 2x CPU at N = 4096² | `bench/tornado-poc` timed runs, `docs/05-GPU-POC-RESULTS.md` |
| GPU transfer ceiling | transfer < 50% wall time | Same POC output — separate `transfer_ms` metric |
| Default-path regression | `ParallelRegressionTest` ratios unchanged | Existing test suite (4 tests, must remain green) |

## Test Strategy

### New tests (Phase 5-01)

`BackendSelectorTest` — 3 tests, in `modules/numja/src/test/java/com/numja/core/`:

1. `defaultBackendIsCpuThread` — assert `BackendSelector.get()` returns `CpuThreadBackend.getInstance()` on fresh JVM.
2. `systemPropertyOverrideSelectsBackend` — set `-Dnumja.backend=...` to a known allowlist value, assert selection; set to unknown value, assert falls back to `CpuThreadBackend` + logs warning.
3. `dispatchThresholdBoundary` — call `sum` on n = `THRESHOLD_GPU - 1` and `THRESHOLD_GPU`, assert CPU path taken in both (no GPU backend installed in test env).

### Regression gate (Phase 5-02)

- Existing `ParallelRegressionTest` (4 tests) MUST remain green. Wire dispatch as a pure indirection — same FJP, same Kahan, same threshold.
- If any ratio regresses >10% from baseline, dispatch wiring is wrong (likely dispatching elementwise paths twice, or skipping Kahan).

### POC verification (Phase 5-03)

- `bench/tornado-poc` runs GEMM at 4096² on CPU (existing FJP) and GPU (TornadoVM). Outputs `docs/05-GPU-POC-RESULTS.md` with: speedup ratio, transfer overhead %, driver/backend used, JDK version, hardware spec.

## Threat Model (ASVS L1 + STRIDE)

| ID | Threat (STRIDE) | Attack vector | Mitigation |
|----|-----------------|---------------|------------|
| HW-01 | Elevation of Privilege | Caller sets `numja.backend` system property to untrusted value at runtime → unexpected class loaded as backend | **Allowlist** of backend names (`cpu-thread` only in Phase 5) + system-property gate requiring `-Dnumja.backend.allow=true` to honor the override. Default = `CpuThreadBackend`, no override possible without explicit opt-in. |
| HW-02 | Denial of Service | Hostile size input (e.g., `Integer.MAX_VALUE` rows) → dispatch path triggers OOM, GPU device lock, or pool starvation | Hard ceiling matching `MEM-02 MAX_CHUNK_ROWS = 100_000` rows. Any dispatch call with `n > ceiling` throws `IllegalArgumentException` before reaching backend. Same ceiling as chunk reader enforces. |
| HW-03 | Tampering / Spoofing | TornadoVM auto-detects GPU adapter; on a multi-GPU box picks wrong device (e.g., integrated GPU instead of discrete) → silent slowdown or wrong results | Explicit `-Dtornado.device=...` opt-in for POC. No auto-detect. POC bench logs the selected device and asserts it matches `-Dtornado.device` before timing. |

**ASVS applicability:**
- V5 Input Validation: yes (HW-02 size ceiling).
- V6 Cryptography: N/A (no crypto in scope).
- V2 Authentication: N/A (no user identity in scope).
- V4 Access Control: HW-01 — backend selection is a privileged operation, gated by system property.

## Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R1 | TornadoVM 5.2.0-jdk21 won't install on JDK 25 host | LOW | HIGH (blocks POC) | Pin POC build to JDK 21 (`<maven.compiler.release>21</maven.compiler.release>` in POC pom). Verify TornadoVM install path in 05-03 first task before writing benchmark. |
| R2 | GPU transfer overhead > 50% wall time → POC fails budget | MEDIUM | MEDIUM | POC doc explicitly reports transfer %. If transfer > 50%, mark POC as "no-go for streaming workloads" rather than "fail" — useful negative result. |
| R3 | EJML delegation regression in `CpuThreadBackend.matmul` (subtle behavior change vs inline) | LOW | HIGH (silent numeric shift) | `CpuThreadBackend.matmul` is a 1:1 copy of `ArrayOps.dot` lines 213-217. No algorithmic change. Verified by `ParallelRegressionTest` + a new golden-value test in Phase 5-02. |
| R4 | Dispatch overhead > 5% on small arrays | LOW | MEDIUM | JIT inlines virtual call on a final-class `CpuThreadBackend`. Measure with JMH in POC module. If > 5%, fall back to direct `ParallelOps` call in hot paths and skip dispatch for n < 10⁴. |
| R5 | Security boundary not enforced (HW-01 bypass) | LOW | HIGH | Unit test asserts: setting `-Dnumja.backend=evil.Class` is rejected even with `-Dnumja.backend.allow=true` (allowlist is name-based, not classloader-based). Test runs as part of `BackendSelectorTest`. |

## Recommended Plan Structure (3 plans / 3 waves)

### 05-01 — HAL skeleton (Wave 1)
- Add `ComputeBackend` interface in `numja.core`.
- Add `CpuThreadBackend` singleton delegating to `ParallelOps` + EJML inline copy for matmul.
- Add `BackendSelector` with `THRESHOLD_GPU` constant (unused but declared).
- Add `BackendSelectorTest` (3 tests).
- **No public API change. No wiring. No dispatch.**

### 05-02 — Dispatch wiring (Wave 2)
- Wire 5-10 hot paths (`ArrayOps.dot` first, then sum/min/max/prod/elementwise trio) through `BackendSelector.get().<op>(...)`.
- Add golden-value test for `CpuThreadBackend.matmul` matching current `ArrayOps.dot` output.
- Verify `ParallelRegressionTest` 4/4 green, ratios within 10% of Phase 2 baseline.

### 05-03 — GPU POC (Wave 3)
- Create `bench/tornado-poc/` module with own `pom.xml` (TornadoVM 5.2.0-jdk21 dep).
- Add `<module>bench/tornado-poc</module>` to root pom.
- Write `GemmBench.java` — 4096² GEMM on CPU (FJP) and GPU (TornadoVM).
- Run on dev box, capture `docs/05-GPU-POC-RESULTS.md` (speedup, transfer %, device, JDK, hardware).
- Output is a **decision doc**, not a code merge. Phase 6+ reads it.

## Open Questions

1. **POC scope: gemm-only, or also reduce?** — POC is cheaper with one op. Gemm has the highest ceiling for GPU win.
2. **POC platform: Windows+CUDA or Linux+CUDA?** — User's dev box is Windows; Linux is more typical for TornadoVM. Need one concrete target.
3. **No-GPU fallback for benchmark?** — If dev box lacks CUDA, benchmark runs CPU-only. POC doc still valuable as "negative result" (establishes that without GPU, FJP is the ceiling).

## Open Questions (RESOLVED)

- **Q1 = gemm-only.** Cheapest POC, highest expected ceiling, matches the Phase 2 flagged op (`multiply_elementwise` 1.42x — gemm is the obvious next target per `context` block).
- **Q2 = Windows + CUDA.** User's box. NVIDIA + CUDA is the best-supported TornadoVM backend (verified in VERSIONS.md: "Windows amd64" listed).
- **Q3 = synthetic CPU fallback with explicit warning label.** POC runs CPU-only baseline if no GPU; `docs/05-GPU-POC-RESULTS.md` labels output as `GPU_ABSENT` and records CPU ceiling. Negative result is still a result.

## Validation Architecture

Phase 5 validation rides on three existing gates plus one new POC artifact; no new test framework or dependency introduced.

1. **Functional correctness** — `BackendSelectorTest` (05-01, 3 tests) + `CpuThreadBackendGoldenTest` (05-02, 6 tests) prove HAL default selection, allowlist + opt-in, two-path parity for `matmul`, and reduce-op wiring. All JUnit 4 in `modules/numja/src/test/java/com/numja/core/`.
2. **Regression gate** — existing `ParallelRegressionTest` (4 tests, median-of-51 timing) MUST stay green through 05-02 dispatch wiring. `scripts/check_regression.ps1` carries the `04-baseline.json` numbers; 05-02 must not inflate the FJP-vs-raw ratios beyond the documented 50% tolerance.
3. **Static constraints** — `grep -c "public static" NumJa.java` = 61 and `ArrayOps.java` = 34 (frozen). `grep -c "<dependency>" modules/numja/pom.xml` unchanged (stdlib-only preserved; TornadoVM dep lives only in `bench/tornado-poc/pom.xml`).
4. **POC artifact** — `bench/tornado-poc/GemmBench` produces the key=value output schema (env / device / jdk / hardware / cpu_baseline_ms / gpu_ms / transfer_ms / transfer_pct / speedup_ratio / result / verdict). Local CPU-only run labels `result=GPU_ABSENT verdict=NO-GO`; Colab NVIDIA run labels per measured speedup vs budgets (GO iff `speedup_ratio >= 2.0 && transfer_pct < 50.0`). Captured into `docs/05-GPU-POC-RESULTS.md` with explicit environment header.

Phase 5 verification is complete when all four gates pass AND the go/no-go verdict is documented.

## Sources

- `.planning/phases/01-baseline-benchmark/VERSIONS.md` — TornadoVM 5.2.0 dual-track, EJML 0.45.0, Vector API status (HIGH).
- `.planning/research/SUMMARY.md` — Backend-first layering guidance, GPU offload pitfalls (MEDIUM, offline-research note).
- `modules/numja/src/main/java/numja/core/ParallelOps.java` — existing FJP threshold, Kahan compensation, task tree (HIGH, direct read).
- `modules/numja/src/main/java/numja/core/ArrayOps.java` — 34 public statics, EJML delegation in `dot` (HIGH, direct read).
- `modules/numja/src/test/java/com/numja/core/ParallelRegressionTest.java` — regression gate methodology (HIGH, direct read).
- `modules/pandas/src/main/java/pandas/internal/ChunkedReadOptions.java` — `MAX_CHUNK_ROWS = 100_000` ceiling for HW-02 (HIGH, direct read).

## Metadata

- **Confidence breakdown:**
  - Standard stack: HIGH (TornadoVM version verified, pom layout standard).
  - Architecture: MEDIUM (op-level interface is defensible but the threshold value `4_096_000` is heuristic until measured).
  - Pitfalls: HIGH (mirror Phase 2 Kahan + threshold lessons; regression test already exists).
- **Research date:** 2026-08-28
- **Valid until:** 2026-09-28 (30 days — stable domain, no fast-moving deps).
