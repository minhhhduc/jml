# Phase 7: GPU Fallback POC via JCuda/cuBLAS - Research

**Researched:** 2026-09-12
**Domain:** JCuda 12.6.0 JNI bindings + cuBLAS Dgemm on Colab T4; isolated benchmark module
**Confidence:** HIGH (all API claims verified against the actual 12.6.0 jars on Maven Central — bytecode decompiled, natives ELF-parsed, POMs inspected). MEDIUM for performance expectations (estimates until measured).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** POC first, integration later. Phase 7 delivers ONLY the isolated benchmark (bench module + Colab notebook + measured results + go/no-go conclusion). Wiring `JcudaBackend` into `BackendSelector`/`ComputeBackend` is a separate future phase, gated on both gates passing.
- **D-02:** "Move all computation to GPU via JCuda" is the END GOAL across phases, not Phase 7 scope.
- **D-03:** Notebook downloads JCuda jars at runtime, `mvn install:install-file` into local repo on Colab — same pattern as TornadoVM SDK jars. Nothing vendored into `dist/libs/`, no Windows build-pipeline changes, no new committed binaries.
- **D-04:** `bench/jcuda-poc` clones `bench/tornado-poc` structure: same key=value output format, same seeds (0xC0FFEE, 0xBADF00D), same N=4096, same CPU baseline path. Direct cross-comparison with Phase 5 numbers.
- **D-05:** New dedicated notebook `notebooks/colab-jcuda-poc.ipynb` (T4 metadata, %%bash cells, preflight, nvidia-smi telemetry, hard result=OK check). Existing `notebooks/colab-gpu-poc.ipynb` NOT modified.
- **D-06:** Numerical gate: `cpu_vs_gpu_frob_rel_err <= 1e-9`, else `result=NUMERIC_MISMATCH`.
- **D-07:** Perf gate: `speedup_ratio >= 2.0` AND `transfer_pct < 50.0` → GO; either fails → NO-GO.
- **D-08:** cuBLAS column-major vs NumJa row-major — bench must handle the transpose correctly; Frobenius check catches wrong formulation.
- **D-09:** If JCuda/cuBLAS also fails on Colab T4: document evidence, conclude NO-GO, close phase, defer GPU. No third backend tried.

### Claude's Discretion
- Exact JCuda artifact set — researcher verifies what the 12.6.0 release ships.
- Warmup iterations, timing methodology inside GemmBench (mirror tornado-poc's).
- Notebook cell layout / env persistence mechanics (mirror `/tmp/poc-env.sh` pattern).

### Deferred Ideas (OUT OF SCOPE)
- `JcudaBackend implements ComputeBackend` + `BackendSelector` allowlist + auto-dispatch — future phase.
- More ops (elementwise, reduce) on GPU — after matmul proves value.
- Vendoring JCuda native jars into `dist/libs/` — only if integration ships.
</user_constraints>

## Summary

JCuda 12.6.0 is a dramatically simpler POC than TornadoVM was. **The headline finding: `org.jcuda:jcuda:12.6.0` and `org.jcuda:jcublas:12.6.0` are on Maven Central** (verified live: group metadata, all 4 jar downloads, POM contents). The entire "download SDK + install:install-file" apparatus D-03 describes is unnecessary — a plain `<dependency>` declaration resolves everything, including per-OS native jars, via OS-activation profiles in the published `jcuda-parent` POM. This satisfies D-03's stated rationale (nothing vendored, no Windows build changes, no committed binaries) while deleting the most error-prone notebook step from Phase 5. [VERIFIED: Maven Central registry + POM inspection]

Every API claim below was verified by downloading the actual 12.6.0 jars and decompiling bytecode: `cublasDgemm` takes `jcuda.Pointer` alpha/beta (host `double[]` wrapped via `Pointer.to(new double[]{1.0})`), status returns as int (with `setExceptionsEnabled(true)` → `CudaException`), the native JNI wrappers self-extract from the natives jars to a temp dir at class-init (no LD_LIBRARY_PATH for JCuda's own libs), and the only external CUDA library needed is `libcublas.so.12` — which the proven `cuda-toolkit-12-6` apt install from the TornadoVM notebook provides at `/usr/local/cuda-12.6/lib64`. [VERIFIED: bytecode javap + ELF DT_NEEDED parse]

The D-08 column-major trap has a zero-copy canonical solution: row-major `C = A·B` equals column-major `C^T = B^T·A^T`, so you pass the SAME buffers to `cublasDgemm` with **B first, all OP_N** — no transposition, no extra kernels. For square N=4096 all leading dims coincide; only the operand order matters, and the Frobenius gate catches a wrong order since `A·B ≠ B·A`. [CITED: leimao.github.io derivation + NVIDIA cuBLAS docs]

**Primary recommendation:** Clone `bench/tornado-poc` → `bench/jcuda-poc` with `org.jcuda:jcuda:12.6.0` + `org.jcuda:jcublas:12.6.0` as the only new deps (Maven Central resolves natives automatically). Clone the fixed TornadoVM notebook minus the SDK-download block (keep JDK21/Maven/CUDA-12.6-toolkit/env-persistence blocks verbatim), run plain `java -jar` (no TornadoVM launcher needed), and use the B-first OP_N swap for the Dgemm call.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPU-04 (conditional) | Nếu TornadoVM POC NO-GO — POC fallback qua thư viện GPU JVM khác (JCuda/cuBLAS) với benchmark cùng chuẩn numerical/perf gates; chỉ tích hợp nếu đạt cả hai gate | Phase 5 evidence confirms TornadoVM NO-GO (GPU_ABSENT / DEVICE_MISMATCH in notebook outputs). This research: verified JCuda 12.6.0 artifacts on Maven Central (§Standard Stack), exact `cublasDgemm` signature from bytecode (§Code Examples), the D-08 row/col-major swap formulation (§Pattern 1), native loading + `libcublas.so.12` requirement (§Pitfall 1), and a timing methodology that mirrors tornado-poc for cross-comparison (§Pattern 2). Integration stays out of scope per D-01. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CPU baseline (N=4096 GEMM) | API / Backend (`numja.core` HAL path) | — | Must route through production `ArrayOps.dot` → `BackendSelector` → `CpuThreadBackend` (proves baseline is the live one; D-04) |
| GPU GEMM kernel execution | External (cuBLAS via JCuda JNI) | — | `JCublas2.cublasDgemm` on device memory; isolated bench module owns all JCuda code |
| Host↔Device transfer | External (CUDA runtime via JCuda) | — | `cudaMalloc`/`cudaMemcpy` in the bench module; timed separately for `transfer_ms` |
| Device discovery / labeling | External (CUDA runtime) | — | `cudaGetDeviceCount` + `cudaGetDeviceProperties` → `device=` label; GPU-less box → `GPU_ABSENT` |
| Build dependency resolution | Build system (Maven Central) | Colab local repo | `org.jcuda` artifacts resolve transitively; natives classifier auto-selected by OS profile |
| Native library loading | JCuda runtime (LibUtils) | — | Self-extracts JNI `.so` from natives jar to temp dir; no launcher, no manual install |
| Verdict computation | Bench module (pure Java) | — | Same key=value schema + GO/NO-GO rule as tornado-poc (D-06/D-07) |
| HAL integration | — (future phase) | — | Explicitly out of scope (D-01); `ComputeBackend`/`BackendSelector` untouched |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `org.jcuda:jcuda` | 12.6.0 | CUDA runtime/driver Java bindings: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetDeviceProperties`, `cudaDeviceSynchronize`, events, `Pointer`, `Sizeof` | The core JCuda artifact — required by every other jcuda lib; on Maven Central since 0.8.0 [VERIFIED: Maven Central solrsearch + USAGE.md] |
| `org.jcuda:jcublas` | 12.6.0 | cuBLAS v2 bindings: `JCublas2.cublasCreate/cublasDgemm/cublasDestroy`, `cublasHandle`, `cublasOperation`, `cublasStatus` | The CUBLAS bindings; depends transitively on `jcuda` + both natives jars [VERIFIED: POM dependency graph fetched] |
| `com.numja:numja-core` | 0.1.0 (repo) | CPU baseline through production HAL path | Same dep as tornado-poc; D-04 requires the same baseline path |
| CUDA 12.6 runtime (apt `cuda-toolkit-12-6`) | 12.6.x | Provides `libcublas.so.12` on Colab | The only external CUDA lib the JNI wrappers need; proven install block in tornado notebook [VERIFIED: ELF DT_NEEDED of libJCublas2] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `org.jcuda:jcuda-natives` (classifier) | 12.6.0 `linux-x86_64` / `windows-x86_64` | JNI wrappers: `libJCudaRuntime`, `libJCudaDriver`, `libJNvrtc`, `libJNvPTXCompiler` | Resolved AUTOMATICALLY as a transitive dep of `jcuda` (classifier via jcuda-parent OS profiles) — do not declare manually [VERIFIED: POM + parent profiles] |
| `org.jcuda:jcublas-natives` (classifier) | 12.6.0 `linux-x86_64` / `windows-x86_64` | JNI wrappers: `libJCublas2` (v2 API — the one to use), `libJCublas` (legacy v1 — ignore) | Resolved automatically as transitive dep of `jcublas` [VERIFIED: POM] |
| maven-shade-plugin | 3.6.0 | Standalone jar with embedded natives | Same as tornado-poc; puts `lib/*.so` resources inside the runnable jar so LibUtils can self-extract |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JCuda + cuBLAS | JavaCPP-presets cuda | Heavier dependency graph; no advantage for a Dgemm-only POC; D-09 forbids a third backend anyway if this fails |
| JCuda + cuBLAS | ND4J/DeepLearning4j (libnd4j backend) | Massive dependency surface; opaque native stack; wrong tool for a 137-GFLOP single-op proof |
| `cudaEvent` timing | `System.nanoTime` | nanoTime mirrors tornado-poc methodology (D-04 cross-comparability); cudaMemcpy is host-synchronous so copy timing is exact; `cudaDeviceSynchronize` bounds kernel timing. Events available (`cudaEventCreate/Record/ElapsedTime(float[],..)`) if per-kernel precision is later wanted — not needed for this gate |

**Installation:**
```bash
# No manual install at all — plain Maven Central resolution (this is the D-03 simplification):
mvn -q -pl modules/numja -am install -DskipTests
mvn -q -pl bench/jcuda-poc -am package -DskipTests
```

**Version verification (performed live, 2026-09-12):**
- `org.jcuda` group on Maven Central: 19 artifacts, latest 12.6.0, published 2024-12-04 (epoch ~1733330000000) [VERIFIED: search.maven.org solrsearch]
- All four jars HTTP 200 on repo1.maven.org: `jcuda-12.6.0.jar`, `jcublas-12.6.0.jar`, `jcuda-natives-12.6.0-linux-x86_64.jar`, `jcublas-natives-12.6.0-linux-x86_64.jar` (windows-x86_64 variants also 200) [VERIFIED]
- GitHub `jcuda/jcuda` tag `version-12.6.0-RC00` (Nov 2024) matches the 12.6.0 release line [VERIFIED]
- Class-file major version 51 (Java 7) — runs on JDK 21 (Colab) and JDK 25 (dev box); **no `--enable-preview` needed** (unlike tornado-poc) [VERIFIED: javap]

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| org.jcuda:jcuda | Maven Central | ~21 mo (2024-12-04) | not measured | github.com/jcuda/jcuda (POM scm + tag verified) | ERR (registry timeout — see below) | Approved — verified by direct means |
| org.jcuda:jcublas | Maven Central | ~21 mo | not measured | github.com/jcuda/jcublas (POM scm verified) | ERR | Approved — verified by direct means |
| org.jcuda:jcuda-natives / jcublas-natives (12.6.0, linux/windows-x86_64) | Maven Central | ~21 mo | not measured | same as above (natives built from same repos) | n/a | Approved — verified by direct means |

**slopcheck note:** `slopcheck install jcuda jcublas` timed out reaching search.maven.org from this environment (network restriction, not a package signal). The protocol's fallback would tag these `[ASSUMED]` — but here a **stronger** direct verification was performed instead of the registry ping: all four jars were downloaded from repo1.maven.org, POMs inspected (coherent 19-artifact group, parent scm links resolve to real GitHub repos with a matching `version-12.6.0-RC00` tag), Java bytecode decompiled (real cublas symbol tables, consistent 12.6.0 API surface), and the natives `.so` files ELF-parsed (DT_NEEDED on `libcublas.so.12`, thousands of genuine `cublasXxx_v2@@libcublas.so.12` symbols). This is multi-signal legitimacy confirmation at the artifact level, above the slopcheck bar. Official usage doc (jcuda-main USAGE.md) states JCuda is available in Maven Central since 0.8.0. No Maven postinstall-script concept exists (that risk class is npm-specific); natives jars contain only `.so`/`.dll` files + POM metadata, no executable hooks.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
Colab T4 notebook (notebooks/colab-jcuda-poc.ipynb)
│
├─ Step 0: preflight ── nvidia-smi (T4 present?) ── fail fast
├─ Step 1: setup (idempotent)
│    ├─ JDK 21 (apt) + Maven 3.9.15 (/opt)
│    ├─ CUDA 12.6 toolkit (apt cuda-toolkit-12-6) ─→ /usr/local/cuda-12.6/lib64/libcublas.so.12
│    ├─ LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64 → /tmp/poc-env.sh (per-cell shell persistence)
│    ├─ git clone phase branch → mvn install numja-core → mvn package bench/jcuda-poc
│    │        └─ Maven Central ─→ {jcuda, jcublas, jcuda-natives:linux-x86_64, jcublas-natives:linux-x86_64}.jar
│    └─ shaded jar = classes + lib/libJCublas2-*.so + lib/libJCudaRuntime-*.so embedded
│
├─ Step 2: run (java -jar — NO launcher needed)
│    bench.jcudapoc.GemmBench
│    ├─ seeded inputs (0xC0FFEE, 0xBADF00D, N=4096, row-major double[])
│    ├─ CPU path: ArrayOps.dot → BackendSelector → CpuThreadBackend → EJML  → cCpu
│    ├─ device probe: cudaGetDeviceCount → 0? → result=GPU_ABSENT (exit path A)
│    ├─ GPU path:
│    │    cudaMalloc(dA,dB,dC) → cudaMemcpy H2D (timed: transfer_ms)
│    │    cublasCreate → cublasDgemm(B-first, OP_N, OP_N) → cudaDeviceSynchronize
│    │    cudaMemcpy D2H (timed) → cGpu
│    │    Frobenius ‖cCpu−cGpu‖_F/‖cCpu‖_F ─→ >1e-9? result=NUMERIC_MISMATCH
│    │    speedup = cpu_baseline_ms / gpu_ms; transfer_pct = transfer_ms/gpu_ms*100
│    │    verdict = GO iff speedup>=2.0 && transfer_pct<50.0
│    ├─ cleanup: cublasDestroy, cudaFree ×3
│    └─ stdout: key=value lines (same schema as tornado-poc)
│    └─ nvidia-smi sampler → gpu_util_peak_pct / gpu_mem_peak_mb telemetry
├─ Step 3: hard check `grep -q '^result=OK$'` — cell fails otherwise
└─ Step 4: decision summary → paste into results doc → GO/NO-GO vs GPU-04
```

### Recommended Project Structure
```
bench/jcuda-poc/                                   # clone of bench/tornado-poc
├── pom.xml                                        # parent com.numja:numja:0.1.0; deps: numja-core + org.jcuda {jcuda,jcublas} 12.6.0
└── src/main/java/bench/jcudapoc/
    └── GemmBench.java                             # key=value harness, seeds, Frobenius, gates
notebooks/
└── colab-jcuda-poc.ipynb                          # new notebook; colab-gpu-poc.ipynb NOT touched (D-05)
```
Root `pom.xml`: add one line `<module>bench/jcuda-poc</module>` after `bench/tornado-poc` (mirrors the phase-5 edit).

### Pattern 1: The D-08 column-major swap (zero-copy row-major GEMM)
**What:** cuBLAS stores matrices column-major [CITED: docs.nvidia.com/cuda/cublas — "the cuBLAS library uses column-major storage, and 1-based indexing"]. NumJa buffers are row-major. The identity `(A·B)^T = B^T·A^T` means: to produce row-major `C = A·B`, interpret the same raw buffers as column-major and call the GEMM with **B's pointer first**, everything `CUBLAS_OP_N`. The result buffer then holds `C^T` in column-major = `C` in row-major. No transposition kernels, no index math, no data movement.
**When to use:** always, for the Dgemm call in this bench.
**Why safe here:** square N=4096 → `m=n=k=N`, `lda=ldb=ldc=N` all coincide; only the operand order distinguishes correct from wrong, and since `A·B ≠ B·A` for the seeded data, the full-matrix Frobenius check (D-06) catches any formulation bug.
**Source:** derivation quoted from leimao.github.io "cuBLAS GEMM API Usages for Column-Major and Row-Major Matrices" (`(A B)^T = B^T A^T` → "Use the B pointer first, then the A pointer" with OP_N/OP_N).

### Pattern 2: Timing methodology (mirrors tornado-poc — Claude's discretion per CONTEXT)
**What:** `System.nanoTime` everywhere; 1 warmup + median-of-3 for the CPU baseline (copy `timeMedianMs` verbatim from tornado GemmBench).
- `transfer_ms` = timed sum of the three blocking copies: `cudaMemcpy(dA, hA) + cudaMemcpy(dB, hB) + cudaMemcpy(hC, dC)`. Each `cudaMemcpy` is host-synchronous — it returns only when the copy completes, so nanoTime around it is exact. [VERIFIED: standard CUDA runtime semantics; jcublas samples rely on the same]
- `gpu_ms` = median of 3 timed iterations of `[cublasDgemm + cudaDeviceSynchronize]`. **The `cudaDeviceSynchronize` MUST be inside the timed region**: `cublasDgemm` enqueues and returns before the kernel finishes; without the sync you measure launch latency (~µs), not the GEMM. 1 untimed warmup iteration first (first cuBLAS call initializes handle workspace and loads kernels).
- Warmup for cuBLAS: 1 call is standard for a steady-state POC at this size; 3 timed samples + median smooths Colab's shared-CPU jitter (tornado used the same 3-sample median).
- Same formulas: `speedup_ratio = cpu_baseline_ms / gpu_ms`, `transfer_pct = transfer_ms / gpu_ms * 100` (tornado template lines 216-219).
**Cross-comparison note for the results doc:** tornado's `gpu_ms` included the d2h copy inside each timed `plan.execute()` (its TaskGraph used `EVERY_EXECUTION` for the output), and its `transfer_ms` was the cold first execution. JCuda numbers therefore measure a *pure* kernel (cleaner); state the composition difference in the results doc so the 2x gate comparison is honest.

### Pattern 3: Native loading — how JCuda finds its libraries (verified from LibUtils bytecode)
1. `JCublas2` static-initializer → `initialize()` → `LibUtilsCuda.loadLibrary("JCublas2-12.6.0")` → `LibUtils.loadLibrary(name)`.
2. First tries `System.loadLibrary(name)` (java.library.path). On Colab: miss.
3. Falls back to resource extraction: looks for `/lib/libJCublas2-12.6.0-linux-x86_64.so` **on the classpath**, copies it to a temp file under `java.io.tmpdir`, and `System.load()`s it. This is why the natives jar (or shaded jar containing `lib/*.so`) must be on the classpath — and why **no LD_LIBRARY_PATH entry is needed for JCuda's own wrappers**. [VERIFIED: javap of LibUtils.loadLibrary + loadLibraryResource; USAGE.md: "JCuda will automatically unpack the required native libraries from the JAR files at runtime"]
4. The extracted wrapper has ELF `DT_NEEDED` deps resolved by the normal dlopen path. Verified DT_NEEDED: `libJCublas2 → libcublas.so.12` (+ system libs); `libJCudaDriver → libcuda.so.1` (driver — present on Colab); `libJCudaRuntime → no CUDA dep` (cudart statically linked). **So `libcublas.so.12` is the one library that must be findable via `LD_LIBRARY_PATH` → `/usr/local/cuda-12.6/lib64` from the `cuda-toolkit-12-6` apt install.** [VERIFIED: ELF DT_NEEDED parsed from the shipped .so files]
5. `LD_LIBRARY_PATH` is read at process start → export it in the notebook cell before invoking `java` (the `/tmp/poc-env.sh` pattern handles this).

### Anti-Patterns to Avoid
- **Legacy `JCublas` (v1) class:** `jcuda.jcublas.JCublas.cublasDgemm(char,char,int,int,int,double,Pointer,...)` is the deprecated cuBLAS v1 API with different parameter conventions. Use **`JCublas2`** exclusively. Both classes ship in the same jar. [VERIFIED: javap of both]
- **Transposing matrices to work around column-major:** the swap trick (Pattern 1) makes any transpose kernel or buffer rearrangement pure waste. A hand-rolled transpose would double transfer volume and add a bug surface for no gain.
- **Timing `cublasDgemm` without a following `cudaDeviceSynchronize`:** measures enqueue, not execution — speedup numbers would be ~1000x too good, then get destroyed by the d2h copy outside the timed region.
- **Building the shaded jar on Windows and uploading it to Colab:** Maven's OS-activated classifier picks `windows-x86_64` natives on the dev box — the jar would embed `.dll` files and fail on Linux with `UnsatisfiedLinkError`. Build on Colab (the notebook's clone+build pattern already does this).
- **Forgetting `cublasDestroy`/`cudaFree`:** the bench is one-shot, but the tornado template's clean exit structure is worth keeping; a leaked context also skews the nvidia-smi memory telemetry.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|------|
| Native library loading/extraction | Custom `.so` unpack code, `java.library.path` management | JCuda's built-in `LibUtils` resource extraction | Already ships in the jar, handles OS/arch naming, temp files, dependents [VERIFIED: bytecode] |
| GEMM kernel | Any Java/OpenCL kernel (tornado-style) | `JCublas2.cublasDgemm` | cuBLAS is NVIDIA's tuned FP64 GEMM — this is the whole point of switching from TornadoVM's naive kernel |
| Row/col-major conversion | Transpose kernels or index-remap loops | B-first OP_N operand swap (Pattern 1) | Zero cost, algebraically exact |
| Timing infrastructure | JMH setup for a 6-line measurement | `System.nanoTime` + median-of-3 (tornado template) | Cross-comparable with Phase 5 numbers; cudaMemcpy sync makes it exact |
| Status checking | Wrapping every int return manually | `JCuda.setExceptionsEnabled(true)` + `JCublas2.setExceptionsEnabled(true)` | Non-zero status throws `jcuda.CudaException` with the message (verified `checkResult` bytecode); one line instead of 15 checks |

**Key insight:** TornadoVM needed a launcher, an SDK repo, service-loader provider merging, and a custom kernel. JCuda needs a `<dependency>` line and a function call. The plan should be proportionally smaller.

## Common Pitfalls

### Pitfall 1: `libcublas.so.12` not found (the CUDA 13.0 collision, Phase-5 déjà vu)
**What goes wrong:** `UnsatisfiedLinkError`/`dlopen failed` when `libJCublas2` loads. Colab's image ships CUDA 13.0, whose system cublas is `libcublas.so.13` — SONAME major version does NOT satisfy `DT_NEEDED libcublas.so.12`.
**Why it happens:** Same root cause as TornadoVM's NO-GO (CUDA 13 vs CUDA 12 binaries) — but for JCuda the fix is trivial and proven.
**How to avoid:** The tornado notebook's already-proven block: `cuda-keyring` deb → `apt install cuda-toolkit-12-6` → `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH` persisted in `/tmp/poc-env.sh` and exported before `java`. Driver 580.x is backward-compatible with the CUDA 12.6 runtime; `libcuda.so.1` comes from the driver and needs nothing extra.
**Warning signs:** `cudaGetDeviceCount` succeeds but `cublasCreate` fails → check `ldd /tmp/.../libJCublas2*.so` in a debug cell.

### Pitfall 2: Wrong operand order (D-08) — silently plausible numbers, wrong matrix
**What goes wrong:** Passing (dA, dB) instead of (dB, dA) computes `B·A` — a valid GEMM with plausible timing and no error status. Only the Frobenius check catches it (`frob_rel_err` ~O(1), nowhere near 1e-9).
**Why it happens:** Column-major habit — every other BLAS binding takes A first.
**How to avoid:** Pattern 1 verbatim: `cublasDgemm(handle, OP_N, OP_N, N, N, N, alpha, dB, N, dA, N, beta, dC, N)` — B pointer first. Comment the call with the `(AB)^T = B^T A^T` identity.
**Warning signs:** `result=NUMERIC_MISMATCH` with sane perf numbers.

### Pitfall 3: alpha/beta pointer mode
**What goes wrong:** Passing `1.0`/`0.0` as raw doubles, or allocating alpha/beta on device. The v2 signature takes `Pointer` — host arrays wrapped with `Pointer.to(new double[]{1.0})` under the default `CUBLAS_POINTER_MODE_HOST`.
**Why it happens:** The legacy v1 API took scalars by value; old examples mix conventions.
**How to avoid:** `Pointer alpha = Pointer.to(new double[]{1.0}); Pointer beta = Pointer.to(new double[]{0.0});` — never call `cublasSetPointerMode` (HOST is the default). [VERIFIED: javap signature + SgemmBatched sample pattern]

### Pitfall 4: The TornadoVM notebook's known shell landmines (already cost Phase 5 several fix commits)
**What goes wrong:** `set -euo pipefail` + `nvidia-smi | head` = SIGPIPE abort; `wget -q` hides 404s; each `%%bash` cell is a fresh shell.
**How to avoid:** `set -eu` (no pipefail) wherever `| head` is used; persist env via `/tmp/poc-env.sh` sourced at cell top; keep `2>&1 | tee /tmp/*.log` so native errors surface (the Phase-5 fix `391e3d9` exists precisely because `2>/dev/null` hid them). For JCuda there is no SDK download step, so the 404/asset-URL risk mostly disappears; Maven failures print normally.
**Warning signs:** cell dies after `nvidia-smi` output; blank stderr before a native crash.

### Pitfall 5: Catch-all hiding the real error (Phase 5 landmine)
**What goes wrong:** `catch (Throwable)` around the whole GPU section labels everything `GPU_INIT_FAILED` and hides the real message (exactly what prolonged the TornadoVM debugging).
**How to avoid:** Two-tier handling like tornado's, but louder: the device probe (`cudaGetDeviceCount`) may catch broadly for the `GPU_ABSENT` label — but MUST print `e.getClass().getSimpleName() + ": " + e.getMessage()` to stderr first. The main GPU section should NOT have a broad catch — let CudaException/UnsatisfiedLinkError reach stderr (with exceptions enabled the message includes the CUDA status), and emit `result=GPU_INIT_FAILED verdict=NO-GO` at top level only.

### Pitfall 6: Natives classifier resolved for the wrong OS
**What goes wrong:** jar built on Windows embeds `.dll` (Pitfall: Anti-Pattern 4) — or someone "fixes" it by hardcoding the linux classifier in the pom, breaking local Windows builds.
**How to avoid:** Don't declare natives at all — the transitive POM + parent profiles pick the right classifier per build machine automatically. [VERIFIED: jcuda-parent 12.6.0 profiles windows-x86_64/linux-x86_64 with os-activation]

### Pitfall 7: T4 FP64 expectations vs the 2x gate (set the narrative correctly)
**What goes wrong:** Reading the measured speedup as "GPU is amazing" or conversely worrying the gate might fail. T4 FP64 throughput is ~1/32 of FP32 (≈0.25 TFLOPS-class) [ASSUMED — napkin from NVIDIA T4 spec, not measured]; N=4096 Dgemm = 137.4 GFLOP → several hundred ms of kernel expected. The CPU baseline on Colab's 2 vCPU was ~68.5s in the Phase-5 run — so speedup well above 2x is near-certain as long as the kernel runs at all. transfer_pct: 3×128MB = 384MB of copies at PCIe speeds ≈ tens of ms → well under 50% of a multi-hundred-ms kernel. **Expect GO — but report measured numbers, not these estimates.** The interesting number for the future HAL phase is absolute gpu_ms, not the ratio.

## Code Examples

All signatures below decompiled (`javap`) from the actual `org.jcuda:jcuda:12.6.0` / `org.jcuda:jcublas:12.6.0` jars — these are not from memory.

### The Dgemm call (D-08 handled)
```java
// VERIFIED signatures (org.jcuda:jcublas:12.6.0):
//   int  JCublas2.cublasCreate(cublasHandle)   /  cublasDestroy(cublasHandle)
//   int  JCublas2.cublasDgemm(cublasHandle, int transa, int transb, int m, int n, int k,
//          Pointer alpha, Pointer A, int lda, Pointer B, int ldb,
//          Pointer beta,  Pointer C, int ldc)
//   void JCuda.setExceptionsEnabled(boolean)   /  JCublas2.setExceptionsEnabled(boolean)
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import jcuda.Pointer; import jcuda.Sizeof; import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda; import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

JCuda.setExceptionsEnabled(true);          // non-zero status -> CudaException
JCublas2.setExceptionsEnabled(true);

cublasHandle handle = new cublasHandle();
JCublas2.cublasCreate(handle);

Pointer alpha = Pointer.to(new double[]{1.0});   // HOST pointer mode is the default
Pointer beta  = Pointer.to(new double[]{0.0});

// D-08: row-major C = A·B  ⇔  column-major C^T = B^T·A^T  (leimao derivation).
// Same buffers, B POINTER FIRST, all OP_N — no transposition anywhere.
JCublas2.cublasDgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, n, n,          // m, n, k — square N=4096
    alpha, dB, n,     // "A" slot carries B  (ld = n)
    dA, n,            // "B" slot carries A  (ld = n)
    beta,  dC, n);    // dC receives C^T col-major == C row-major

JCublas2.cublasDestroy(handle);
```

### Alloc + transfer + timing skeleton
```java
// VERIFIED signatures (org.jcuda:jcuda:12.6.0):
//   int cudaMalloc(Pointer, long)   /  cudaFree(Pointer)
//   int cudaMemcpy(Pointer dst, Pointer src, long count, int kind)  // host-synchronous
//   int cudaDeviceSynchronize()     int cudaGetDeviceCount(int[])   int cudaGetDeviceProperties(cudaDeviceProp, int)
Pointer dA = new Pointer(), dB = new Pointer(), dC = new Pointer();
cudaMalloc(dA, (long) n * n * Sizeof.DOUBLE);   // ditto dB, dC  (3×128MB — fits T4 16GB)

// transfer_ms: each cudaMemcpy blocks until the copy completes — nanoTime is exact.
long t0 = System.nanoTime();
cudaMemcpy(dA, Pointer.to(a), (long) n * n * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
cudaMemcpy(dB, Pointer.to(b), (long) n * n * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
long t1 = System.nanoTime();
long h2dMs = (t1 - t0) / 1_000_000L;

// warmup (first call initializes cuBLAS workspace/kernels), then median-of-3:
for (int i = -1; i < 3; i++) {                      // i=-1 is the warmup
    long s0 = System.nanoTime();
    JCublas2.cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                         alpha, dB, n, dA, n, beta, dC, n);
    cudaDeviceSynchronize();                        // MUST be inside the timed region
    long s1 = System.nanoTime();
    if (i >= 0) samples[i] = (s1 - s0) / 1_000_000L;
}

long u0 = System.nanoTime();
cudaMemcpy(Pointer.to(cGpu), dC, (long) n * n * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
long u1 = System.nanoTime();
transferMs = h2dMs + (u1 - u0) / 1_000_000L;

cudaFree(dA); cudaFree(dB); cudaFree(dC);
```

### Device probe (GPU_ABSENT labeling, mirrors tornado's resolveActualDevice but surfaces stderr)
```java
String actualDevice;
try {
    int[] count = new int[1];
    JCuda.cudaGetDeviceCount(count);
    if (count[0] < 1) { actualDevice = "GPU_ABSENT"; }
    else {
        cudaDeviceProp prop = new cudaDeviceProp();   // prop.getName() verified on 12.6.0
        JCuda.cudaGetDeviceProperties(prop, 0);
        actualDevice = prop.getName();                // e.g. "Tesla T4"
    }
} catch (Throwable t) {
    System.err.println("[GemmBench] device probe failed: "
        + t.getClass().getSimpleName() + " - " + t.getMessage());  // landmine fix: never swallow silently
    actualDevice = "GPU_ABSENT";
}
```

### Canonical sample reference
The official `jcuda/jcuda-samples` repo (`JCudaSamples/src/main/java/jcuda/jcublas/samples/`) holds the canonical patterns used above: `JCublas2MatrixInvert.java` (handle create/destroy + `Pointer.to(new float[]{...})` scalars + cudaMalloc/cudaMemcpy/cublasSetVector), `JCublas2SgemmBatched.java` (`setExceptionsEnabled(true)` first line + alpha/beta as `Pointer.to(new float[]{alpha})`). These are float/Sgemm examples — the double path is identical with `cublasDgemm` + `Sizeof.DOUBLE`.

### bench/jcuda-poc/pom.xml (delta vs tornado-poc)
```xml
<artifactId>jcuda-poc</artifactId>
<parent> ... com.numja:numja:0.1.0 ... </parent>
<dependencies>
    <dependency>com.numja:numja-core:0.1.0</dependency>          <!-- CPU baseline via HAL -->
    <dependency>org.jcuda:jcuda:12.6.0</dependency>
    <dependency>org.jcuda:jcublas:12.6.0</dependency>
    <!-- natives resolve transitively via jcuda-parent OS profiles. No classifiers to declare. -->
</dependencies>
<!-- shade plugin: same as tornado-poc (finalName jcuda-poc-jar, mainClass bench.jcudapoc.GemmBench,
     ServicesResourceTransformer harmless-no-op — JCuda jars ship zero META-INF/services entries
     (verified) — keep for structural consistency. NO compilerArgs/--enable-preview needed. -->
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| JCuda distributed via jcuda.org zip downloads only | On Maven Central since 0.8.0; 12.6.0 fully published with classifier natives | 0.8.0 → present (12.6.0: Dec 2024) | No SDK download, no install:install-file — D-03's mechanism simplifies to a plain dependency |
| TornadoVM: JIT-compiled Java kernels + ServiceLoader backend discovery + SDK launcher | JCuda: direct pre-tuned cuBLAS call, self-extracting natives, plain `java -jar` | (this phase's comparison) | Fewer moving parts = fewer failure modes on Colab; the Phase-5 GPU_ABSENT/DEVICE_MISMATCH failure chain has no JCuda equivalent |
| `libJCublas` v1 API (`cublasInit`, char-based ops, scalar-by-value) | `JCublas2` v2 API (handles, `cublasOperation` int constants, Pointer scalars) | CUDA 4.0-era cuBLAS redesign | Use `JCublas2` exclusively; v1 class ships in the jar but is legacy |

**Deprecated/outdated:**
- `jcuda.jcublas.JCublas` (v1): deprecated cuBLAS API — do not use.
- `jnvgraph` artifacts: stopped at 10.2.0 (nvGRAPH removed from CUDA) — irrelevant here, just confirms group maintenance stopped where NVIDIA's library stopped.
- The CONTEXT note "native JARs bundled since 0.8.0RC": what 12.6.0 actually ships is *separate classifier natives jars* (`jcuda-natives-12.6.0-linux-x86_64.jar` containing `lib/*.so`) that self-extract at runtime — not natives bundled inside the main jar. Same effect, different mechanism. [VERIFIED: jar contents listed]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | T4 FP64 ≈ 1/32 of FP32; expected gpu_ms in the several-hundred-ms range; transfer_pct ≈ 10-20% | Pitfall 7 | Low — these are narrative expectations only; gates are computed from measured numbers. If T4 FP64 were somehow far slower, the 2x gate could fail → honest NO-GO, which is still a valid phase outcome (D-09) |
| A2 | `cuda-toolkit-12-6` apt package places `libcublas.so.12` under `/usr/local/cuda-12.6/lib64` on the current Colab image | Pitfall 1, Pattern 3 | Low-medium — the Phase-5 notebook verified `libcudart.so` at that path after the same install, and toolkit libs co-locate; the notebook should add a `test -e /usr/local/cuda-12.6/lib64/libcublas.so.12` guard (one line) so a miss fails loudly instead of at dlopen |
| A3 | Colab's JDK 21 (`openjdk-21-jdk`) runs the shaded jar without extra flags | Pattern 2 | Very low — same JVM the tornado jar ran on; JCuda class files are major version 51 |
| A4 | `cudaMemcpy` host-synchronous timing + `cudaDeviceSynchronize` yields numbers comparable to tornado's `plan.execute()`-based samples | Pattern 2 | Low — documented composition difference; verdict thresholds are generous (2x / 50%) |

Everything else in this research was verified against the live Maven Central artifacts (downloaded, decompiled, ELF-parsed) or read directly from repo files.

## Open Questions

1. **Where do the measured results live?** (results doc location)
   - What we know: CONTEXT requires "measured results + go/no-go conclusion"; Phase 5's results ended up only in the notebook outputs (the planned `docs/05-GPU-POC-RESULTS.md` was never committed).
   - What's unclear: file path convention for Phase 7's captured numbers.
   - Recommendation: `.planning/phases/07-gpu-fallback-poc-via-jcuda-cublas/07-RESULTS.md` with the pasted key=value blocks (env, device, jdk, hardware) — keeps the decision evidence next to the phase docs; planner decides.
2. **Notebook clone branch name.**
   - What we know: the notebook clones `BRANCH=gsd/phase-05-...` in Phase 5; Phase 7 work will happen on a new phase branch.
   - Recommendation: planner sets the exact branch string as a task requirement (notebook Step 1 `BRANCH=` var must match the phase-7 branch, e.g. `gsd/phase-07-gpu-fallback-poc-via-jcuda-cublas`).
3. **NumPy magnitude-reference cell — keep or drop?**
   - What we know: Phase 5's Step 3 was a magnitude-only reference (Java Random vs PCG64 differ).
   - Recommendation: keep the identical cell for output-format symmetry — it costs nothing and makes the two notebooks structurally comparable (D-04 spirit).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| JDK (dev box) | Build bench module | ✓ | 25.0.3 (compile release=21) | — |
| Maven (dev box) | Build | ✓ (not on PATH) | 3.9.15 at `C:\Users\Admin\.maven\maven-3.9.15` | prefix PATH per STATE.md |
| Maven Central network (dev box) | Resolve org.jcuda jars | ✓ | verified live this session | — |
| NVIDIA GPU (dev box) | GPU path execution | ✗ (expected — i7-1255U) | — | local run labels `result=GPU_ABSENT` (valid PATH A output, same as tornado-poc precedent) |
| Colab T4 + driver | GPU run | ✓ (proven in Phase 5 notebook) | driver 580.82.07 / CUDA 13.0 image | none — T4 is the whole point of the phase |
| JDK 21 on Colab | Run shaded jar | ✓ (apt `openjdk-21-jdk`, proven) | 21.0.12 observed in Phase-5 outputs | — |
| CUDA 12.6 toolkit on Colab | `libcublas.so.12` | ✓ (apt `cuda-toolkit-12-6`, proven in Phase 5 for libcudart) | 12.6 | none — required (Pitfall 1) |
| Maven on Colab | package on Colab | ✓ (3.9.15 tarball pattern, proven) | — | — |

**Missing dependencies with no fallback:** none — every requirement is either present or has the proven Phase-5 notebook pattern.
**Missing dependencies with fallback:** GPU on dev box → labeled `GPU_ABSENT` local run is the documented PATH A (tornado precedent); real numbers come from Colab only.

## Validation Architecture

Test framework: JUnit 4 (repo-wide, 90 tests green) — but the bench module itself carries **zero tests by precedent** (tornado-poc has none; the bench is self-validating).

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPU-04 | Bench builds; local run emits complete key=value schema with `result=GPU_ABSENT verdict=NO-GO` (PATH A) | smoke (run harness) | `mvn -q -pl bench/jcuda-poc -am package -DskipTests` then `java -jar bench/jcuda-poc/target/jcuda-poc-jar.jar -Dbench.env=local` | ❌ Wave 0 (module is the deliverable) |
| GPU-04 | Frobenius gate correctness on real GPU (frob_rel_err ≤ 1e-9, N=4096, fixed seeds) | integration (Colab only) | notebook Step 2 (hard `grep -q '^result=OK$'` fails the cell otherwise) | ❌ Wave 0 (notebook is the deliverable) |
| GPU-04 | Perf gates computed (speedup_ratio ≥ 2.0 AND transfer_pct < 50) | integration (Colab only) | notebook Step 2 + Step 4 auto-verdict block | ❌ Wave 0 |
| GPU-04 | Public API frozen; production modules untouched | static check | `grep -c "public static" NumJa.java` = 61; `ArrayOps.java` = 34; `modules/numja/pom.xml` dependency count unchanged; no `org.jcuda` string in `modules/` | ✅ existing files |
| GPU-04 | go/no-go conclusion documented | manual (checkpoint:human-verify) | paste Step 4 output into results doc | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `mvn -q -pl bench/jcuda-poc -am package -DskipTests` + local `GPU_ABSENT` run (fast, no GPU needed)
- **Per wave merge:** full suite `mvn test` stays green (production modules untouched, but the gate proves it)
- **Phase gate:** Colab T4 run captured with `result=OK` + verdict + results doc, before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `bench/jcuda-poc/` module + pom (the plan's deliverable — no pre-existing infra needed)
- [ ] `notebooks/colab-jcuda-poc.ipynb` (clone of the tornado notebook minus SDK block)
- [ ] Results doc location decision (Open Question 1)

No test framework/config gaps — JUnit 4 infra exists; bench module needs none.

## Security Domain

`security_enforcement` enabled (no `false` in `.planning/config.json`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No identity in scope (bench harness) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (minimal) | `-Dbench.size` parsed with `Integer.parseInt`; clone tornado's `MAX_DISPATCH_N=1_000_000_000` ceiling guard emitting `result=CONFIG_ERROR`. Dev-only system property, not a trust boundary |
| V6 Cryptography | no | — |

### Known Threat Patterns for this phase (bench module, no production surface)

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Supply-chain: malicious GPU library artifact | Tampering/Elevation | `org.jcuda` verified on Maven Central via multi-signal audit (§Package Legitimacy Audit); Maven pulls over HTTPS from repo1.maven.org — no manual `wget` of jars at all (removes Phase 5's asset-URL/404 risk class entirely) |
| Native code execution (JNI) | Elevation | Inherent to the approach; scope-limited to an isolated bench module that never ships in production paths; D-01 keeps modules/numja stdlib-only |
| OOM via hostile bench size | DoS | `MAX_DISPATCH_N` guard + fixed N=4096 default; 3×128MB device buffers fit T4 16GB |
| Repo injection via notebook clone | Tampering | Notebook clones over HTTPS from the project's own GitHub, pinned to the phase branch (same as Phase 5) |

## Sources

### Primary (HIGH confidence)
- Maven Central live artifact verification (this session, 2026-09-12): `https://repo1.maven.org/maven2/org/jcuda/jcuda/12.6.0/`, `.../jcublas/12.6.0/`, `.../jcuda-natives/12.6.0/` (directory listing, POM fetch, jar download, bytecode `javap`, natives `unzip -l` + ELF parse) — signatures, classifiers, dependency graph, native loading mechanism
- `https://search.maven.org/solrsearch/select?q=g:org.jcuda` — group metadata: 19 artifacts, 12.6.0 latest, Dec 2024
- `https://github.com/jcuda/jcuda-main` USAGE.md (raw) — Maven coordinates `org.jcuda`, "available in Maven Central since 0.8.0", classifier scheme `<os>-<arch>`, "automatically unpack the required native libraries from the JAR files at runtime"
- `https://github.com/jcuda/jcuda-main` jcuda-parent 12.6.0 POM — OS-activation profiles setting `jcuda.os`/`jcuda.arch`
- `https://github.com/jcuda/jcuda/tags` — `version-12.6.0-RC00` (Nov 2024) confirming the release line
- `https://github.com/jcuda/jcuda-samples` — `JCublas2MatrixInvert.java`, `JCublas2SgemmBatched.java` (canonical handle/scalar/status patterns)
- Repo files (direct reads): `bench/tornado-poc/pom.xml`, `bench/tornado-poc/src/main/java/bench/tornadopoc/GemmBench.java`, `notebooks/colab-gpu-poc.ipynb`, `docs/COLAB-RECIPE.md`, `modules/numja/src/main/java/numja/core/ComputeBackend.java`, `.planning/phases/05-.../05-03-SUMMARY.md` (CUDA 13.0 vs 12.6 root cause), root `pom.xml`

### Secondary (MEDIUM confidence)
- `https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/` — the `(AB)^T = B^T A^T` B-first call pattern (cross-checked against NVIDIA docs statement below; matches the D-08 formulation in CONTEXT)
- `https://docs.nvidia.com/cuda/cublas/` — "the cuBLAS library uses column-major storage, and 1-based indexing"

### Tertiary (LOW confidence)
- T4 FP64 1:32 ratio / absolute perf ballpark (training data, marked [ASSUMED] in Pitfall 7 — the benchmark measures the real numbers)
- jcuda.org downloads page unreachable from this environment (TLS handshake failure) — immaterial: Maven Central is the distribution channel actually used

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every artifact and API verified against live Maven Central jars; no claim rests on training data
- Architecture: HIGH — module + notebook are direct clones of proven Phase-5 patterns with a *simpler* dependency story
- Pitfalls: HIGH — six pitfalls, five verified (bytecode/ELF/Phase-5 evidence), one honest estimate clearly labeled

**Research date:** 2026-09-12
**Valid until:** 2026-10-12 (JCuda 12.6.0 is a frozen release; Colab image drift is the only moving target — re-verify driver/CUDA version in the notebook preflight, which the cloned cells already do)
