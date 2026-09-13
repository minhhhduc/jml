# Phase 7 JCuda/cuBLAS POC Results

**Status: MEASURED — GO (2026-09-13, Colab T4)**

## Scope Note

**D-03 (revised 2026-09-11):** Code-only in this session: do not manually download JCuda JARs, run `mvn install:install-file`, resolve Maven dependencies, or run Colab. Declare the pinned Maven Central coordinates (`org.jcuda:jcuda:12.6.0`, `org.jcuda:jcublas:12.6.0`) in the isolated bench POM; a future build/Colab run may resolve them. Nothing is vendored into `dist/libs/`, no Windows build-pipeline changes, and no committed binaries.

Production modules remain JCuda-free; only `bench/jcuda-poc` is in scope.

## Execution Metadata

- **Execution date (UTC):** 2026-09-13 07:11 (notebook `nvidia-smi` timestamp)
- **Commit tested:** branch head `3f1260a` on `gsd/phase-07-gpu-fallback-poc-via-jcuda-cublas` (fresh single-branch clone; evidence retained in the executed `notebooks/colab-jcuda-poc.ipynb` committed alongside this file)
- **Runner:** Colab T4 (`notebooks/colab-jcuda-poc.ipynb`)
- **GPU / T4 model:** Tesla T4 (driver reports CUDA 13.0 capability)
- **NVIDIA driver:** 580.82.07
- **JDK:** 21.0.12
- **CUDA runtime:** toolkit 12.6 (`cuda-toolkit-12-6`, `libcublas 12.6.4.1-1` from ubuntu2204 repo)
- **`libcublas.so.12` path:** `/usr/local/cuda-12.6/lib64`
- **Maven version:** 3.9.15
- **Benchmark size:** 4096

## Executed Protocol

1. Maven build succeeded without quiet flags (fail-on-error `mvn -B`, no `|| true`); shaded jar built at `bench/jcuda-poc/target/jcuda-poc-jar.jar` (shade warning: overlapping `META-INF/MANIFEST.MF` resources — cosmetic).
2. Colab run captured stdout+stderr via `tee /tmp/poc-gpu.log`; hard gates: exactly one `result=OK` **and** in-process `cuda_alloc_delta_mib > 0 && >= 0.5 * cuda_alloc_expected_mib`.
3. Legacy `JCublas` v1 check: `JCublas2` used in source; no `jcuda.jcublas.JCublas` reference in `GemmBench.java`.
4. Local no-GPU validation (`result=GPU_ABSENT` path) **not performed** — local JAR execution is out of scope per D-03; the GPU_ABSENT/GPU_INIT_FAILED code paths remain source-verified only.
5. External `nvidia-smi` telemetry demoted to informational (see Telemetry note); it never gates.

## D-04 Key=Value Log (measured)

```text
cpu_baseline_ms=67340.383673
gpu_ms=548.583396
transfer_ms=95.084199
speedup_ratio=122.75322979881076
transfer_pct=17.332678986149993
device=Tesla T4
result=OK
verdict=GO
env=colab
jdk=21.0.12
hardware=Linux+amd64+2cores
size=4096
cpu_vs_gpu_frob_rel_err=9.848062e-17
cpu_vs_gpu_max_abs_err=9.094947e-13
cpu_vs_gpu_max_rel_err=8.881206e-16
cpu_vs_gpu_mae=5.426763e-14
```

Per-sample timings (medians gate the run):

```text
cpu_sample_1_ms=6.725854e+04
cpu_sample_2_ms=6.856316e+04
cpu_sample_3_ms=6.734038e+04
gpu_sample_1_ms=5.487216e+02
gpu_sample_2_ms=5.484919e+02
gpu_sample_3_ms=5.485834e+02
```

## In-Process CUDA Device-Memory Evidence (hard gate)

Measured inside the JVM with `cudaMemGetInfo` around the three `cudaMalloc` calls (before any `cudaFree`):

```text
cuda_device_index=0
cuda_total_mib=1.491269e+04
cuda_free_mib_before_alloc=1.480781e+04
cuda_free_mib_after_alloc=1.442381e+04
cuda_alloc_delta_mib=3.840000e+02
cuda_alloc_expected_mib=3.840000e+02
```

Delta matches expectation exactly: 3 × 4096² × 8 B = 402,653,184 B = 384.0 MiB. The three Dgemm buffers (`dA`, `dB`, `dC`) verifiably reside on device 0 (Tesla T4, ~14.9 GB total).

## Stderr (captured)

```text
[0.000s][warning][os,container] Cgroup memory controller path at '/sys/fs/cgroup' seems to have moved to '/../../jupyter-children', detected limits won't be accurate
[0.001s][warning][os,container] Cgroup cpu controller path at '/sys/fs/cgroup' seems to have moved to '/../../jupyter-children', detected limits won't be accurate
```

JVM container-detection warnings only; no CUDA/cuBLAS/numeric errors.

## GPU Telemetry (informational — not a gate)

```text
gpu_telemetry_samples=2594 (at run time) / 2700 (post-run read of same log)
gpu_util_peak_pct=0 (at run time) / 100 (post-run read)
gpu_mem_peak_mb=0 (at run time) / 501 (post-run read)
```

**Telemetry note:** two reads of the same `nvidia-smi` sampler disagreed (all-zeros vs 100%/501 MB), confirming external telemetry is unreliable in this Colab container (`[N/A]` fields coerced to 0 by earlier awk). This is why the hard gate uses in-process `cudaMemGetInfo` evidence. The 501 MB peak is consistent with 384 MiB buffers + cuBLAS workspace + CUDA context.

## Accuracy and Timing Analysis

- **Frobenius relative error:** 9.848062e-17 — PASS (`<= 1e-9` required; 8 orders below gate). Absolute Frobenius error ≈ 9.848e-17 × ‖C‖_F(4.19e6) ≈ 4.1e-10 over 16.7M entries.
- **Maximum absolute error:** 9.094947e-13 — consistent with double-precision round-off across a 4096-term dot product (expected accumulation ~1e-11); CPU and GPU sum in different orders.
- **Maximum relative error:** 8.881206e-16.
- **Mean absolute error:** 5.426763e-14.
- **Layout proof:** a row/column-major mapping error would produce `frob_rel_err` ~O(1); 9.8e-17 confirms the `(A×B)^T = B^T×A^T` mapping.
- **Speedup:** 67340.38 / 548.58 = **122.75x** — PASS (`>= 2.0`). Caveat: CPU baseline ran on a 2-vCPU Colab container (`hardware=Linux+amd64+2cores`) with single-library EJML `ArrayOps.dot`; this is "T4 vs 2-core container", not a tuned-CPU comparison. GPU samples are tight (548.49–548.72 ms, 0.04% spread).
- **Transfer:** 95.08 / 548.58 × 100 = **17.33%** — PASS (`< 50`). H2D 2×128 MiB + D2H 128 MiB.
- **Timing-composition caveat:** `gpu_ms` is kernel-only Dgemm bounded by `cudaDeviceSynchronize`; `transfer_ms` is H2D+D2H, kept separate. Cross-POC comparisons with Phase 5-style inclusive timings are invalid.

## Conclusion

**Conclusion: GO**

All defined gates pass with measured evidence: exactly one `result=OK`, `cpu_vs_gpu_frob_rel_err = 9.848e-17 <= 1e-9`, `speedup_ratio = 122.75 >= 2.0`, `transfer_pct = 17.33 < 50.0`, and direct in-process device-memory evidence (`cuda_alloc_delta_mib = 384.0`, equal to expected).

**Scope of this GO:** the POC demonstrates JCuda 12.6 + cuBLAS Dgemm is numerically correct and dramatically faster on a free Colab T4 at N=4096, with an isolated bench module and no production-module changes.

**Not yet established by this GO (prerequisites for any HAL integration decision):**
1. CPU baseline is a 2-vCPU container — speedup on representative hardware (e.g. the 10-core dev machine) will be materially lower.
2. Only N=4096 measured. `transfer_ms` (~95 ms, size-proportional) is a fixed cost per call, so GPU loses below some N threshold; that dispatch threshold is unmeasured.
3. T4 is not a deployment target; no Windows GPU run exists.

Production API (NumJa 61 / ArrayOps 34) and production dependencies remain unchanged regardless of this outcome. Per D-09, no third GPU backend was attempted.
