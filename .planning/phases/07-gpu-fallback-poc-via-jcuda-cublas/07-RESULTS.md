# Phase 7 JCuda/cuBLAS POC Results

**Status: PENDING FUTURE EXECUTION**

This is an evidence template, not a measurement record. Do not fill it until the deferred protocol below has completed. It makes no current GO/NO-GO claim.

## Scope Note

**D-03 (revised 2026-09-11):** Code-only in this session: do not manually download JCuda JARs, run `mvn install:install-file`, resolve Maven dependencies, or run Colab. Declare the pinned Maven Central coordinates (`org.jcuda:jcuda:12.6.0`, `org.jcuda:jcublas:12.6.0`) in the isolated bench POM; a future build/Colab run may resolve them. Nothing is vendored into `dist/libs/`, no Windows build-pipeline changes, and no committed binaries.

This session performed none of those future actions. Production modules remain JCuda-free; only `bench/jcuda-poc` is in scope.

## Deferred Execution Metadata

- **Execution date (UTC):** PENDING
- **Commit tested:** PENDING
- **Runner:** PENDING (local or Colab T4)
- **GPU / T4 model:** PENDING
- **NVIDIA driver:** PENDING
- **JDK:** PENDING
- **CUDA runtime:** PENDING
- **`libcublas.so.12` path:** PENDING
- **Maven version:** PENDING
- **Benchmark size:** PENDING (expected `4096`)

## Future Manual-Only Protocol

Do not execute these steps in this source-only session.

1. On the intended Linux environment, the future Maven package command must succeed before any benchmark assertion:
   ```bash
   mvn -q -pl modules/numja -am install -DskipTests
   mvn -q -pl bench/jcuda-poc -am package -DskipTests
   ```
   Stop on any failed command. Do not use `|| true`, a status fallback, or another construct that masks a packaging failure.
2. Only after successful packaging, run the local shaded JAR with stdout and stderr captured:
   ```bash
   java -Dbench.env=local -Dbench.size=4096 -jar bench/jcuda-poc/target/jcuda-poc-jar.jar 2>&1 | tee /tmp/poc-local.log
   ```
   Retain `/tmp/poc-local.log` unchanged as evidence.
3. For the future local no-GPU validation, require every D-04 key listed below, exactly one local `result=GPU_ABSENT` **or** exactly one local `result=GPU_INIT_FAILED`, exactly one `verdict=NO-GO`, and no `result=OK`. Any other result is an investigation failure, not a substitute for this check.
4. If checking that legacy `JCublas` output or code is absent, use a standalone failure-on-match conditional. Do not attach an exit-status fallback that could hide a failed preceding command:
   ```bash
   if grep -q 'jcuda\.jcublas\.JCublas' /tmp/poc-local.log; then
     echo 'ERROR: legacy JCublas v1 was observed' >&2
     exit 1
   fi
   ```
5. For the future Colab T4 run, use `notebooks/colab-jcuda-poc.ipynb`, preserve `/tmp/poc-gpu.log`, and retain its stderr, telemetry, and environment metadata. Its hard gate requires `result=OK`; do not treat incomplete output as success.
6. Record GO only when `result=OK`, `cpu_vs_gpu_frob_rel_err <= 1e-9`, `speedup_ratio >= 2.0`, and `transfer_pct < 50.0`. Otherwise retain native/device/numeric/performance evidence and record NO-GO.
7. Per D-09, if JCuda/cuBLAS fails, do not try a third backend in this phase. Defer GPU work to a future milestone.

## Future Complete D-04 Key=Value Log

Paste the complete key=value output from the accepted future run here. Every field is required.

```text
cpu_baseline_ms=PENDING
gpu_ms=PENDING
transfer_ms=PENDING
speedup_ratio=PENDING
transfer_pct=PENDING
device=PENDING
result=PENDING
verdict=PENDING
env=PENDING
jdk=PENDING
hardware=PENDING
size=PENDING
cpu_vs_gpu_frob_rel_err=PENDING
cpu_vs_gpu_max_abs_err=PENDING
cpu_vs_gpu_max_rel_err=PENDING
cpu_vs_gpu_mae=PENDING
```

## Future Stderr

```text
PENDING — paste complete stderr retained by the captured log.
```

## Future GPU Telemetry

```text
gpu_telemetry_samples=PENDING
gpu_util_peak_pct=PENDING
gpu_util_mean_pct=PENDING
gpu_mem_peak_mb=PENDING
gpu_mem_mean_mb=PENDING
```

## Future Accuracy and Timing Analysis

- **Frobenius relative error:** PENDING (`cpu_vs_gpu_frob_rel_err <= 1e-9` required)
- **Maximum absolute error:** PENDING
- **Maximum relative error:** PENDING
- **Mean absolute error:** PENDING
- **Speedup calculation:** PENDING (`cpu_baseline_ms / gpu_ms`; must be `>= 2.0`)
- **Transfer calculation:** PENDING (`transfer_ms / gpu_ms * 100`; must be `< 50.0`)
- **Timing-composition caveat:** JCuda `gpu_ms` is kernel-only Dgemm time bounded by device synchronization; `transfer_ms` is H2D plus D2H time. The Phase 5 TornadoVM timing included output transfer in its timed execution, so cross-POC absolute GPU timing composition differs.

## Future Conclusion

**Conclusion: GO|NO-GO**

PENDING — state the evidence-backed conclusion only after all required logs, environment details, numerical gate, and performance gate are recorded. Production API and production dependencies remain unchanged regardless of this POC outcome.
