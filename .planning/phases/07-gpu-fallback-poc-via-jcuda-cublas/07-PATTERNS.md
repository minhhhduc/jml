# Phase 7: GPU Fallback POC via JCuda/cuBLAS - Pattern Map

**Mapped:** 2026-09-12  
**Files analyzed:** 5  
**Analogs found:** 4 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `bench/jcuda-poc/pom.xml` | config | build/package | `bench/tornado-poc/pom.xml` | exact structure |
| `bench/jcuda-poc/src/main/java/bench/jcudapoc/GemmBench.java` | service / benchmark harness | batch | `bench/tornado-poc/src/main/java/bench/tornadopoc/GemmBench.java` | exact harness, external API differs |
| `pom.xml` | config | build/package | current root `pom.xml` | exact |
| `notebooks/colab-jcuda-poc.ipynb` | config / runbook | batch / process I/O | `notebooks/colab-gpu-poc.ipynb` | exact notebook flow, runtime differs |
| `.planning/phases/07-gpu-fallback-poc-via-jcuda-cublas/07-RESULTS.md` | evidence document | transform | none | no analogous committed results document |

## Pattern Assignments

### `bench/jcuda-poc/pom.xml` (config, build/package)

**Analog:** `bench/tornado-poc/pom.xml`

**Parent, artifact, and isolated CPU-baseline dependency** (lines 8-32):
```xml
<parent>
    <groupId>com.numja</groupId>
    <artifactId>numja</artifactId>
    <version>0.1.0</version>
    <relativePath>../../pom.xml</relativePath>
</parent>
<artifactId>tornado-poc</artifactId>
<packaging>jar</packaging>
<dependency>
    <groupId>com.numja</groupId>
    <artifactId>numja-core</artifactId>
    <version>0.1.0</version>
</dependency>
```
Copy the layout, rename artifact/final jar to `jcuda-poc`, retain `numja-core`, and add only pinned Maven Central dependencies `org.jcuda:jcuda:12.6.0` and `org.jcuda:jcublas:12.6.0`. Do **not** declare native classifiers or use `install:install-file`: Maven resolves OS-native transitives.

**Runnable shaded-jar pattern** (lines 54-108):
```xml
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-shade-plugin</artifactId>
  <version>3.6.0</version>
  <executions><execution><phase>package</phase><goals><goal>shade</goal></goals>
    <configuration><finalName>tornado-poc-jar</finalName>
      <transformers>
        <transformer implementation="org.apache.maven.plugins.shade.resource.ServicesResourceTransformer"/>
        <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
          <mainClass>bench.tornadopoc.GemmBench</mainClass>
        </transformer>
      </transformers>
    </configuration>
  </execution></executions>
</plugin>
```
Keep the shade/plugin/filter structure and change final name/main class. Remove Tornado-specific dependencies and compiler `--enable-preview`; research verifies ordinary Java 21 execution.

### `bench/jcuda-poc/src/main/java/bench/jcudapoc/GemmBench.java` (service / benchmark harness, batch)

**Analog:** `bench/tornado-poc/src/main/java/bench/tornadopoc/GemmBench.java`

**Imports and CPU baseline through production HAL** (lines 15-18, 68-85):
```java
import numja.core.ArrayOps;
import numja.core.NDArray;
import java.util.Arrays;

final double[] a = seededRandom(n * n, 0xC0FFEEL);
final double[] b = seededRandom(n * n, 0xBADF00DL);
final double[] cCpu = new double[n * n];
final double cpuBaselineMs = timeMedianMs(3, () -> {
    final NDArray A = new NDArray(aRows);
    final NDArray B = new NDArray(bRows);
    final NDArray C = ArrayOps.dot(A, B);
    System.arraycopy(C.getData().data, 0, cCpu, 0, n * n);
});
emit("cpu_baseline_ms", String.valueOf(cpuBaselineMs));
```
Retain this baseline, fixed seeds, reshape helper, and `java.util.Arrays`; replace all Tornado imports/types with JCuda 12.6 `Pointer`, `Sizeof`, `JCuda`, `JCublas2`, `cublasHandle`, and static `CUBLAS_OP_N`/`cudaMemcpyKind` imports.

**Input guard and machine-readable failure output** (lines 53-66, 87-97):
```java
final int n = Integer.parseInt(System.getProperty("bench.size", "4096"));
final long cells = (long) n * n;
if (cells > MAX_DISPATCH_N) {
    System.out.println("result=CONFIG_ERROR");
    System.out.println("verdict=NO-GO");
    System.out.println("error_msg=size " + n + " exceeds MAX_DISPATCH_N=" + MAX_DISPATCH_N);
    return;
}
```
Keep this guard and key=value contract. Replace the Tornado opt-in property branch with a JCuda `cudaGetDeviceCount` probe: no device or a probe exception emits `GPU_ABSENT`/`NO-GO`, logs its class/message to stderr, and emits the existing tail metadata.

**Timing, numerical gate, verdict, and helpers** (lines 151-159, 166-230, 281-326):
```java
final long[] samples = new long[iters];
for (int i = 0; i < iters; i++) {
    final long s0 = System.nanoTime();
    plan.execute();
    final long s1 = System.nanoTime();
    samples[i] = (s1 - s0) / 1_000_000L;
}
gpuMs = median(samples);

final double frobRelErr = Math.sqrt(sse) / Math.max(Math.sqrt(frobCpu), eps);
emit("cpu_vs_gpu_frob_rel_err", String.format("%.3e", frobRelErr));
if (frobRelErr > 1e-9) result = "NUMERIC_MISMATCH";
final String verdict = (speedup >= 2.0 && transferPct < 50.0) ? "GO" : "NO-GO";
```
Copy `timeMedianMs`, `median`, `seededRandom`, the full Frobenius metric/output set, `emit`, and `emitTail` (rename `tornado.device` to a JCuda-appropriate label or omit it consistently). For GPU timing use one untimed cuBLAS warmup then median-of-three `[cublasDgemm + cudaDeviceSynchronize]`; time H2D plus D2H copies separately as `transfer_ms`.

**JCuda-specific core pattern (from verified research; no codebase analog):** enable exceptions, create a handle, allocate/copy three device buffers, then call `JCublas2.cublasDgemm` with **`dB` first, then `dA`**, `CUBLAS_OP_N` for both operands, host `Pointer.to(new double[]{1.0})`/`Pointer.to(new double[]{0.0})`, and `cudaDeviceSynchronize` inside the timer. This is the required zero-copy row-major mapping `(A B)^T = B^T A^T`. Always destroy the handle and free all three buffers; do not broadly catch the GPU execution section—allow native/CUDA errors to reach stderr and label the top-level outcome `GPU_INIT_FAILED`.

### `pom.xml` (config, build/package)

**Analog:** `pom.xml`

**Reactor module list** (lines 65-73):
```xml
<modules>
    <module>modules/numja</module>
    <module>modules/matplotlib</module>
    <module>modules/seaborn</module>
    <module>modules/pandas</module>
    <module>modules/sklearn</module>
    <module>bench</module>
    <module>bench/tornado-poc</module>
</modules>
```
Add exactly one sibling entry, `<module>bench/jcuda-poc</module>`, immediately after `bench/tornado-poc`. Do not alter production modules, root dependency management, or compiler settings.

### `notebooks/colab-jcuda-poc.ipynb` (config / runbook, batch/process I/O)

**Analog:** `notebooks/colab-gpu-poc.ipynb`

**Idempotent Colab setup and per-cell environment persistence** (setup cell lines 20-103):
```bash
set -Eeu
trap 'echo "ERROR: Step 1 failed at line $LINENO: $BASH_COMMAND" >&2' ERR
CUDA12_LIB=/usr/local/cuda-12.6/lib64
# install cuda-toolkit-12-6 when required
cat > /tmp/poc-env.sh <<EOF
export JAVA_HOME=$JAVA_HOME
export PATH=$MVN_DIR/bin:\$PATH
export LD_LIBRARY_PATH=$CUDA12_LIB:\${LD_LIBRARY_PATH:-}
EOF
```
Copy the JDK 21, Maven 3.9.15, CUDA 12.6, branch-clone, `/tmp/poc-env.sh`, and build flow. Delete every Tornado SDK/launcher/local-repository setup section. Add a hard `test -e "$CUDA12_LIB/libcublas.so.12"` guard before running Java. Build on Colab, not Windows, so Maven selects Linux-native JCuda artifacts.

**Run, telemetry, hard success gate, and decision extraction** (Step 2 lines 104-149; Step 4 lines 169-186):
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 -f /tmp/nv-sample.log &
# run benchmark with 2>&1 | tee /tmp/poc-gpu.log
awk -F', ' 'NR>1 { ... }' /tmp/nv-sample.log
grep -q '^result=OK$' /tmp/poc-gpu.log || { echo "ERROR: GPU benchmark did not complete successfully."; exit 1; }
```
Keep the T4 preflight, telemetry sampler, `2>&1 | tee`, hard `result=OK` gate, NumPy magnitude-only cell, and the key=value decision summary. Use `java -jar bench/jcuda-poc/target/jcuda-poc-jar.jar`; no Tornado launcher or `-Dtornado.device`. Use `set -eu`, not `pipefail`, in cells piping `nvidia-smi` into `head`.

**Security exclusion:** do not copy the unrelated credential-bearing outbound `curl` cell at notebook lines 26-35. It is outside the POC and must be omitted from the new notebook; its exposed credential should be revoked independently.

### `.planning/phases/07-gpu-fallback-poc-via-jcuda-cublas/07-RESULTS.md` (evidence document, transform)

**Analog:** none. No committed Phase 5 results document exists; `07-VALIDATION.md` lines 35-38 establishes the required outcome.

Create the minimal evidence record after the Colab run: execution date, Colab/T4/JDK/CUDA environment, the CPU/GPU key=value blocks, telemetry, Frobenius result, speedup and transfer calculations, an explicit GPU-04 GO/NO-GO conclusion, and a statement that production paths were untouched. State that JCuda `gpu_ms` is pure kernel timing while the Phase 5 Tornado timing included output transfer, so cross-POC composition differs.

## Shared Patterns

### Output and gate contract
**Source:** `bench/tornado-poc/src/main/java/bench/tornadopoc/GemmBench.java` lines 85, 187-230, 311-326.  
**Apply to:** JCuda `GemmBench`, notebook logs, and results evidence. Preserve key=value stdout, the full-matrix Frobenius check (`<= 1e-9`), and the `speedup >= 2.0 && transfer_pct < 50.0` GO rule.

### CPU baseline isolation
**Source:** `bench/tornado-poc/src/main/java/bench/tornadopoc/GemmBench.java` lines 73-84.  
**Apply to:** JCuda benchmark only. Route CPU measurement through `ArrayOps.dot`; never add JCuda imports/dependencies under `modules/`.

### Native failure visibility
**Source:** `notebooks/colab-gpu-poc.ipynb` Step 2 lines 104-149 and `GemmBench.java` lines 199-210.  
**Apply to:** GPU probe, notebook runner, and results capture. Preserve stderr with `2>&1 | tee`; log probe failures before `GPU_ABSENT`; do not hide CUDA/`UnsatisfiedLinkError` messages behind catch-all handling.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `.planning/phases/07-gpu-fallback-poc-via-jcuda-cublas/07-RESULTS.md` | evidence document | transform | Phase 5 stored evidence in notebook output; no committed results-document format exists. |

## Metadata

**Analog search scope:** root reactor POM, `bench/tornado-poc`, `notebooks/colab-gpu-poc.ipynb`, Phase 7 planning files  
**Files scanned:** 6  
**Pattern extraction date:** 2026-09-12
