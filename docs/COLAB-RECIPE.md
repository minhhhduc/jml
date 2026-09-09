# Colab NVIDIA T4 GPU Recipe for `bench/tornado-poc`

This recipe runs the Phase 5 GPU POC on Google Colab's free NVIDIA T4 runtime.
The same shaded jar (`bench/tornado-poc/target/tornado-poc-jar.jar`) produced
locally works unmodified on Colab — only the JVM flags differ.

## Prerequisites

- Local Maven build produces the shaded jar (see Step 1 below).
- Google account with Colab access (free tier is fine).
- This recipe pins versions per `.planning/phases/01-baseline-benchmark/VERSIONS.md`
  (verified live 2026-08-26): TornadoVM `5.2.0-jdk21`, OpenCL/CUDA driver via
  Colab's pre-installed NVIDIA stack.

## Step 1 — Build the POC jar locally

From the repo root:

```bash
mvn -q -pl bench/tornado-poc -am package -DskipTests
ls -lh bench/tornado-poc/target/tornado-poc-jar.jar
```

Expect: BUILD SUCCESS and the shaded jar present. If `Could not resolve
dependency uk.ac.manchester.tornado:tornado-api:5.2.0-jdk21`, Maven cannot
reach Maven Central — confirm network and rerun.

## Step 2 — Open Colab

1. Go to https://colab.research.google.com/ and create a new notebook.
2. Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save.
3. (Optional) Rename the notebook to `tornado-poc.ipynb`.

## Step 3 — Install JDK 21

The POC module is compiled with `--release 21`; TornadoVM 5.2.0-jdk21 is built
against OpenJDK 21. The Colab default is JDK 11 — install JDK 21 in a cell:

```bash
!apt-get update -qq
!apt-get install -y openjdk-21-jdk-headless
!update-alternatives --set java /usr/lib/jvm/java-21-openjdk-amd64/bin/java
!java -version
```

Expect: `openjdk version "21..."` printed.

## Step 4 — Confirm NVIDIA driver

```bash
!nvidia-smi
```

Expect: a T4 row in the table. If absent, the runtime type didn't switch —
go back to Step 2.

## Step 5 — Install TornadoVM 5.2.0-jdk21 (CUDA + OpenCL backends)

TornadoVM ships its own OpenCL driver layer on top of the NVIDIA CUDA stack
Colab already provides. Download and extract:

```bash
!wget -q https://github.com/beehive-lab/TornadoVM/releases/download/v5.2.0/tornadovm-5.2.0-jdk21-cuda-linux-amd64.tar.gz
!tar -xzf tornadovm-5.2.0-jdk21-cuda-linux-amd64.tar.gz
!ls tornadovm-5.2.0-jdk21-cuda/bin/
```

Set env vars (TornadoVM 5.2.0 ships a `setenv.sh`):

```python
import os
os.environ['TORNADO_SDK'] = '/content/tornadovm-5.2.0-jdk21-cuda'
os.environ['PATH'] = f"{os.environ['TORNADO_SDK']}/bin/tornado:" + os.environ['PATH']
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['TORNADO_SDK']}/lib:" + os.environ.get('LD_LIBRARY_PATH','')
```

## Step 6 — Upload the POC jar

From your local box:

```bash
# Option A — Colab file upload UI (small files)
#   Click the file icon in the left sidebar → Upload → select tornado-poc-jar.jar
# Option B — push to GCS / Drive / scp; for one-off, the UI is fastest.
```

The jar lands at `/content/tornado-poc-jar.jar` by default. Verify:

```bash
!ls -lh /content/tornado-poc-jar.jar
```

## Step 7 — Run the POC

This is the actual benchmark — emits the labelled key=value stdout:

```bash
!java -cp /content/tornado-poc-jar.jar \
    -Dtornado.device=nvidia:0:0 \
    -Dbench.env=colab \
    -Dbench.size=4096 \
    bench.tornadopoc.GemmBench
```

Expect 11 lines of key=value output:

```
cpu_baseline_ms=<number>
gpu_ms=<number>
transfer_ms=<number>
speedup_ratio=<number>
transfer_pct=<number>
device=<nvidia:0:0 or full device string>
result=OK  (or GPU_INIT_FAILED / DEVICE_MISMATCH)
verdict=GO  (or NO-GO / INSUFFICIENT_DATA)
env=colab
tornado.device=nvidia:0:0
jdk=<version>
hardware=Linux+amd64+<n>cores
size=4096
```

## Step 8 — Capture the results

1. Copy the stdout block.
2. Open `.planning/phases/05-hardware-abstraction-layer-gpu-poc/05-GPU-POC-RESULTS.md`
   in the repo.
3. Paste the captured stdout under the `## Colab (NVIDIA T4)` heading.
4. Verify all six required fields are present: `speedup_ratio`, `transfer_pct`,
   `device`, `jdk`, `hardware`, `environment`.
5. Add an explicit go/no-go recommendation at the bottom of the section:
   - `verdict=GO` from the runner → "GO recommendation: pursue GPU backend
     milestone in v0.4.0+".
   - `verdict=NO-GO` or `result=GPU_INIT_FAILED` → "NO-GO: GPU path does not
     beat the 2x speedup or 50% transfer budget on T4; defer GPU backend work
     until the next hardware refresh."

## What to do on failure

- **`GPU_INIT_FAILED`** — TornadoVM driver failed to load. Confirm `nvidia-smi`
  shows T4 and `TORNADO_SDK` is set; rerun Step 5.
- **`DEVICE_MISMATCH`** — Runner requested `nvidia:0:0` but runtime picked a
  different backend. List available devices with
  `!java -cp /content/tornado-poc-jar.jar -Ddebug.tornado=true ...` (or
  check `TornadoRuntimeProvider.getTornadoRuntime().getDrivers()` in a scratch
  cell). Adjust `-Dtornado.device` to match.
- **OOM on `-Dbench.size=8192`** — T4 has 16 GB VRAM but the JVM heap is shared
  with the host buffer. Drop back to 4096 or `-Xmx4g`.

## Why PATH A (local CPU-only) is also a valid run

Local dev box (i7-1255U) has no NVIDIA GPU. Running without `-Dtornado.device`
emits `result=GPU_ABSENT verdict=NO-GO` plus a real CPU baseline number.
This is the documented negative result for HW-03 — establishes that the local
box can't exercise the GPU path without external hardware. Either path
concludes the POC; Colab is preferred for the actual GO/NO-GO recommendation.

---

**Pin references** (per `VERSIONS.md` HIGH confidence):
- TornadoVM `5.2.0-jdk21`: https://github.com/beehive-lab/TornadoVM/releases
- Colab runtime T4: 16 GB VRAM, 2.5 GHz boost, 2560 CUDA cores.
- This recipe assumes the Colab image ships NVIDIA driver `>=525.x`.
