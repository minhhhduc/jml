package bench.tornadopoc;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;

import numja.core.ArrayOps;
import numja.core.NDArray;

import java.util.Arrays;

/**
 * Phase 5 POC: 4096^2 GEMM via numja-core (production CPU path) vs TornadoVM (GPU candidate).
 *
 * <p>Two labelled paths are emitted as {@code key=value} lines on stdout so the
 * decision doc ({@code docs/05-GPU-POC-RESULTS.md}) can be hand-pasted by the
 * runner. No auto-detect of a GPU: per HW-03, the device is opt-in via
 * {@code -Dtornado.device=<backend:device:idx>}. With the flag absent, the
 * runner emits {@code result=GPU_ABSENT verdict=NO-GO} and a CPU baseline —
 * the local dev box is i7-1255U (Intel, no NVIDIA GPU), so a local run is the
 * expected outcome.
 *
 * <p>The same shaded jar must run on either:
 * <ul>
 *   <li><b>PATH A</b> - local CPU-only. {@code result=GPU_ABSENT}, useful negative
 *       result (confirms local box can't exercise the GPU path without external hw).
 *   <li><b>PATH B</b> - Google Colab NVIDIA T4. {@code -Dtornado.device=nvidia:0:0}
 *       triggers the GPU path; speedup_ratio and transfer_pct are real numbers.
 *</ul>
 *
 * <p>System properties consumed:
 * <ul>
 *   <li>{@code -Dtornado.device=...} - REQUIRED for GPU run; absent = CPU-only.
 *   <li>{@code -Dbench.env=<local|colab>} - labels the output environment.
 *   <li>{@code -Dbench.size=4096} - N for the square GEMM (default 4096).
 *</ul>
 */
public final class GemmBench {

    /** Hard ceiling: HW-02 dispatcher guards 10^9 cells; 4096^2 = ~16.7M, well below. */
    private static final long MAX_DISPATCH_N = 1_000_000_000L;

    private GemmBench() {}

    public static void main(String[] args) {
        final String env = System.getProperty("bench.env", "local");
        final String requestedDevice = System.getProperty("tornado.device");
        final int n = Integer.parseInt(System.getProperty("bench.size", "4096"));

        // Pre-flight: HW-02 guard. 4096^2 = 16.7M cells, far below the ceiling.
        final long cells = (long) n * n;
        if (cells > MAX_DISPATCH_N) {
            System.out.println("result=CONFIG_ERROR");
            System.out.println("verdict=NO-GO");
            System.out.println("error_msg=size " + n + " exceeds MAX_DISPATCH_N=" + MAX_DISPATCH_N);
            System.out.flush();
            return;
        }

        // Fixed-seed input data so the CPU and GPU paths compute the same C.
        final double[] a = seededRandom(n * n, 0xC0FFEE_L);
        final double[] b = seededRandom(n * n, 0xBADF00D_L);
        final double[] cCpu = new double[n * n];

        // ---- CPU baseline (production path: ArrayOps.dot -> BackendSelector -> CpuThreadBackend.matmul)
        // NDArray(double[][]) constructor wraps a 2D double array; we reshape the flat
        // row-major buffer so the baseline matches the same logical input as the GPU path.
        final double[][] aRows = reshapeFlat(a, n, n);
        final double[][] bRows = reshapeFlat(b, n, n);
        final double cpuBaselineMs = timeMedianMs(3, () -> {
            final NDArray A = new NDArray(aRows);
            final NDArray B = new NDArray(bRows);
            final NDArray C = ArrayOps.dot(A, B);
            final double[] flat = C.getData().data;
            System.arraycopy(flat, 0, cCpu, 0, n * n);
        });
        emit("cpu_baseline_ms", String.valueOf(cpuBaselineMs));

        // ---- GPU path (only when -Dtornado.device is set)
        if (requestedDevice == null || requestedDevice.isBlank()) {
            emit("gpu_ms", "NA");
            emit("transfer_ms", "NA");
            emit("speedup_ratio", "NA");
            emit("transfer_pct", "NA");
            emit("result", "GPU_ABSENT");
            emit("verdict", "NO-GO");
            emitTail(env, requestedDevice, n);
            return;
        }

        // GPU allocation: DoubleArray is the TornadoVM raw-buffer type; copy from flat double[].
        final DoubleArray aTornado = DoubleArray.fromArray(a);
        final DoubleArray bTornado = DoubleArray.fromArray(b);
        final DoubleArray cTornado = new DoubleArray(n * n);

        long transferMs;
        long gpuMs;
        String result;
        String actualDevice;

        try {
            // HW-03 mitigation: assert the runtime picks a device matching the requested prefix.
            actualDevice = resolveActualDevice(requestedDevice);
            if (!actualDevice.toLowerCase().contains(requestedDevice.split(":")[0].toLowerCase())) {
                System.err.println("[GemmBench] device mismatch: requested=" + requestedDevice
                        + " actual=" + actualDevice + " - aborting (HW-03)");
                emit("device", actualDevice);
                emit("gpu_ms", "NA");
                emit("transfer_ms", "NA");
                emit("speedup_ratio", "NA");
                emit("transfer_pct", "NA");
                emit("result", "DEVICE_MISMATCH");
                emit("verdict", "NO-GO");
                emitTail(env, requestedDevice, n);
                return;
            }
            emit("device", actualDevice);

            // Build the TaskGraph: out[i,j] = sum_k a[i,k] * b[k,j]
            // Kernel takes KernelContext as first arg (raw-array API pattern); see
            // tornado-examples/.../kernelcontext/compute/MatrixMultiplication2DV1.java.
            final KernelContext context = new KernelContext();
            final TaskGraph taskGraph = new TaskGraph("s0")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, aTornado, bTornado)
                    .task("t0", GemmBench::gemmKernel, context, aTornado, bTornado, cTornado, n)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, cTornado);

            final WorkerGrid2D worker = new WorkerGrid2D(n, n);
            worker.setLocalWork(16, 16, 1);
            final GridScheduler grid = new GridScheduler("s0.t0", worker);

            final ImmutableTaskGraph snap = taskGraph.snapshot();
            final TornadoExecutionPlan plan = new TornadoExecutionPlan(snap)
                    .withGridScheduler(grid);

            // Cold transfer (FIRST_EXECUTION) - measure this for transfer_ms.
            final long t0 = System.nanoTime();
            plan.execute();
            final long t1 = System.nanoTime();
            transferMs = (t1 - t0) / 1_000_000L;

            // Timed iterations (warm cache on device; transfer amortised into kernel call).
            final int iters = 3;
            final long[] samples = new long[iters];
            for (int i = 0; i < iters; i++) {
                final long s0 = System.nanoTime();
                plan.execute();
                final long s1 = System.nanoTime();
                samples[i] = (s1 - s0) / 1_000_000L;
            }
            gpuMs = median(samples);

            // Numerical verification: full-matrix CPU-vs-GPU comparison.
            // For double-precision GEMM at N=4096 the theoretical error is O(N*eps) ~ 1e-12.
            // Threshold 1e-9 = ~1000x safety margin; catches any kernel bug (wrong index,
            // missing barrier, partial sum, off-by-one) that would otherwise hide behind
            // a single-cell check.
            double maxAbsErr = 0.0;
            double maxRelErr = 0.0;
            double sse = 0.0;     // sum of squared errors  (= ||A-B||_F^2)
            double mae = 0.0;     // mean absolute error    (= MAE for matrix equality)
            double frobCpu = 0.0; // ||cCpu||_F^2
            final double eps = 1e-12;
            for (int i = 0; i < n * n; i++) {
                final double cpu = cCpu[i];
                final double gpu = cTornado.get(i);
                final double diff = cpu - gpu;
                final double absDiff = Math.abs(diff);
                if (absDiff > maxAbsErr) maxAbsErr = absDiff;
                final double denom = Math.max(Math.abs(cpu), eps);
                final double rel = absDiff / denom;
                if (rel > maxRelErr) maxRelErr = rel;
                sse += diff * diff;
                mae += absDiff;
                frobCpu += cpu * cpu;
            }
            final double frobRelErr = Math.sqrt(sse) / Math.max(Math.sqrt(frobCpu), eps);
            mae /= (n * n);
            emit("cpu_vs_gpu_max_abs_err", String.format("%.3e", maxAbsErr));
            emit("cpu_vs_gpu_max_rel_err", String.format("%.3e", maxRelErr));
            emit("cpu_vs_gpu_frob_rel_err", String.format("%.3e", frobRelErr));
            emit("cpu_vs_gpu_mae", String.format("%.3e", mae));
            if (frobRelErr > 1e-9) {
                System.err.println("[GemmBench] numeric mismatch frob_rel_err=" + frobRelErr
                        + " > 1e-9 - aborting (kernel produced wrong output)");
                result = "NUMERIC_MISMATCH";
                gpuMs = -1;
            } else {
                result = "OK";
            }
        } catch (UnsatisfiedLinkError | TornadoRuntimeException e) {
            // No GPU / driver missing / native init failed - label and exit 0.
            System.err.println("[GemmBench] GPU init failed: " + e.getClass().getSimpleName()
                    + " - " + e.getMessage());
            emit("gpu_ms", "NA");
            emit("transfer_ms", "NA");
            emit("speedup_ratio", "NA");
            emit("transfer_pct", "NA");
            emit("result", "GPU_INIT_FAILED");
            emit("verdict", "NO-GO");
            emitTail(env, requestedDevice, n);
            return;
        }

        if ("OK".equals(result)) {
            emit("gpu_ms", String.valueOf(gpuMs));
            emit("transfer_ms", String.valueOf(transferMs));
            final double speedup = cpuBaselineMs / (double) gpuMs;
            final double transferPct = (double) transferMs / (double) gpuMs * 100.0;
            emit("speedup_ratio", String.format("%.3f", speedup));
            emit("transfer_pct", String.format("%.2f", transferPct));
            emit("result", result);
            final String verdict = (speedup >= 2.0 && transferPct < 50.0) ? "GO" : "NO-GO";
            emit("verdict", verdict);
        } else {
            emit("gpu_ms", "NA");
            emit("transfer_ms", "NA");
            emit("speedup_ratio", "NA");
            emit("transfer_pct", "NA");
            emit("result", result);
            emit("verdict", "NO-GO");
        }
        emitTail(env, requestedDevice, n);
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    /** Reshape a row-major flat {@code double[]} into a {@code double[][]} for NDArray. */
    private static double[][] reshapeFlat(double[] flat, int rows, int cols) {
        final double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, out[i], 0, cols);
        }
        return out;
    }

    /**
     * HW-03 mitigation: query the actual selected device and assert it matches the
     * requested prefix. Returns the human-readable device string the runtime picked,
     * or {@code "GPU_ABSENT"} if no backend is loaded.
     */
    private static String resolveActualDevice(String requestedDevice) {
        try {
            final TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();
            final String defaultDevice = runtime.getDefaultDevice().getDeviceName();
            return defaultDevice;
        } catch (Throwable t) {
            // Driver layer not loaded (typical on local box). Signal absence explicitly.
            return "GPU_ABSENT";
        }
    }

    /**
     * TornadoVM kernel - executed on the device. Reads from the device-side
     * copies of {@code a}, {@code b}; writes into {@code c}. Row-major flat layout.
     * Follows the kernelcontext MatrixMultiplication2D example pattern.
     * ponytail: simple naive GEMM - sufficient for POC speedup measurement;
     * production swap-in would use Tile/Vec operators.
     */
    public static void gemmKernel(KernelContext context, DoubleArray a, DoubleArray b, DoubleArray c, int n) {
        final int row = context.globalIdx;
        final int col = context.globalIdy;
        if (row >= n || col >= n) return;
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += a.get(row * n + k) * b.get(k * n + col);
        }
        c.set(row * n + col, sum);
    }

    /** Deterministic PRNG - same seed produces the same array across runs. */
    private static double[] seededRandom(int size, long seed) {
        final java.util.Random rng = new java.util.Random(seed);
        final double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = rng.nextDouble();
        }
        return out;
    }

    /** Time a Runnable across {@code iters} runs; return median in milliseconds. */
    private static double timeMedianMs(int iters, Runnable r) {
        // Warm-up
        r.run();
        final long[] samples = new long[iters];
        for (int i = 0; i < iters; i++) {
            final long s0 = System.nanoTime();
            r.run();
            final long s1 = System.nanoTime();
            samples[i] = (s1 - s0) / 1_000_000L;
        }
        return median(samples);
    }

    private static long median(long[] samples) {
        final long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static void emit(String k, String v) {
        System.out.println(k + "=" + v);
    }

    private static void emitTail(String env, String requestedDevice, int n) {
        emit("env", env);
        emit("tornado.device", requestedDevice == null ? "unset" : requestedDevice);
        emit("jdk", System.getProperty("java.version"));
        emit("hardware", System.getProperty("os.name") + "+" + System.getProperty("os.arch")
                + "+" + Runtime.getRuntime().availableProcessors() + "cores");
        emit("size", String.valueOf(n));
        // Default device for the CPU baseline - produced in NDArray -> CpuThreadBackend path.
        if (requestedDevice == null || requestedDevice.isBlank()) {
            System.out.println("device=cpu-thread");
        }
    }
}
