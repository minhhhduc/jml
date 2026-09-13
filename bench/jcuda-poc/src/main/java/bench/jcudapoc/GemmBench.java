package bench.jcudapoc;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaGetDevice;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemGetInfo;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import numja.core.ArrayOps;
import numja.core.NDArray;

/** Isolated CPU-versus-cuBLAS Dgemm proof of concept; production modules remain JCuda-free. */
public final class GemmBench {
    private static final int DEFAULT_N = 4096;
    private static final int MAX_N = 4096;
    private static final double FROB_REL_ERR_LIMIT = 1e-9;

    private GemmBench() {}

    public static void main(String[] args) {
        final String env = System.getProperty("bench.env", "local");
        final Integer n = parseSize(env);
        if (n == null) return;

        final int cells = n * n;
        final double[] a = seededRandom(cells, 0xC0FFEEL);
        final double[] b = seededRandom(cells, 0xBADF00DL);
        final double[] cCpu = new double[cells];
        final Runnable cpuDot = () -> {
            final NDArray result = ArrayOps.dot(new NDArray(reshapeFlat(a, n)), new NDArray(reshapeFlat(b, n)));
            System.arraycopy(result.getData().data, 0, cCpu, 0, cells);
        };
        cpuDot.run(); // Untimed warmup.
        final TimingSamples cpu = timeSamplesMs(3, cpuDot);

        final ProbeResult probe = probeDevice();
        if (probe.result() != null) {
            emitSamples("cpu_sample", cpu.samplesMs());
            emitResult(cpu.medianMs(), "NA", "NA", "NA", "NA", probe.device(), probe.result(), "NO-GO", n, env, "NA");
            return;
        }

        try {
            final GpuResult gpu = runGpu(cpu.medianMs(), a, b, cCpu, n);
            emitSamples("cpu_sample", cpu.samplesMs());
            emitSamples("gpu_sample", gpu.samplesMs());
            emitDeviceMemory(gpu.memory());
            emit("cpu_vs_gpu_max_abs_err", format(gpu.errors().maxAbsErr()));
            emit("cpu_vs_gpu_max_rel_err", format(gpu.errors().maxRelErr()));
            emit("cpu_vs_gpu_mae", format(gpu.errors().mae()));
            emitResult(cpu.medianMs(), gpu.gpuMs(), gpu.transferMs(), gpu.speedupRatio(), gpu.transferPct(),
                    probe.device(), gpu.result(), gpu.verdict(), n, env, format(gpu.errors().frobRelErr()));
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            System.err.println("[GemmBench] GPU initialization failed: " + e.getClass().getName() + ": " + e.getMessage());
            emitSamples("cpu_sample", cpu.samplesMs());
            emitResult(cpu.medianMs(), "NA", "NA", "NA", "NA", probe.device(), "GPU_INIT_FAILED", "NO-GO", n, env, "NA");
        }
    }

    private static Integer parseSize(String env) {
        final String requested = System.getProperty("bench.size", String.valueOf(DEFAULT_N));
        final int n;
        try {
            n = Integer.parseInt(requested);
        } catch (NumberFormatException e) {
            System.err.println("[GemmBench] invalid bench.size: " + requested);
            emitResult("NA", "NA", "NA", "NA", "NA", "NA", "CONFIG_ERROR", "NO-GO", "NA", env, "NA");
            return null;
        }
        if (n <= 0 || n > MAX_N) {
            System.err.println("[GemmBench] bench.size must be in 1.." + MAX_N + ": " + n);
            emitResult("NA", "NA", "NA", "NA", "NA", "NA", "CONFIG_ERROR", "NO-GO", n, env, "NA");
            return null;
        }
        return n;
    }

    private static ProbeResult probeDevice() {
        try {
            JCuda.setExceptionsEnabled(true);
            final int[] count = new int[1];
            JCuda.cudaGetDeviceCount(count);
            if (count[0] < 1) return new ProbeResult("GPU_ABSENT", "GPU_ABSENT");
            final cudaDeviceProp properties = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(properties, 0);
            return new ProbeResult(properties.getName(), null);
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            System.err.println("[GemmBench] CUDA initialization failed: " + e.getClass().getName() + ": " + e.getMessage());
            return new ProbeResult("GPU_INIT_FAILED", "GPU_INIT_FAILED");
        }
    }

    private static GpuResult runGpu(double cpuBaselineMs, double[] a, double[] b, double[] cCpu, int n) {
        JCublas2.setExceptionsEnabled(true);
        final long bytes = (long) n * n * Sizeof.DOUBLE;
        final Pointer dA = new Pointer();
        final Pointer dB = new Pointer();
        final Pointer dC = new Pointer();
        boolean aAllocated = false;
        boolean bAllocated = false;
        boolean cAllocated = false;
        cublasHandle handle = null;
        final DeviceMemory memory;
        try {
            final long[] freeBefore = new long[1];
            final long[] total = new long[1];
            final int[] device = new int[1];
            cudaGetDevice(device);
            cudaMemGetInfo(freeBefore, total);
            cudaMalloc(dA, bytes);
            aAllocated = true;
            cudaMalloc(dB, bytes);
            bAllocated = true;
            cudaMalloc(dC, bytes);
            cAllocated = true;
            final long[] freeAfter = new long[1];
            cudaMemGetInfo(freeAfter, total);
            memory = new DeviceMemory(
                    device[0],
                    mib(total[0]),
                    mib(freeBefore[0]),
                    mib(freeAfter[0]),
                    (double) bytes * 3 / (1024.0 * 1024.0));

            handle = new cublasHandle();
            JCublas2.cublasCreate(handle);
            final cublasHandle activeHandle = handle;
            final Pointer alpha = Pointer.to(new double[] {1.0});
            final Pointer beta = Pointer.to(new double[] {0.0});

            final long h2dStart = System.nanoTime();
            cudaMemcpy(dA, Pointer.to(a), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dB, Pointer.to(b), bytes, cudaMemcpyHostToDevice);
            final long h2dNs = System.nanoTime() - h2dStart;

            // Row-major A×B is column-major (A×B)^T = B^T×A^T, so cuBLAS receives B first.
            dgemm(activeHandle, alpha, dB, dA, beta, dC, n);
            cudaDeviceSynchronize(); // Untimed warmup initializes cuBLAS workspaces and kernels.

            final TimingSamples gpu = timeSamplesMs(3, () -> {
                dgemm(activeHandle, alpha, dB, dA, beta, dC, n);
                cudaDeviceSynchronize();
            });
            final double gpuMs = gpu.medianMs();

            final double[] cGpu = new double[n * n];
            final long d2hStart = System.nanoTime();
            cudaMemcpy(Pointer.to(cGpu), dC, bytes, cudaMemcpyDeviceToHost);
            final double transferMs = (h2dNs + System.nanoTime() - d2hStart) / 1_000_000.0;

            final ErrorMetrics errors = compare(cCpu, cGpu);
            if (errors.frobRelErr() > FROB_REL_ERR_LIMIT) {
                return new GpuResult(errors, gpu.samplesMs(), memory, "NA", transferMs, "NA", "NA", "NUMERIC_MISMATCH", "NO-GO");
            }
            if (memory.freeDecreaseMiB() < memory.expectedBytesMiB() * 0.5) {
                return new GpuResult(errors, gpu.samplesMs(), memory, "NA", transferMs, "NA", "NA", "NO_DEVICE_MEMORY_EVIDENCE", "NO-GO");
            }

            final double speedupRatio = cpuBaselineMs / gpuMs;
            final double transferPct = transferMs / gpuMs * 100.0;
            final String verdict = speedupRatio >= 2.0 && transferPct < 50.0 ? "GO" : "NO-GO";
            return new GpuResult(errors, gpu.samplesMs(), memory, gpuMs, transferMs, speedupRatio, transferPct, "OK", verdict);
        } finally {
            try {
                if (handle != null) JCublas2.cublasDestroy(handle);
            } finally {
                try {
                    if (aAllocated) cudaFree(dA);
                } finally {
                    try {
                        if (bAllocated) cudaFree(dB);
                    } finally {
                        if (cAllocated) cudaFree(dC);
                    }
                }
            }
        }
    }

    private static double mib(long bytes) { return bytes / (1024.0 * 1024.0); }

    private static void dgemm(cublasHandle handle, Pointer alpha, Pointer dB, Pointer dA, Pointer beta, Pointer dC, int n) {
        JCublas2.cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, dB, n, dA, n, beta, dC, n);
    }

    private static ErrorMetrics compare(double[] expected, double[] actual) {
        double maxAbs = 0.0, maxRel = 0.0, sumAbs = 0.0, sumSquared = 0.0, expectedSquared = 0.0;
        for (int i = 0; i < expected.length; i++) {
            final double difference = expected[i] - actual[i];
            final double absolute = Math.abs(difference);
            maxAbs = Math.max(maxAbs, absolute);
            maxRel = Math.max(maxRel, absolute / Math.max(Math.abs(expected[i]), 1e-12));
            sumAbs += absolute;
            sumSquared += difference * difference;
            expectedSquared += expected[i] * expected[i];
        }
        return new ErrorMetrics(maxAbs, maxRel, sumAbs / expected.length,
                Math.sqrt(sumSquared) / Math.max(Math.sqrt(expectedSquared), 1e-12));
    }

    private static double[][] reshapeFlat(double[] values, int n) {
        final double[][] rows = new double[n][n];
        for (int row = 0; row < n; row++) System.arraycopy(values, row * n, rows[row], 0, n);
        return rows;
    }

    private static double[] seededRandom(int length, long seed) {
        final Random random = new Random(seed);
        final double[] values = new double[length];
        for (int i = 0; i < length; i++) values[i] = random.nextDouble();
        return values;
    }

    private static TimingSamples timeSamplesMs(int iterations, Runnable action) {
        final double[] samplesMs = new double[iterations];
        for (int i = 0; i < iterations; i++) {
            final long start = System.nanoTime();
            action.run();
            samplesMs[i] = (System.nanoTime() - start) / 1_000_000.0;
        }
        return new TimingSamples(samplesMs, median(samplesMs));
    }

    private static double median(double[] samples) {
        final double[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static String format(double value) { return String.format(Locale.ROOT, "%.6e", value); }

    private static void emitDeviceMemory(DeviceMemory memory) {
        emit("cuda_device_index", String.valueOf(memory.device()));
        emit("cuda_total_mib", format(memory.totalMiB()));
        emit("cuda_free_mib_before_alloc", format(memory.freeMiBBefore()));
        emit("cuda_free_mib_after_alloc", format(memory.freeMiBAfter()));
        emit("cuda_alloc_delta_mib", format(memory.freeDecreaseMiB()));
        emit("cuda_alloc_expected_mib", format(memory.expectedBytesMiB()));
    }

    private static void emitSamples(String prefix, double[] samplesMs) {
        for (int i = 0; i < samplesMs.length; i++) emit(prefix + "_" + (i + 1) + "_ms", format(samplesMs[i]));
    }

    private static void emitResult(Object cpuBaselineMs, Object gpuMs, Object transferMs, Object speedupRatio, Object transferPct,
                                   String device, String result, String verdict, Object n, String env, String frobRelErr) {
        emit("cpu_baseline_ms", cpuBaselineMs.toString());
        emit("gpu_ms", gpuMs.toString());
        emit("transfer_ms", transferMs.toString());
        emit("speedup_ratio", speedupRatio.toString());
        emit("transfer_pct", transferPct.toString());
        emit("device", device);
        emit("result", result);
        emit("verdict", verdict);
        emit("env", env);
        emit("jdk", System.getProperty("java.version"));
        emit("hardware", System.getProperty("os.name") + "+" + System.getProperty("os.arch") + "+" + Runtime.getRuntime().availableProcessors() + "cores");
        emit("size", n.toString());
        emit("cpu_vs_gpu_frob_rel_err", frobRelErr);
    }

    private static void emit(String key, String value) { System.out.println(key + "=" + value); }

    private record ProbeResult(String device, String result) {}
    private record TimingSamples(double[] samplesMs, double medianMs) {}
    private record DeviceMemory(int device, double totalMiB, double freeMiBBefore, double freeMiBAfter, double expectedBytesMiB) {
        double freeDecreaseMiB() { return freeMiBBefore - freeMiBAfter; }
    }
    private record GpuResult(ErrorMetrics errors, double[] samplesMs, DeviceMemory memory, Object gpuMs, Object transferMs,
                             Object speedupRatio, Object transferPct, String result, String verdict) {}
    private record ErrorMetrics(double maxAbsErr, double maxRelErr, double mae, double frobRelErr) {}
}
