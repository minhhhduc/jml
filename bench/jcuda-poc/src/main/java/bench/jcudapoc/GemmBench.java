package bench.jcudapoc;

import java.util.Arrays;
import java.util.Random;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import numja.core.ArrayOps;
import numja.core.NDArray;

/** Isolated CPU-versus-cuBLAS Dgemm proof of concept; production modules remain JCuda-free. */
public final class GemmBench {
    private static final int DEFAULT_N = 4096;
    private static final int MAX_N = 4096;

    private GemmBench() {}

    public static void main(String[] args) {
        final String env = System.getProperty("bench.env", "local");
        final Integer n = parseSize(env);
        if (n == null) return;

        final int cells = n * n;
        final double[] a = seededRandom(cells, 0xC0FFEEL);
        final double[] b = seededRandom(cells, 0xBADF00DL);
        final double[] cCpu = new double[cells];
        final double cpuBaselineMs = timeMedianMs(3, () -> {
            final NDArray result = ArrayOps.dot(new NDArray(reshapeFlat(a, n)), new NDArray(reshapeFlat(b, n)));
            System.arraycopy(result.getData().data, 0, cCpu, 0, cells);
        });

        final ProbeResult probe = probeDevice();
        if (probe.result() != null) {
            emitResult(cpuBaselineMs, "NA", "NA", "NA", "NA", probe.device(), probe.result(), "NO-GO", n, env, "NA");
            return;
        }

        // Task 2 adds the cuBLAS Dgemm execution path for the discovered device.
        emitResult(cpuBaselineMs, "NA", "NA", "NA", "NA", probe.device(), "GPU_INIT_FAILED", "NO-GO", n, env, "NA");
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

    private static double timeMedianMs(int iterations, Runnable action) {
        action.run();
        final long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            final long start = System.nanoTime();
            action.run();
            samples[i] = System.nanoTime() - start;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2] / 1_000_000.0;
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
}
