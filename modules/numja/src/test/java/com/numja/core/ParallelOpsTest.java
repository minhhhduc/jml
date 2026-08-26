package numja.core;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * Tests for {@link ParallelOps}: threshold gate, ForkJoin elementwise + reduce correctness.
 */
public class ParallelOpsTest {

    private static double[] randomVec(int n, long seed) {
        Random rng = new Random(seed);
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.nextDouble();
        return a;
    }

    private static double relErr(double actual, double expected) {
        double denom = Math.max(Math.abs(expected), 1e-30);
        return Math.abs(actual - expected) / denom;
    }

    @Test
    public void THRESHOLD_is100k() {
        assertEquals(100_000, ParallelOps.THRESHOLD);
    }

    @Test
    public void belowThreshold_usesSequential() {
        int n = ParallelOps.THRESHOLD - 1;
        double[] a = randomVec(n, 11L);
        double[] b = randomVec(n, 12L);
        double[] out = new double[n];
        ParallelOps.elementwiseBinary(a, b, out, Double::sum);

        for (int i = 0; i < n; i++) {
            assertEquals(a[i] + b[i], out[i], 1e-12);
        }
    }

    @Test
    public void elementwiseBinary_matchesSequential() {
        int n = 1_000_000;
        double[] a = randomVec(n, 21L);
        double[] b = randomVec(n, 22L);
        double[] out = new double[n];
        ParallelOps.elementwiseBinary(a, b, out, Double::sum);

        for (int idx : new int[]{0, n / 2, n - 1}) {
            assertEquals(a[idx] + b[idx], out[idx], 1e-9);
        }
    }

    @Test
    public void sum_matchesSequential() {
        int n = 1_000_000;
        double[] data = randomVec(n, 31L);

        double expected = 0.0;
        for (int i = 0; i < n; i++) expected += data[i];

        double actual = ParallelOps.sum(data);
        assertEquals("parallel sum must match sequential within relErr 1e-13", 0.0, relErr(actual, expected), 1e-13);
    }

    @Test
    public void min_matchesSequential() {
        int n = 500_000;
        double[] data = randomVec(n, 41L);
        assertEquals(sequentialMin(data), ParallelOps.min(data), 0.0);
    }

    @Test
    public void max_matchesSequential() {
        int n = 500_000;
        double[] data = randomVec(n, 51L);
        assertEquals(sequentialMax(data), ParallelOps.max(data), 0.0);
    }

    private static double sequentialMin(double[] data) {
        double m = Double.MAX_VALUE;
        for (double v : data) if (v < m) m = v;
        return m;
    }

    private static double sequentialMax(double[] data) {
        double m = -Double.MAX_VALUE;
        for (double v : data) if (v > m) m = v;
        return m;
    }
}
