package numja.core;

import numja.NumJa;
import numja.core.NDArray;
import numja.core.ParallelOps;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Cross-path correctness for elementwise ops above THRESHOLD (100_000).
 * Parallel results must match a forced-sequential reference within IEEE rounding.
 */
public class ParallelElementwiseTest {

    private static final int N = 1_000_000;

    private static NDArray randomVec(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[] a1 = new double[n];
        for (int i = 0; i < n; i++) a1[i] = rng.nextDouble();
        return new NDArray(a1);
    }

    private static double relErr(double actual, double expected) {
        double denom = Math.max(Math.abs(expected), 1e-30);
        return Math.abs(actual - expected) / denom;
    }

    @After
    public void reset() {
        ParallelOps.resetThresholdForTesting();
    }

    @Test
    public void add_matchesSequential_aboveThreshold() {
        NDArray a = randomVec(N, 50L);
        NDArray b = randomVec(N, 51L);

        // Reference: force sequential path.
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        NDArray expected = NumJa.add(a, b);

        // Parallel path.
        ParallelOps.resetThresholdForTesting();
        NDArray actual = NumJa.add(a, b);

        double[] e = expected.toDoubleArray();
        double[] ac = actual.toDoubleArray();
        for (int idx : new int[]{0, N / 2, N - 1}) {
            assertTrue("add[" + idx + "] relErr " + relErr(ac[idx], e[idx]),
                    relErr(ac[idx], e[idx]) <= 1e-13);
        }
    }

    @Test
    public void multiply_matchesSequential_aboveThreshold() {
        NDArray a = randomVec(N, 50L);
        NDArray b = randomVec(N, 51L);

        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        NDArray expected = NumJa.multiply(a, b);

        ParallelOps.resetThresholdForTesting();
        NDArray actual = NumJa.multiply(a, b);

        double[] e = expected.toDoubleArray();
        double[] ac = actual.toDoubleArray();
        for (int idx : new int[]{0, N / 2, N - 1}) {
            assertTrue("multiply[" + idx + "] relErr " + relErr(ac[idx], e[idx]),
                    relErr(ac[idx], e[idx]) <= 1e-13);
        }
    }

    @Test
    public void exp_matchesSequential_aboveThreshold() {
        NDArray a = randomVec(N, 50L);

        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        NDArray expected = a.exp();

        ParallelOps.resetThresholdForTesting();
        NDArray actual = a.exp();

        double[] e = expected.toDoubleArray();
        double[] ac = actual.toDoubleArray();
        for (int idx : new int[]{0, N / 2, N - 1}) {
            assertTrue("exp[" + idx + "] abs delta " + Math.abs(ac[idx] - e[idx]),
                    Math.abs(ac[idx] - e[idx]) <= 1e-12);
        }
    }

    @Test
    public void add_forcedSequential_matchesParallel() {
        java.util.Random rng = new java.util.Random(50L);
        double[] a = new double[N];
        double[] b = new double[N];
        for (int i = 0; i < N; i++) {
            a[i] = rng.nextDouble();
            b[i] = rng.nextDouble();
        }

        double[] out1 = new double[N];
        double[] out2 = new double[N];

        // Sequential branch via test threshold override.
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        ParallelOps.elementwiseBinary(a, b, out1, Double::sum);

        // Parallel branch via default threshold.
        ParallelOps.resetThresholdForTesting();
        ParallelOps.elementwiseBinary(a, b, out2, Double::sum);

        // Both paths must produce bit-identical results.
        for (int idx : new int[]{0, N / 2, N - 1}) {
            assertTrue("out1[" + idx + "] != out2[" + idx + "]",
                    relErr(out1[idx], out2[idx]) == 0.0);
        }
    }

    @Test
    public void scalarAdd_matchesSequential_aboveThreshold() {
        NDArray a = randomVec(N, 50L);

        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        NDArray expected = a.add(0.5);

        ParallelOps.resetThresholdForTesting();
        NDArray actual = a.add(0.5);

        double[] e = expected.toDoubleArray();
        double[] ac = actual.toDoubleArray();
        for (int idx : new int[]{0, N / 2, N - 1}) {
            assertTrue("scalarAdd[" + idx + "] delta " + Math.abs(ac[idx] - e[idx]),
                    Math.abs(ac[idx] - e[idx]) == 0.0);
        }
    }
}
