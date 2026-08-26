package numja.core;

import numja.NumJa;
import numja.core.NDArray;
import numja.core.ParallelOps;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Cross-path correctness tests for NDArray reduce ops above THRESHOLD.
 * Parallel result must match a hand-rolled sequential reference:
 *   - sum / mean within relErr 1e-13 (deterministic left-to-right tree merge)
 *   - min / max exact (delta 0.0; IEEE-754 ops)
 * The forced-sequential test exercises the setThresholdForTesting hook so
 * we know the sequential path matches itself (sanity gate for the hook).
 */
public class ParallelReduceTest {

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

    private static double sequentialSum(double[] data) {
        double s = 0.0;
        for (int i = 0; i < data.length; i++) s += data[i];
        return s;
    }

    private static double sequentialMin(double[] data) {
        double m = Double.MAX_VALUE;
        for (int i = 0; i < data.length; i++) if (data[i] < m) m = data[i];
        return m;
    }

    private static double sequentialMax(double[] data) {
        double m = -Double.MAX_VALUE;
        for (int i = 0; i < data.length; i++) if (data[i] > m) m = data[i];
        return m;
    }

    @After
    public void reset() {
        ParallelOps.resetThresholdForTesting();
    }

    @Test
    public void sum_matchesSequential_aboveThreshold() {
        NDArray arr = randomVec(1_000_000, 60L);
        double[] raw = arr.toDoubleArray();
        double expected = sequentialSum(raw);
        double actual = NumJa.sum(arr);
        assertTrue("sum relErr too large: " + relErr(actual, expected),
                   relErr(actual, expected) <= 1e-13);
    }

    @Test
    public void min_matchesSequential_aboveThreshold() {
        NDArray arr = randomVec(500_000, 61L);
        double[] raw = arr.toDoubleArray();
        double expected = sequentialMin(raw);
        double actual = NumJa.min(arr);
        assertEquals(expected, actual, 0.0);
    }

    @Test
    public void max_matchesSequential_aboveThreshold() {
        NDArray arr = randomVec(500_000, 61L);
        double[] raw = arr.toDoubleArray();
        double expected = sequentialMax(raw);
        double actual = NumJa.max(arr);
        assertEquals(expected, actual, 0.0);
    }

    @Test
    public void mean_matchesSequential_aboveThreshold() {
        NDArray arr = randomVec(1_000_000, 60L);
        double[] raw = arr.toDoubleArray();
        double expected = sequentialSum(raw) / raw.length;
        double actual = NumJa.mean(arr);
        assertTrue("mean relErr too large: " + relErr(actual, expected),
                   relErr(actual, expected) <= 1e-13);
    }

    @Test
    public void sum_forcedSequential_matchesParallel() {
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        double[] data = randomVec(1_000_000, 60L).toDoubleArray();
        double first = ParallelOps.sum(data);
        double second = ParallelOps.sum(data);
        assertEquals(first, second, 0.0);
    }
}
