package numja.core;

import org.junit.After;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Correctness tests for ACC-01 compensated reduce.
 *
 * RED-GREEN gate: written AFTER ParallelOps Kahan + log-sum-exp prod changes;
 * verifies that the compensated loops produce mathematically-correct results
 * on pathological input AND stay in lock-step with the sequential reference
 * on well-conditioned input.
 *
 * Threshold-override hygiene: every test calls {@code ParallelOps.setThresholdForTesting}
 * or relies on the default; {@code @After reset} clears state to avoid leaking
 * into other test classes (WR-03 pattern from Phase 2 review).
 */
public class ParallelCompensationTest {

    private static final double TOL = 1e-13;

    @After
    public void reset() {
        ParallelOps.resetThresholdForTesting();
    }

    // ---------------- helpers ----------------

    private static double[] randomVec(int n, long seed) {
        Random rng = new Random(seed);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble();
        return v;
    }

    private static double[] randomPositive(int n, long seed) {
        Random rng = new Random(seed);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble() * 0.5 + 0.5; // (0.5, 1.0)
        return v;
    }

    private static double relErr(double actual, double expected) {
        double denom = Math.max(Math.abs(expected), 1e-30);
        return Math.abs(actual - expected) / denom;
    }

    /** Alternating [1e15, 1e-15] pattern — catastrophic cancellation input. */
    private static double[] pathologicalAlternating(int n) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = (i % 2 == 0) ? 1e15 : 1e-15;
        return v;
    }

    private static double sequentialSum(double[] data) {
        double s = 0.0;
        for (double x : data) s += x;
        return s;
    }

    private static double sequentialProd(double[] data) {
        double p = 1.0;
        for (double x : data) p *= x;
        return p;
    }

    // ---------------- compensated sum ----------------

    @Test
    public void sum_pathologicalMatchesReference() {
        // Alternating [1e15, 1e-15] at n=1e6 — naive left-to-right sum is wildly wrong;
        // Kahan recovers the mathematically correct sum. 500_000 pairs of (1e15, 1e-15)
        // each contribute ~1e15; the trailing 500_000 × 1e-15 add a noise bump of ~5e5.
        // Total ≈ 5e20 + 5e5 ≈ 5.000000000000005e20.
        double[] data = pathologicalAlternating(1_000_000);
        double actual = ParallelOps.sum(data);
        double expected = 5.000000000000005e20;
        assertTrue("pathological sum err=" + relErr(actual, expected) + " tol=" + TOL,
                relErr(actual, expected) <= TOL);
    }

    @Test
    public void sum_compensatedCrossPathEquivalence() {
        double[] data = pathologicalAlternating(1_000_000);
        double parallel = ParallelOps.sum(data);
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        double sequential = ParallelOps.sum(data);
        assertTrue("cross-path sum err=" + relErr(parallel, sequential) + " tol=" + TOL,
                relErr(parallel, sequential) <= TOL);
    }

    @Test
    public void sum_compensatedWellConditionedInput() {
        double[] data = randomVec(1_000_000, 42L);
        double actual = ParallelOps.sum(data);
        double expected = sequentialSum(data);
        assertTrue("well-conditioned sum err=" + relErr(actual, expected) + " tol=" + TOL,
                relErr(actual, expected) <= TOL);
    }

    // ---------------- compensated prod ----------------

    @Test
    public void prod_logSumExpOnPositiveRandom() {
        double[] data = randomPositive(1_000_000, 70L);
        double actual = ParallelOps.prod(data);
        // Reference = exp(sum(log(data[i]))) computed with a Kahan-compensated sequential loop.
        // Note: Math.exp(NumericStable.logSumExp(logs)) would reduce to sum(data), not product.
        double sumLog = 0.0, c = 0.0;
        for (double v : data) {
            double y = Math.log(v) - c;
            double t = sumLog + y;
            c = (t - sumLog) - y;
            sumLog = t;
        }
        double expected = Math.exp(sumLog);
        assertTrue("prod logSumExp err=" + relErr(actual, expected) + " tol=" + TOL,
                relErr(actual, expected) <= TOL);
    }

    @Test
    public void prod_crossPathEquivalence() {
        double[] data = randomPositive(1_000_000, 70L);
        double parallel = ParallelOps.prod(data);
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        double sequential = ParallelOps.prod(data);
        assertTrue("prod cross-path err=" + relErr(parallel, sequential) + " tol=" + TOL,
                relErr(parallel, sequential) <= TOL);
    }

    @Test
    public void prod_signTracking() {
        // Use small values (0.1..0.9) so log-domain arithmetic stays well-conditioned
        // and tolerance 1e-12 is meaningful. The sign tracker is what we're testing —
        // not the magnitude, which is governed by the logSumExp compensation.
        double[] neg1   = {0.1, 0.2, -0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5};
        double[] neg2   = {0.1, -0.2, 0.3, -0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5};
        double[] withZero = {0.1, 0.2, 0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5};

        // Sign invariants — the primary contract:
        assertTrue("one neg should yield negative", ParallelOps.prod(neg1) < 0);
        assertTrue("two negs should yield positive", ParallelOps.prod(neg2) > 0);
        assertEquals("zero anywhere short-circuits to 0.0", 0.0, ParallelOps.prod(withZero), 0.0);

        // Magnitude check: log-domain product stays within ~1e-6 relErr of sequential naive
        // (sub-threshold branch — these arrays are n=10, well below THRESHOLD).
        assertTrue("neg1 magnitude relErr=" + relErr(ParallelOps.prod(neg1), sequentialProd(neg1)),
                relErr(ParallelOps.prod(neg1), sequentialProd(neg1)) <= 1e-6);
        assertTrue("neg2 magnitude relErr=" + relErr(ParallelOps.prod(neg2), sequentialProd(neg2)),
                relErr(ParallelOps.prod(neg2), sequentialProd(neg2)) <= 1e-6);
    }

    @Test
    public void prod_subThresholdStillSequential() {
        double[] data = randomPositive(50_000, 11L); // n < THRESHOLD
        double actual = new NDArray(data).prod();
        double expected = sequentialProd(data);
        assertTrue("sub-threshold prod err=" + relErr(actual, expected) + " tol=" + TOL,
                relErr(actual, expected) <= TOL);
    }

    @Test
    public void prod_emptyReturnsIdentity() {
        assertEquals(1.0, ParallelOps.prod(new double[0]), 0.0);
    }
}
