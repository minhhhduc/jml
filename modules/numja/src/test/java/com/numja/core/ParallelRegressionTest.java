package numja.core;

import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Timing-based regression gates for Phase 2 (CPU Parallel Core).
 *
 * Why this exists (vs just running JMH): JMH gives macro numbers across forks but is slow
 * and lives outside the test loop. This class asserts the structural property in-process:
 * "the parallel path doesn't silently regress back to raw-loop parity AND the threshold
 * gate keeps small arrays on the sequential branch."
 *
 * Methodology notes (02-RESEARCH.md Pitfall 3 — hybrid P/E-core variance):
 *  - Hybrid CPU scheduling causes single-run timing to be unreliable.
 *  - Each measurement takes the MEDIAN of 51 timed iterations after 10 warmup iterations
 *    plus 200 extra JIT warmup calls (amortises ForkJoinPool worker thread spin-up).
 *  - Direct {@link ParallelOps} calls (NOT {@code NumJa.add}) so the timed work is the
 *    elementwise/reduce body itself with no NDArray allocation overhead.
 *
 * What each test asserts:
 *  - largeArray_parallelDoesNotRegressMoreThan30Percent: at n = 10^7 the FJP path must complete
 *    in <= 130% of the raw-loop time. Observed ratio on this i7-1255U is ~0.95-1.10 (parallel
 *    slightly faster or comparable). The user-visible Phase 2 EJML-SIMD-vs-FJP speedup (~2-4x
 *    vs Phase 1 baseline) is verified by JMH in {@code 02-BASELINE-AFTER.md}. This in-process
 *    gate catches regressions where the parallel path silently degrades back to raw-loop
 *    parity or worse, while tolerating hybrid P/E-core variance.
 *  - smallArray_at10k_thresholdGatePreserved: at n = 10_000 both override and default take the
 *    raw-loop branch (n < THRESHOLD = 100_000). A threshold regression that drops THRESHOLD
 *    below 10_000 would force FJP at n=10k and inflate this ratio. Bound 1.50 catches that.
 *  - smallArray_at100k_thresholdBoundaryStable: at n = 100_000 == THRESHOLD the gate is
 *    `n < THRESHOLD`, so default takes FJP. FJP-vs-raw overhead at this boundary on this
 *    machine is ~1.9-2.2x (ForkJoin worker spin-up is expensive for ~100us work). Bound 3.0
 *    catches catastrophic regressions only.
 *  - reduceLarge_parallelDoesNotRegressMoreThan30Percent: at n = 10^7 FJP sum must complete in
 *    <= 130% of raw sum. Observed ratio ~0.55-0.75 (FJP faster). User-visible EJML-vs-FJP
 *    ratio verified by JMH.
 *
 * Seeds (47L, 48L) match {@code CoreBench.SmallArrayState} for apples-to-apples timing
 * comparison with the JMH baseline.
 */
public class ParallelRegressionTest {

    private static final int MEASURE_ITERS = 51;
    private static final int WARMUP_ITERS = 10;
    private static final int JIT_WARMUP_CALLS = 200;

    private static double[] randomArray(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.nextDouble();
        return a;
    }

    private static long medianNanos(long[] xs) {
        java.util.Arrays.sort(xs);
        int n = xs.length;
        if (n % 2 == 1) return xs[n / 2];
        return (xs[n / 2 - 1] + xs[n / 2]) / 2;
    }

    private static long timeOp(Runnable op) {
        for (int i = 0; i < WARMUP_ITERS; i++) op.run();
        long[] samples = new long[MEASURE_ITERS];
        for (int i = 0; i < MEASURE_ITERS; i++) {
            long t0 = System.nanoTime();
            op.run();
            samples[i] = System.nanoTime() - t0;
        }
        return medianNanos(samples);
    }

    private static Runnable addOp(double[] a, double[] b, double[] out) {
        return () -> ParallelOps.elementwiseBinary(a, b, out, Double::sum);
    }

    private static Runnable sumOp(double[] a) {
        return () -> { double s = ParallelOps.sum(a); if (Double.isNaN(s)) throw new AssertionError(); };
    }

    @After
    public void reset() {
        ParallelOps.resetThresholdForTesting();
    }

    @Test
    public void largeArray_parallelDoesNotRegressMoreThan30Percent() {
        final int n = 10_000_000;
        double[] a = randomArray(n, 47L);
        double[] b = randomArray(n, 48L);
        double[] out = new double[n];

        for (int i = 0; i < JIT_WARMUP_CALLS; i++) ParallelOps.elementwiseBinary(a, b, out, Double::sum);

        // Sequential baseline: override forces raw-loop branch inside elementwiseBinary.
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        long sequentialNanos = timeOp(addOp(a, b, out));

        // Parallel path: default threshold = 100_000, n = 10^7 -> FJP RecursiveAction.
        ParallelOps.resetThresholdForTesting();
        long parallelNanos = timeOp(addOp(a, b, out));

        double ratio = (double) parallelNanos / (double) sequentialNanos;
        // Conservative lower bound — parallel must not regress catastrophically vs raw seq.
        // See class javadoc for tuning rationale.
        assertTrue(
            "elementwiseBinary n=" + n + " ratio=" + ratio
                + " (parallel " + (parallelNanos / 1_000_000.0) + " ms"
                + " vs sequential " + (sequentialNanos / 1_000_000.0) + " ms)"
                + " must be <= 1.30",
            ratio <= 1.30);
    }

    @Test
    public void smallArray_at10k_thresholdGatePreserved() {
        final int n = 10_000;
        double[] a = randomArray(n, 47L);
        double[] b = randomArray(n, 48L);
        double[] out = new double[n];

        for (int i = 0; i < JIT_WARMUP_CALLS; i++) ParallelOps.elementwiseBinary(a, b, out, Double::sum);

        // Both paths take raw-loop branch (n < THRESHOLD = 100_000). Threshold regression below
        // 10_000 would force FJP here and inflate ratio.
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        long sequentialNanos = timeOp(addOp(a, b, out));

        ParallelOps.resetThresholdForTesting();
        long parallelNanos = timeOp(addOp(a, b, out));

        double ratio = (double) parallelNanos / (double) sequentialNanos;
        assertTrue(
            "elementwiseBinary n=" + n + " ratio=" + ratio
                + " (parallel " + (parallelNanos / 1_000.0) + " us"
                + " vs sequential " + (sequentialNanos / 1_000.0) + " us)"
                + " must be <= 1.50",
            ratio <= 1.50);
    }

    @Test
    public void smallArray_at100k_thresholdBoundaryStable() {
        final int n = 100_000;
        double[] a = randomArray(n, 47L);
        double[] b = randomArray(n, 48L);
        double[] out = new double[n];

        for (int i = 0; i < JIT_WARMUP_CALLS; i++) ParallelOps.elementwiseBinary(a, b, out, Double::sum);

        // Override forces raw-loop; default takes FJP (gate is `n < THRESHOLD`, 100_000 is NOT
        // < 100_000). FJP-vs-raw overhead at boundary on this machine ~1.9-2.2x.
        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        long sequentialNanos = timeOp(addOp(a, b, out));

        ParallelOps.resetThresholdForTesting();
        long parallelNanos = timeOp(addOp(a, b, out));

        double ratio = (double) parallelNanos / (double) sequentialNanos;
        assertTrue(
            "elementwiseBinary n=" + n + " ratio=" + ratio
                + " (parallel " + (parallelNanos / 1_000_000.0) + " ms"
                + " vs sequential " + (sequentialNanos / 1_000_000.0) + " ms)"
                + " must be <= 3.00 (FJP boundary overhead at n=100k on this machine ~2x)",
            ratio <= 3.00);
    }

    @Test
    public void reduceLarge_parallelDoesNotRegressMoreThan30Percent() {
        final int n = 10_000_000;
        double[] a = randomArray(n, 47L);

        for (int i = 0; i < JIT_WARMUP_CALLS; i++) {
            double s = ParallelOps.sum(a);
            if (Double.isNaN(s)) throw new AssertionError();
        }

        ParallelOps.setThresholdForTesting(Integer.MAX_VALUE);
        long sequentialNanos = timeOp(sumOp(a));

        ParallelOps.resetThresholdForTesting();
        long parallelNanos = timeOp(sumOp(a));

        double ratio = (double) parallelNanos / (double) sequentialNanos;
        assertTrue(
            "sum n=" + n + " ratio=" + ratio
                + " (parallel " + (parallelNanos / 1_000_000.0) + " ms"
                + " vs sequential " + (sequentialNanos / 1_000_000.0) + " ms)"
                + " must be <= 1.30",
            ratio <= 1.30);
    }
}
