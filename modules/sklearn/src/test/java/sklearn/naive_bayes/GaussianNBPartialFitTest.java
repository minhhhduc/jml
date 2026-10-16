package sklearn.naive_bayes;

import numja.core.NDArray;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Cross-path equivalence tests for GaussianNB.partial_fit + finalize
 * versus one-shot fit().
 *
 * <p>Trust boundary: caller-controlled NDArray + int[] labels (no I/O).
 * STRIDE-T402 stale-state defense verified by init_idempotentAfterFirstCall</p>
 */
public class GaussianNBPartialFitTest {

    private static final long SEED = 20260828L;
    private static final int N = 1000;
    private static final int D = 4;
    private static final int N_CLASSES = 3;

    /** Build a deterministic synthetic dataset with n_classes balanced. */
    private static double[][] makeX(int n, int d, long seed) {
        double[][] x = new double[n][d];
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) x[i][j] = rng.nextGaussian();
        }
        return x;
    }

    private static int[] makeY(int n, int nClasses, long seed) {
        Random rng = new Random(seed);
        int[] y = new int[n];
        for (int i = 0; i < n; i++) y[i] = rng.nextInt(nClasses);
        return y;
    }

    private static NDArray toNdArray(double[][] data) {
        return new NDArray(data);
    }

    private static double[][] slice(double[][] data, int from, int to) {
        double[][] out = new double[to - from][];
        for (int k = 0; k < out.length; k++) out[k] = data[from + k];
        return out;
    }

    private static int[] slice(int[] data, int from, int to) {
        return Arrays.copyOfRange(data, from, to);
    }

    @Test
    public void twoPartialFits_equalOneFit() {
        double[][] Xfull = makeX(N, D, SEED);
        int[] yFull = makeY(N, N_CLASSES, SEED + 1);

        double[][] X1 = slice(Xfull, 0, N / 2);
        double[][] X2 = slice(Xfull, N / 2, N);
        int[] y1 = slice(yFull, 0, N / 2);
        int[] y2 = slice(yFull, N / 2, N);

        GaussianNB a = new GaussianNB();
        a.partial_fit(toNdArray(X1), y1).partial_fit(toNdArray(X2), y2).finalize_fit();

        GaussianNB b = new GaussianNB();
        b.fit(toNdArray(Xfull), yFull);

        int[] pa = a.predict(toNdArray(Xfull));
        int[] pb = b.predict(toNdArray(Xfull));

        assertArrayEquals("Categorical predictions must match", pb, pa);
        assertEquals("Score must match within 1e-9",
            b.score(toNdArray(Xfull), yFull),
            a.score(toNdArray(Xfull), yFull),
            1e-9);
    }

    @Test
    public void singlePartialFit_equalsFit() {
        double[][] X = makeX(N, D, SEED + 2);
        int[] y = makeY(N, N_CLASSES, SEED + 3);

        GaussianNB a = new GaussianNB();
        a.partial_fit(toNdArray(X), y).finalize_fit();

        GaussianNB b = new GaussianNB();
        b.fit(toNdArray(X), y);

        int[] pa = a.predict(toNdArray(X));
        int[] pb = b.predict(toNdArray(X));
        assertArrayEquals("First-call partial_fit must equal fit", pb, pa);
    }

    @Test
    public void init_resetsState() {
        // Two fresh instances, identical sequence -> identical predictions.
        double[][] X = makeX(N, D, SEED + 4);
        int[] y = makeY(N, N_CLASSES, SEED + 5);

        double[][] X1 = slice(X, 0, N / 2);
        double[][] X2 = slice(X, N / 2, N);
        int[] y1 = slice(y, 0, N / 2);
        int[] y2 = slice(y, N / 2, N);

        GaussianNB a = new GaussianNB();
        a.partial_fit(toNdArray(X1), y1).partial_fit(toNdArray(X2), y2).finalize_fit();

        GaussianNB b = new GaussianNB();
        b.partial_fit(toNdArray(X1), y1).partial_fit(toNdArray(X2), y2).finalize_fit();

        assertArrayEquals(
            "Identical sequences on fresh instances must match",
            a.predict(toNdArray(X)),
            b.predict(toNdArray(X)));
    }

    @Test
    public void init_idempotentAfterFirstCall() {
        // Three consecutive partial_fit calls with the same data must equal one-shot fit.
        double[][] X = makeX(N, D, SEED + 6);
        int[] y = makeY(N, N_CLASSES, SEED + 7);

        GaussianNB a = new GaussianNB();
        a.partial_fit(toNdArray(X), y)
         .partial_fit(toNdArray(X), y)
         .partial_fit(toNdArray(X), y)
         .finalize_fit();

        GaussianNB b = new GaussianNB();
        b.fit(toNdArray(X), y);

        // Same data partial-fitted 3x vs single fit — predictions must match.
        assertArrayEquals(
            "Same-data triple partial_fit must equal single fit",
            a.predict(toNdArray(X)),
            b.predict(toNdArray(X)));
    }

    @Test
    public void predictBeforeFinalize_throws() {
        double[][] X = makeX(N, D, SEED + 8);
        int[] y = makeY(N, N_CLASSES, SEED + 9);

        GaussianNB a = new GaussianNB();
        a.partial_fit(toNdArray(X), y);

        try {
            a.predict(toNdArray(X));
            fail("predict before finalize must throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(
                "Error message must contain 'Not fitted', got: " + e.getMessage(),
                e.getMessage().contains("Not fitted"));
        }
    }
}
