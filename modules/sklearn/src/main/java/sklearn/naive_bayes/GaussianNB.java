package sklearn.naive_bayes;

import numja.core.NDArray;

/**
 * Gaussian Naive Bayes Classifier
 * P(y|X) proportional to P(y) * product of P(x_i|y)
 *
 * <p>Streaming-capable: {@link #partial_fit(NDArray, int[])} accepts
 * successive chunks and {@link #finalize_fit()} divides the running
 * accumulators into means/variances. Cross-path equivalence:
 * {@code partial_fit(X1,y1).partial_fit(X2,y2).finalize_fit()} yields
 * predictions identical to {@code fit(X_full, y_full)}.
 *
 * <p>Trust boundary: caller-controlled NDArray (no I/O).
 * STRIDE-T402 (stale state across partial_fit calls) is mitigated by
 * the {@code initialized} flag — first call invokes {@code init}, subsequent
 * calls only invoke {@code accumulate}.
 */
public class GaussianNB {
    // Running accumulators (per-class, per-feature).
    private double[][] sum;     // running sum
    private double[][] M2;      // running sum of squared diffs from running mean
    private double[][] means;   // set in divide()
    private double[][] variances;
    private double[] classPriors;
    private int[] counts;
    private int n_classes;
    private boolean fitted = false;
    // WR-05: explicit identity init; first partial_fit flips to true.
    private boolean initialized = false;

    public GaussianNB() {}

    public GaussianNB fit(NDArray X, int[] y) {
        init(X, y);
        accumulate(X, y);
        divide();
        return this;
    }

    /**
     * Incrementally fit on a chunk. First call invokes {@code init}
     * (allocates accumulators from {@code y} cardinality and {@code X}
     * width); subsequent calls only update running accumulators.
     *
     * <p>Caller MUST invoke {@link #finalize_fit()} before {@link #predict}.
     *
     * @since ASVS-L1
     */
    public GaussianNB partial_fit(NDArray X, int[] y) {
        if (!initialized) init(X, y);
        accumulate(X, y);
        return this;
    }

    /**
     * Divide running accumulators into means/variances/classPriors
     * and flip {@code fitted} to true. Idempotent within an instance.
     *
     * <p>Named {@code finalize_fit} (not {@code finalize}) to avoid
     * overriding {@link Object#finalize()}.
     *
     * @since ASVS-L1
     */
    public GaussianNB finalize_fit() {
        divide();
        return this;
    }

    public int[] predict(NDArray X) {
        check(); double[][] data = extract2D(X);
        int[] pred = new int[data.length];
        for (int i = 0; i < data.length; i++) pred[i] = argmax(logProbs(data[i]));
        return pred;
    }

    public double[][] predictProba(NDArray X) {
        check(); double[][] data = extract2D(X);
        double[][] proba = new double[data.length][n_classes];
        for (int i = 0; i < data.length; i++) {
            double[] lp = logProbs(data[i]);
            double max = Double.NEGATIVE_INFINITY;
            for (double v : lp) if (v > max) max = v;
            double sum = 0;
            for (int c = 0; c < n_classes; c++) { proba[i][c] = Math.exp(lp[c] - max); sum += proba[i][c]; }
            for (int c = 0; c < n_classes; c++) proba[i][c] /= sum;
        }
        return proba;
    }

    public double score(NDArray X, int[] y) {
        int[] p = predict(X); int ok = 0;
        for (int i = 0; i < y.length; i++) if (p[i] == y[i]) ok++;
        return (double) ok / y.length;
    }

    /**
     * Allocate per-class running accumulators sized to n_classes (derived
     * from {@code y}) and feature count d (from {@code X}).
     */
    private void init(NDArray X, int[] y) {
        double[][] data = extract2D(X);
        int n = data.length, d = data[0].length;
        n_classes = 0;
        for (int l : y) if (l > n_classes) n_classes = l;
        n_classes++;

        sum = new double[n_classes][d];
        M2 = new double[n_classes][d];
        means = new double[n_classes][d];
        variances = new double[n_classes][d];
        classPriors = new double[n_classes];
        counts = new int[n_classes];
        initialized = true;
    }

    /**
     * Merge a new chunk into the running accumulators using Chan's
     * parallel-update formula. Per-chunk M2 is computed with the
     * chunk's column means (matching the original one-shot fit formula),
     * then merged into the running M2.
     */
    private void accumulate(NDArray X, int[] y) {
        double[][] data = extract2D(X);
        int n = data.length, d = data[0].length;

        int[] chunkCount = new int[n_classes];
        double[][] chunkSum = new double[n_classes][d];
        double[][] chunkM2 = new double[n_classes][d];
        double[][] chunkMean = new double[n_classes][d];

        // Pass 1: per-class sum + count for this chunk.
        for (int i = 0; i < n; i++) {
            int c = y[i];
            chunkCount[c]++;
            for (int j = 0; j < d; j++) chunkSum[c][j] += data[i][j];
        }

        // Per-class column mean for this chunk.
        for (int c = 0; c < n_classes; c++) {
            if (chunkCount[c] > 0) {
                double inv = 1.0 / chunkCount[c];
                for (int j = 0; j < d; j++) chunkMean[c][j] = chunkSum[c][j] * inv;
            }
        }

        // Pass 2: M2 for this chunk using the chunk's per-column mean.
        for (int i = 0; i < n; i++) {
            int c = y[i];
            for (int j = 0; j < d; j++) {
                double diff = data[i][j] - chunkMean[c][j];
                chunkM2[c][j] += diff * diff;
            }
        }

        // Merge chunk into running state using Chan's parallel formula.
        for (int c = 0; c < n_classes; c++) {
            int kc = chunkCount[c];
            if (kc == 0) continue;
            int oldCount = counts[c];
            int newCount = oldCount + kc;

            for (int j = 0; j < d; j++) {
                double runningSum = sum[c][j] + chunkSum[c][j];
                if (oldCount == 0) {
                    M2[c][j] = chunkM2[c][j];
                } else {
                    double oldMean = sum[c][j] / oldCount;
                    double delta = chunkMean[c][j] - oldMean;
                    M2[c][j] = M2[c][j] + chunkM2[c][j]
                        + delta * delta * oldCount * kc / newCount;
                }
                sum[c][j] = runningSum;
            }
            counts[c] = newCount;
        }
    }

    /**
     * Divide running accumulators into the public {@code means},
     * {@code variances}, and {@code classPriors} arrays; flip fitted=true.
     */
    private void divide() {
        int total = 0;
        for (int c = 0; c < n_classes; c++) total += counts[c];

        for (int c = 0; c < n_classes; c++) {
            int n = counts[c];
            if (n > 0) {
                classPriors[c] = (double) n / total;
                double inv = 1.0 / n;
                for (int j = 0; j < sum[c].length; j++) {
                    means[c][j] = sum[c][j] * inv;
                    variances[c][j] = M2[c][j] * inv + 1e-9;
                }
            }
        }
        fitted = true;
    }

    private double[] logProbs(double[] x) {
        double[] lp = new double[n_classes];
        for (int c = 0; c < n_classes; c++) {
            lp[c] = Math.log(classPriors[c]);
            for (int j = 0; j < x.length; j++) {
                double diff = x[j] - means[c][j];
                lp[c] -= 0.5 * (Math.log(2 * Math.PI * variances[c][j]) + diff * diff / variances[c][j]);
            }
        }
        return lp;
    }

    private int argmax(double[] a) { int idx = 0; for (int i = 1; i < a.length; i++) if (a[i] > a[idx]) idx = i; return idx; }
    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }

    private double[][] extract2D(NDArray arr) {
        int[] s = arr.getShape(); double[][] r = new double[s[0]][s[1]];
        for (int i = 0; i < s[0]; i++) for (int j = 0; j < s[1]; j++) r[i][j] = arr.get(i, j);
        return r;
    }
}
