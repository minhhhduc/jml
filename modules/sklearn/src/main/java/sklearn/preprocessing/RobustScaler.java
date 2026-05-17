package sklearn.preprocessing;

import numja.core.NDArray;
import numja.NumJa;
import java.util.Arrays;

/**
 * RobustScaler - Scale features using statistics that are robust to outliers
 * Uses median and IQR (interquartile range) instead of mean and std
 * X_scaled = (X - median) / IQR
 */
public class RobustScaler {
    private double[] median;
    private double[] iqr;
    private boolean fitted = false;

    public RobustScaler() {}

    public RobustScaler fit(NDArray X) {
        int[] s = X.getShape(); int n = s[0], d = s[1];
        median = new double[d]; iqr = new double[d];

        for (int j = 0; j < d; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = X.get(i, j);
            Arrays.sort(col);
            median[j] = percentile(col, 0.5);
            double q1 = percentile(col, 0.25);
            double q3 = percentile(col, 0.75);
            iqr[j] = q3 - q1;
            if (iqr[j] < 1e-10) iqr[j] = 1.0;
        }
        fitted = true;
        return this;
    }

    public NDArray transform(NDArray X) {
        check(); int[] s = X.getShape(); int n = s[0], d = s[1];
        double[][] r = new double[n][d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) r[i][j] = (X.get(i, j) - median[j]) / iqr[j];
        return NumJa.array(r);
    }

    public NDArray fitTransform(NDArray X) { return fit(X).transform(X); }

    public NDArray inverseTransform(NDArray X) {
        check(); int[] s = X.getShape(); int n = s[0], d = s[1];
        double[][] r = new double[n][d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) r[i][j] = X.get(i, j) * iqr[j] + median[j];
        return NumJa.array(r);
    }

    private double percentile(double[] sorted, double p) {
        double idx = p * (sorted.length - 1);
        int lo = (int) Math.floor(idx), hi = (int) Math.ceil(idx);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
    }

    public double[] getMedian() { return median.clone(); }
    public double[] getIQR() { return iqr.clone(); }
    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }
}
