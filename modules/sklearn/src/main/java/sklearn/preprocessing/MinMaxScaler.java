package sklearn.preprocessing;

import numja.core.NDArray;
import numja.NumJa;

/**
 * MinMaxScaler - Scale features to a given range [0, 1]
 * X_scaled = (X - X_min) / (X_max - X_min)
 */
public class MinMaxScaler {
    private double[] min;
    private double[] max;
    private boolean fitted = false;

    public MinMaxScaler() {}

    public MinMaxScaler fit(NDArray X) {
        int[] s = X.getShape(); int n = s[0], d = s[1];
        min = new double[d]; max = new double[d];
        for (int j = 0; j < d; j++) { min[j] = Double.MAX_VALUE; max[j] = -Double.MAX_VALUE; }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double v = X.get(i, j);
                if (v < min[j]) min[j] = v;
                if (v > max[j]) max[j] = v;
            }
        }
        fitted = true;
        return this;
    }

    public NDArray transform(NDArray X) {
        check(); int[] s = X.getShape(); int n = s[0], d = s[1];
        double[][] r = new double[n][d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) {
                double range = max[j] - min[j];
                r[i][j] = range > 1e-10 ? (X.get(i, j) - min[j]) / range : 0.0;
            }
        return NumJa.array(r);
    }

    public NDArray fitTransform(NDArray X) { return fit(X).transform(X); }

    public NDArray inverseTransform(NDArray X) {
        check(); int[] s = X.getShape(); int n = s[0], d = s[1];
        double[][] r = new double[n][d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) r[i][j] = X.get(i, j) * (max[j] - min[j]) + min[j];
        return NumJa.array(r);
    }

    public double[] getMin() { return min.clone(); }
    public double[] getMax() { return max.clone(); }
    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }
}
