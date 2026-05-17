package sklearn.naive_bayes;

import numja.core.NDArray;

/**
 * Gaussian Naive Bayes Classifier
 * P(y|X) proportional to P(y) * product of P(x_i|y)
 */
public class GaussianNB {
    private double[][] means;
    private double[][] variances;
    private double[] classPriors;
    private int n_classes;
    private boolean fitted = false;

    public GaussianNB() {}

    public GaussianNB fit(NDArray X, int[] y) {
        double[][] data = extract2D(X);
        int n = data.length, d = data[0].length;
        n_classes = 0;
        for (int l : y) if (l > n_classes) n_classes = l;
        n_classes++;

        means = new double[n_classes][d];
        variances = new double[n_classes][d];
        classPriors = new double[n_classes];
        int[] counts = new int[n_classes];
        for (int l : y) counts[l]++;
        for (int c = 0; c < n_classes; c++) classPriors[c] = (double) counts[c] / n;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) means[y[i]][j] += data[i][j];
        for (int c = 0; c < n_classes; c++)
            if (counts[c] > 0) for (int j = 0; j < d; j++) means[c][j] /= counts[c];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) {
                double diff = data[i][j] - means[y[i]][j];
                variances[y[i]][j] += diff * diff;
            }
        for (int c = 0; c < n_classes; c++)
            if (counts[c] > 0) for (int j = 0; j < d; j++) variances[c][j] = variances[c][j] / counts[c] + 1e-9;

        fitted = true;
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
