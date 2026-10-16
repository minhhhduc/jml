package sklearn.svm;

import numja.core.NDArray;

/**
 * Support Vector Regression (SVR)
 * Epsilon-insensitive SVR using simplified SMO
 * Supports linear and RBF kernels
 */
public class SVR {
    private double C;
    private double epsilon;
    private String kernel;
    private double gamma;
    private int max_iter;
    private double tol;
    private double[] alphaPlus, alphaMinus;
    private double b;
    private double[][] X_train;
    private double[] y_train;
    private boolean fitted = false;

    public SVR() { this(1.0, 0.1, "rbf"); }
    public SVR(double C, double epsilon, String kernel) {
        this.C = C; this.epsilon = epsilon; this.kernel = kernel;
        this.gamma = -1; this.max_iter = 1000; this.tol = 1e-3;
    }
    public SVR setC(double C) { this.C = C; return this; }
    public SVR setEpsilon(double e) { this.epsilon = e; return this; }
    public SVR setKernel(String k) { this.kernel = k; return this; }
    public SVR setGamma(double g) { this.gamma = g; return this; }

    public SVR fit(NDArray X, double[] y) {
        X_train = extract2D(X); y_train = y.clone();
        int n = X_train.length;
        if (gamma < 0) gamma = 1.0 / X_train[0].length;

        alphaPlus = new double[n]; alphaMinus = new double[n]; b = 0;

        // Simplified SMO for SVR
        for (int iter = 0; iter < max_iter; iter++) {
            int changed = 0;
            for (int i = 0; i < n; i++) {
                double fi = predict_single(X_train[i]);
                double error = fi - y[i];

                // Check KKT conditions
                boolean violates = false;
                if (error > epsilon + tol && (alphaPlus[i] > 0 || alphaMinus[i] < C)) violates = true;
                if (error < -epsilon - tol && (alphaMinus[i] > 0 || alphaPlus[i] < C)) violates = true;

                if (violates) {
                    int j = (i + 1 + (int)(Math.random() * (n - 1))) % n;
                    double fj = predict_single(X_train[j]);

                    double Kii = kernelFunc(X_train[i], X_train[i]);
                    double Kjj = kernelFunc(X_train[j], X_train[j]);
                    double Kij = kernelFunc(X_train[i], X_train[j]);
                    double eta = Kii + Kjj - 2 * Kij;
                    if (eta < 1e-12) continue;

                    // Update using gradient descent step
                    double grad_i = error;
                    if (grad_i > epsilon) grad_i -= epsilon;
                    else if (grad_i < -epsilon) grad_i += epsilon;
                    else grad_i = 0;

                    double delta = grad_i / eta;
                    double oldAp = alphaPlus[i], oldAm = alphaMinus[i];

                    if (delta > 0) {
                        alphaPlus[i] = Math.min(C, alphaPlus[i] + delta);
                        alphaMinus[i] = Math.max(0, alphaMinus[i] - delta);
                    } else {
                        alphaMinus[i] = Math.min(C, alphaMinus[i] - delta);
                        alphaPlus[i] = Math.max(0, alphaPlus[i] + delta);
                    }

                    if (Math.abs(alphaPlus[i] - oldAp) > 1e-5 || Math.abs(alphaMinus[i] - oldAm) > 1e-5) {
                        // Update bias
                        b = 0; int sv = 0;
                        for (int k = 0; k < n; k++) {
                            double a = alphaPlus[k] - alphaMinus[k];
                            if (Math.abs(a) > 1e-10) {
                                b += y[k] - predict_no_bias(X_train[k]);
                                sv++;
                            }
                        }
                        if (sv > 0) b /= sv;
                        changed++;
                    }
                }
            }
            if (changed == 0) break;
        }
        fitted = true;
        return this;
    }

    public double[] predict(NDArray X) {
        check(); double[][] data = extract2D(X);
        double[] pred = new double[data.length];
        for (int i = 0; i < data.length; i++) pred[i] = predict_single(data[i]);
        return pred;
    }

    public double score(NDArray X, double[] y) {
        double[] p = predict(X);
        double my = 0; for (double v : y) my += v; my /= y.length;
        double sst = 0, ssr = 0;
        for (int i = 0; i < y.length; i++) { sst += (y[i]-my)*(y[i]-my); ssr += (y[i]-p[i])*(y[i]-p[i]); }
        return 1.0 - ssr / sst;
    }

    private double predict_single(double[] x) { return predict_no_bias(x) + b; }

    private double predict_no_bias(double[] x) {
        double sum = 0;
        for (int i = 0; i < X_train.length; i++) {
            double a = alphaPlus[i] - alphaMinus[i];
            if (Math.abs(a) > 1e-10) sum += a * kernelFunc(X_train[i], x);
        }
        return sum;
    }

    private double kernelFunc(double[] a, double[] b) {
        if (kernel.equals("linear")) {
            double dot = 0; for (int i = 0; i < a.length; i++) dot += a[i] * b[i]; return dot;
        } else {
            double sum = 0; for (int i = 0; i < a.length; i++) { double d = a[i] - b[i]; sum += d * d; }
            return Math.exp(-gamma * sum);
        }
    }

    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }
    private double[][] extract2D(NDArray arr) {
        int[] s = arr.getShape(); double[][] r = new double[s[0]][s[1]];
        for (int i = 0; i < s[0]; i++) for (int j = 0; j < s[1]; j++) r[i][j] = arr.get(i, j);
        return r;
    }
}
