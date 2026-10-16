package sklearn.svm;

import numja.core.NDArray;

/**
 * Support Vector Classification (SVC)
 * Implements SMO (Sequential Minimal Optimization) algorithm
 * Supports linear and RBF kernels
 */
public class SVC {
    private double C;           // Regularization parameter
    private double tol;
    private int max_iter;
    private String kernel;      // "linear" or "rbf"
    private double gamma;       // RBF kernel parameter
    private double[] alphas;
    private double b;
    private double[][] X_train;
    private int[] y_train;
    private int n_classes;
    private boolean fitted = false;

    // For multi-class: one-vs-one classifiers
    private SVC[] ovoPairs;
    private int[][] ovoClasses;

    public SVC() { this(1.0, "rbf"); }
    public SVC(double C, String kernel) {
        this.C = C; this.kernel = kernel; this.tol = 1e-3;
        this.max_iter = 1000; this.gamma = -1;
    }
    public SVC setC(double C) { this.C = C; return this; }
    public SVC setKernel(String k) { this.kernel = k; return this; }
    public SVC setGamma(double g) { this.gamma = g; return this; }
    public SVC setMaxIter(int m) { this.max_iter = m; return this; }

    public SVC fit(NDArray X, int[] y) {
        double[][] data = extract2D(X);
        n_classes = 0; for (int l : y) if (l > n_classes) n_classes = l; n_classes++;

        if (n_classes == 2) {
            fitBinary(data, y);
        } else {
            // One-vs-One for multi-class
            int nPairs = n_classes * (n_classes - 1) / 2;
            ovoPairs = new SVC[nPairs]; ovoClasses = new int[nPairs][2];
            int idx = 0;
            for (int i = 0; i < n_classes; i++) {
                for (int j = i + 1; j < n_classes; j++) {
                    // Extract samples for class i and j
                    int count = 0;
                    for (int l : y) if (l == i || l == j) count++;
                    double[][] subX = new double[count][];
                    int[] subY = new int[count];
                    int si = 0;
                    for (int k = 0; k < y.length; k++) {
                        if (y[k] == i) { subX[si] = data[k]; subY[si] = 1; si++; }
                        else if (y[k] == j) { subX[si] = data[k]; subY[si] = -1; si++; }
                    }
                    ovoPairs[idx] = new SVC(C, kernel);
                    ovoPairs[idx].gamma = this.gamma;
                    ovoPairs[idx].max_iter = this.max_iter;
                    ovoPairs[idx].fitBinaryRaw(subX, subY);
                    ovoClasses[idx] = new int[]{i, j};
                    idx++;
                }
            }
        }
        fitted = true;
        return this;
    }

    private void fitBinary(double[][] data, int[] y) {
        // Convert to +1/-1
        int[] yBin = new int[y.length];
        for (int i = 0; i < y.length; i++) yBin[i] = y[i] == 0 ? -1 : 1;
        fitBinaryRaw(data, yBin);
    }

    private void fitBinaryRaw(double[][] data, int[] y) {
        X_train = data; y_train = y;
        int n = data.length;
        if (gamma < 0) gamma = 1.0 / data[0].length;

        alphas = new double[n]; b = 0;

        // SMO simplified
        for (int iter = 0; iter < max_iter; iter++) {
            int changed = 0;
            for (int i = 0; i < n; i++) {
                double Ei = decision(data[i]) - y[i];
                if ((y[i] * Ei < -tol && alphas[i] < C) || (y[i] * Ei > tol && alphas[i] > 0)) {
                    int j = (i + 1 + (int)(Math.random() * (n - 1))) % n;
                    double Ej = decision(data[j]) - y[j];

                    double oldAi = alphas[i], oldAj = alphas[j];
                    double L, H;
                    if (y[i] != y[j]) { L = Math.max(0, alphas[j] - alphas[i]); H = Math.min(C, C + alphas[j] - alphas[i]); }
                    else { L = Math.max(0, alphas[i] + alphas[j] - C); H = Math.min(C, alphas[i] + alphas[j]); }
                    if (L >= H) continue;

                    double eta = 2 * kernelFunc(data[i], data[j]) - kernelFunc(data[i], data[i]) - kernelFunc(data[j], data[j]);
                    if (eta >= 0) continue;

                    alphas[j] -= y[j] * (Ei - Ej) / eta;
                    alphas[j] = Math.min(H, Math.max(L, alphas[j]));
                    if (Math.abs(alphas[j] - oldAj) < 1e-5) continue;

                    alphas[i] += y[i] * y[j] * (oldAj - alphas[j]);

                    double b1 = b - Ei - y[i] * (alphas[i] - oldAi) * kernelFunc(data[i], data[i])
                        - y[j] * (alphas[j] - oldAj) * kernelFunc(data[i], data[j]);
                    double b2 = b - Ej - y[i] * (alphas[i] - oldAi) * kernelFunc(data[i], data[j])
                        - y[j] * (alphas[j] - oldAj) * kernelFunc(data[j], data[j]);

                    if (0 < alphas[i] && alphas[i] < C) b = b1;
                    else if (0 < alphas[j] && alphas[j] < C) b = b2;
                    else b = (b1 + b2) / 2;
                    changed++;
                }
            }
            if (changed == 0) break;
        }
    }

    public int[] predict(NDArray X) {
        check(); double[][] data = extract2D(X);
        int[] pred = new int[data.length];

        if (n_classes == 2) {
            for (int i = 0; i < data.length; i++) pred[i] = decision(data[i]) >= 0 ? 1 : 0;
        } else {
            for (int i = 0; i < data.length; i++) {
                int[] votes = new int[n_classes];
                for (int p = 0; p < ovoPairs.length; p++) {
                    double d = ovoPairs[p].decision(data[i]);
                    votes[d >= 0 ? ovoClasses[p][0] : ovoClasses[p][1]]++;
                }
                int best = 0; for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = c;
                pred[i] = best;
            }
        }
        return pred;
    }

    public double score(NDArray X, int[] y) {
        int[] p = predict(X); int ok = 0;
        for (int i = 0; i < y.length; i++) if (p[i] == y[i]) ok++;
        return (double) ok / y.length;
    }

    private double decision(double[] x) {
        double sum = b;
        for (int i = 0; i < X_train.length; i++) {
            if (alphas[i] > 0) sum += alphas[i] * y_train[i] * kernelFunc(X_train[i], x);
        }
        return sum;
    }

    private double kernelFunc(double[] a, double[] b) {
        if (kernel.equals("linear")) {
            double dot = 0; for (int i = 0; i < a.length; i++) dot += a[i] * b[i]; return dot;
        } else { // rbf
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
