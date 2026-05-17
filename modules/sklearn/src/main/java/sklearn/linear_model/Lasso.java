package sklearn.linear_model;

import numja.core.NDArray;

/**
 * Linear Model trained with L1 prior as regularizer (aka the Lasso).
 * Minimizes the objective function:
 * (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
 */
public class Lasso {
    private double alpha;
    private int max_iter;
    private double[] weights;
    private double bias;
    private boolean fitted = false;

    public Lasso() { this(1.0, 1000); }
    public Lasso(double alpha) { this(alpha, 1000); }
    public Lasso(double alpha, int max_iter) {
        this.alpha = alpha;
        this.max_iter = max_iter;
    }

    public Lasso fit(NDArray X, double[] y) {
        int[] shape = X.getShape();
        int n = shape[0];
        int d = shape[1];
        
        double[][] X_data = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) X_data[i][j] = X.get(i, j);
        }

        weights = new double[d];
        bias = 0.0;

        // Precompute X column sums of squares
        double[] xSqSums = new double[d];
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < n; i++) {
                xSqSums[j] += X_data[i][j] * X_data[i][j];
            }
        }

        // Coordinate Descent
        for (int iter = 0; iter < max_iter; iter++) {
            // Update bias
            double sumError = 0;
            for (int i = 0; i < n; i++) {
                double pred = bias;
                for (int j = 0; j < d; j++) pred += weights[j] * X_data[i][j];
                sumError += (y[i] - pred);
            }
            bias += sumError / n;

            // Update weights
            for (int j = 0; j < d; j++) {
                if (xSqSums[j] == 0) continue;

                double rho = 0;
                for (int i = 0; i < n; i++) {
                    double pred = bias;
                    for (int k = 0; k < d; k++) {
                        if (k != j) pred += weights[k] * X_data[i][k];
                    }
                    rho += X_data[i][j] * (y[i] - pred);
                }

                // Soft thresholding
                if (rho < -alpha * n) {
                    weights[j] = (rho + alpha * n) / xSqSums[j];
                } else if (rho > alpha * n) {
                    weights[j] = (rho - alpha * n) / xSqSums[j];
                } else {
                    weights[j] = 0.0;
                }
            }
        }

        fitted = true;
        return this;
    }

    public double[] predictRegression(NDArray X) {
        if (!fitted) throw new IllegalStateException("Model not fitted");
        int[] shape = X.getShape();
        int n = shape[0];
        int d = shape[1];

        double[] pred = new double[n];
        for (int i = 0; i < n; i++) {
            double p = bias;
            for (int j = 0; j < d; j++) p += weights[j] * X.get(i, j);
            pred[i] = p;
        }
        return pred;
    }

    public double score(NDArray X, double[] y) {
        double[] p = predictRegression(X);
        double my = 0; for (double v : y) my += v; my /= y.length;
        double sst = 0, ssr = 0;
        for (int i = 0; i < y.length; i++) { sst += (y[i]-my)*(y[i]-my); ssr += (y[i]-p[i])*(y[i]-p[i]); }
        return 1.0 - ssr / sst;
    }
}
