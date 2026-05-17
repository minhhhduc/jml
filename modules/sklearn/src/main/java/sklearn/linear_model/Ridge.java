package sklearn.linear_model;

import numja.core.NDArray;

import java.util.Random;

/**
 * Linear least squares with l2 regularization.
 * Minimizes the objective function:
 * ||y - Xw||^2_2 + alpha * ||w||^2_2
 */
public class Ridge {
    private double alpha;
    private int max_iter;
    private double learning_rate;
    private double[] weights;
    private double bias;
    private boolean fitted = false;

    public Ridge() { this(1.0, 1000, 0.01); }
    public Ridge(double alpha) { this(alpha, 1000, 0.01); }
    public Ridge(double alpha, int max_iter, double learning_rate) {
        this.alpha = alpha;
        this.max_iter = max_iter;
        this.learning_rate = learning_rate;
    }

    public Ridge fit(NDArray X, double[] y) {
        int[] shape = X.getShape();
        int n = shape[0];
        int d = shape[1];
        
        double[][] X_data = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) X_data[i][j] = X.get(i, j);
        }

        weights = new double[d];
        bias = 0.0;
        
        // Gradient Descent
        for (int iter = 0; iter < max_iter; iter++) {
            double[] dw = new double[d];
            double db = 0;

            for (int i = 0; i < n; i++) {
                double pred = bias;
                for (int j = 0; j < d; j++) pred += weights[j] * X_data[i][j];
                double error = pred - y[i];

                db += error;
                for (int j = 0; j < d; j++) {
                    dw[j] += error * X_data[i][j];
                }
            }

            // Apply gradients with L2 penalty
            for (int j = 0; j < d; j++) {
                weights[j] -= learning_rate * ((2.0 * dw[j] / n) + 2.0 * alpha * weights[j]);
            }
            bias -= learning_rate * (2.0 * db / n);
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
