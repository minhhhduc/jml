package sklearn.ensemble;

import numja.core.NDArray;
import numja.NumJa;
import sklearn.tree.DecisionTree;

/**
 * Gradient Boosting for regression.
 * Builds an additive model in a forward stage-wise fashion.
 */
public class GradientBoostingRegressor {
    private int n_estimators;
    private double learning_rate;
    private int max_depth;
    private DecisionTree[] trees;
    private double initialPrediction;
    private boolean fitted = false;

    public GradientBoostingRegressor() { this(100, 0.1, 3); }
    public GradientBoostingRegressor(int n_estimators, double learning_rate, int max_depth) {
        this.n_estimators = n_estimators;
        this.learning_rate = learning_rate;
        this.max_depth = max_depth;
    }

    public GradientBoostingRegressor fit(NDArray X, double[] y) {
        int n = y.length;
        trees = new DecisionTree[n_estimators];

        // Initialize with mean
        double sum = 0;
        for (double v : y) sum += v;
        initialPrediction = sum / n;

        double[] F = new double[n];
        for (int i = 0; i < n; i++) F[i] = initialPrediction;

        double[] residuals = new double[n];
        for (int t = 0; t < n_estimators; t++) {
            // Compute negative gradients (residuals for MSE)
            for (int i = 0; i < n; i++) residuals[i] = y[i] - F[i];

            // Fit tree to residuals
            trees[t] = new DecisionTree(max_depth, 2, "regressor");
            trees[t].fit(X, residuals);

            // Update predictions
            double[] h = trees[t].predictRegression(X);
            for (int i = 0; i < n; i++) {
                F[i] += learning_rate * h[i];
            }
        }

        fitted = true;
        return this;
    }

    public double[] predictRegression(NDArray X) {
        if (!fitted) throw new IllegalStateException("Model not fitted");
        int n = X.getShape()[0];
        double[] pred = new double[n];
        for (int i = 0; i < n; i++) pred[i] = initialPrediction;

        for (int t = 0; t < n_estimators; t++) {
            double[] h = trees[t].predictRegression(X);
            for (int i = 0; i < n; i++) pred[i] += learning_rate * h[i];
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
