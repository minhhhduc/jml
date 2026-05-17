package sklearn.ensemble;

import numja.core.NDArray;
import numja.NumJa;
import java.util.HashMap;
import java.util.Map;

/**
 * Gradient Boosting for classification.
 * Uses One-Vs-Rest strategy internally for multi-class classification.
 */
public class GradientBoostingClassifier {
    private int n_estimators;
    private double learning_rate;
    private int max_depth;
    
    private int[] classes;
    private GradientBoostingRegressor[] regressors;
    private boolean fitted = false;

    public GradientBoostingClassifier() { this(100, 0.1, 3); }
    public GradientBoostingClassifier(int n_estimators, double learning_rate, int max_depth) {
        this.n_estimators = n_estimators;
        this.learning_rate = learning_rate;
        this.max_depth = max_depth;
    }

    public GradientBoostingClassifier fit(NDArray X, int[] y) {
        // Find unique classes
        Map<Integer, Integer> classMap = new HashMap<>();
        for (int c : y) classMap.put(c, 1);
        classes = new int[classMap.size()];
        int idx = 0;
        for (int c : classMap.keySet()) classes[idx++] = c;

        // Fit a regressor for each class (One-Vs-Rest)
        regressors = new GradientBoostingRegressor[classes.length];
        
        for (int i = 0; i < classes.length; i++) {
            int targetClass = classes[i];
            double[] y_bin = new double[y.length];
            for (int j = 0; j < y.length; j++) {
                y_bin[j] = (y[j] == targetClass) ? 1.0 : 0.0;
            }

            regressors[i] = new GradientBoostingRegressor(n_estimators, learning_rate, max_depth);
            regressors[i].fit(X, y_bin);
        }

        fitted = true;
        return this;
    }

    public int[] predict(NDArray X) {
        if (!fitted) throw new IllegalStateException("Model not fitted");
        int n = X.getShape()[0];
        int k = classes.length;
        
        double[][] scores = new double[k][];
        for (int i = 0; i < k; i++) {
            scores[i] = regressors[i].predictRegression(X);
        }

        int[] pred = new int[n];
        for (int i = 0; i < n; i++) {
            double maxScore = -Double.MAX_VALUE;
            int bestClass = classes[0];
            for (int c = 0; c < k; c++) {
                if (scores[c][i] > maxScore) {
                    maxScore = scores[c][i];
                    bestClass = classes[c];
                }
            }
            pred[i] = bestClass;
        }
        return pred;
    }

    public double score(NDArray X, int[] y) {
        int[] p = predict(X); int ok = 0;
        for (int i = 0; i < y.length; i++) if (p[i] == y[i]) ok++;
        return (double) ok / y.length;
    }
}
