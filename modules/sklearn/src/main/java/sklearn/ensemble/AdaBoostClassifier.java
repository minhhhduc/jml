package sklearn.ensemble;

import numja.core.NDArray;
import numja.NumJa;
import sklearn.tree.DecisionTree;

import java.util.Random;

/**
 * AdaBoost Classifier.
 * Uses resampling to support base estimators without sample_weight support.
 * Fits Decision Stumps (max_depth=1) iteratively.
 */
public class AdaBoostClassifier {
    private int n_estimators;
    private DecisionTree[] estimators;
    private double[] estimatorWeights;
    private int[] classes;
    private boolean fitted = false;
    private Random random = new Random(42);

    public AdaBoostClassifier() { this(50); }
    public AdaBoostClassifier(int n_estimators) {
        this.n_estimators = n_estimators;
    }

    public AdaBoostClassifier fit(NDArray X, int[] y) {
        int[] shape = X.getShape();
        int n = shape[0];
        int d = shape[1];
        
        double[][] X_data = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) X_data[i][j] = X.get(i, j);
        }

        // Binary mapping assuming labels are -1 and 1 or 0 and 1
        classes = new int[]{ Integer.MAX_VALUE, Integer.MIN_VALUE };
        for (int c : y) {
            if (c < classes[0]) classes[0] = c;
            if (c > classes[1]) classes[1] = c;
        }

        int[] y_mapped = new int[n];
        for (int i = 0; i < n; i++) y_mapped[i] = (y[i] == classes[1]) ? 1 : -1;

        double[] weights = new double[n];
        for (int i = 0; i < n; i++) weights[i] = 1.0 / n;

        estimators = new DecisionTree[n_estimators];
        estimatorWeights = new double[n_estimators];

        for (int t = 0; t < n_estimators; t++) {
            // Resample based on weights
            double[] cumulativeWeights = new double[n];
            cumulativeWeights[0] = weights[0];
            for (int i = 1; i < n; i++) cumulativeWeights[i] = cumulativeWeights[i-1] + weights[i];

            double[][] X_sample = new double[n][d];
            int[] y_sample = new int[n];

            for (int i = 0; i < n; i++) {
                double r = random.nextDouble() * cumulativeWeights[n - 1];
                int idx = 0;
                while (idx < n - 1 && cumulativeWeights[idx] < r) idx++;
                X_sample[i] = X_data[idx];
                y_sample[i] = y_mapped[idx];
            }

            // Fit stump
            DecisionTree stump = new DecisionTree(1, 2, "classifier");
            stump.fit(NumJa.array(X_sample), y_sample);

            // Compute error
            int[] pred = stump.predict(NumJa.array(X_data));
            for (int i = 0; i < n; i++) {
                // DecisionTree returns 0 and 1, map back to -1 and 1
                if (pred[i] == 0) pred[i] = -1;
            }

            double error = 0;
            for (int i = 0; i < n; i++) {
                if (pred[i] != y_mapped[i]) error += weights[i];
            }
            error /= cumulativeWeights[n - 1];

            if (error >= 0.5) {
                // Worse than random guessing, stop
                this.n_estimators = t;
                break;
            }

            double alpha = 0.5 * Math.log((1.0 - error) / Math.max(error, 1e-10));
            estimators[t] = stump;
            estimatorWeights[t] = alpha;

            // Update weights
            double sumW = 0;
            for (int i = 0; i < n; i++) {
                weights[i] *= Math.exp(-alpha * y_mapped[i] * pred[i]);
                sumW += weights[i];
            }
            for (int i = 0; i < n; i++) weights[i] /= sumW;
        }

        fitted = true;
        return this;
    }

    public int[] predict(NDArray X) {
        if (!fitted) throw new IllegalStateException("Model not fitted");
        int[] shape = X.getShape();
        int n = shape[0];

        double[] scores = new double[n];
        for (int t = 0; t < n_estimators; t++) {
            int[] pred = estimators[t].predict(X);
            for (int i = 0; i < n; i++) {
                int p = (pred[i] == 0) ? -1 : 1;
                scores[i] += estimatorWeights[t] * p;
            }
        }

        int[] finalPred = new int[n];
        for (int i = 0; i < n; i++) {
            finalPred[i] = (scores[i] > 0) ? classes[1] : classes[0];
        }
        return finalPred;
    }

    public double score(NDArray X, int[] y) {
        int[] p = predict(X); int ok = 0;
        for (int i = 0; i < y.length; i++) if (p[i] == y[i]) ok++;
        return (double) ok / y.length;
    }
}
