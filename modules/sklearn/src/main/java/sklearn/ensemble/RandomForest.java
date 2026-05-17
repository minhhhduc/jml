package sklearn.ensemble;

import numja.core.NDArray;
import sklearn.tree.DecisionTree;
import sklearn.utils.ParallelUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

/**
 * Random Forest - Ensemble of Decision Trees with bagging
 * Unified class: supports both classification and regression
 */
public class RandomForest {
    private int n_estimators;
    private int max_depth;
    private int min_samples_split;
    private String mode;
    private int n_jobs;
    private DecisionTree[] trees;
    private Random random;
    private boolean fitted = false;

    public RandomForest() { this(100, 10, 2, "classifier", 0); }
    public RandomForest(int n_estimators, String mode) { this(n_estimators, 10, 2, mode, 0); }
    public RandomForest(int n_estimators, int max_depth, int min_samples_split, String mode) {
        this(n_estimators, max_depth, min_samples_split, mode, 0);
    }
    public RandomForest(int n_estimators, int max_depth, int min_samples_split, String mode, int n_jobs) {
        this.n_estimators = n_estimators;
        this.max_depth = max_depth;
        this.min_samples_split = min_samples_split;
        this.mode = mode;
        this.n_jobs = n_jobs;
        this.random = new Random(42);
    }

    /** Fit for classification */
    public RandomForest fit(NDArray X, int[] y) {
        double[][] data = extract2D(X);
        int n = data.length, d = data[0].length;
        this.mode = "classifier";
        trees = new DecisionTree[n_estimators];

        List<Callable<DecisionTree>> tasks = new ArrayList<>();
        for (int t = 0; t < n_estimators; t++) {
            tasks.add(() -> {
                int[] indices = bootstrapIndices(n);
                double[][] bootX = new double[n][d];
                int[] bootY = new int[n];
                for (int i = 0; i < n; i++) {
                    bootX[i] = data[indices[i]];
                    bootY[i] = y[indices[i]];
                }
                DecisionTree tree = new DecisionTree(max_depth, min_samples_split, "classifier");
                tree.fit(numja.NumJa.array(bootX), bootY);
                return tree;
            });
        }
        
        List<DecisionTree> results = ParallelUtils.execute(tasks, n_jobs);
        for (int t = 0; t < n_estimators; t++) trees[t] = results.get(t);
        fitted = true;
        return this;
    }

    /** Fit for regression */
    public RandomForest fit(NDArray X, double[] y) {
        double[][] data = extract2D(X);
        int n = data.length, d = data[0].length;
        this.mode = "regressor";
        trees = new DecisionTree[n_estimators];

        List<Callable<DecisionTree>> tasks = new ArrayList<>();
        for (int t = 0; t < n_estimators; t++) {
            tasks.add(() -> {
                int[] indices = bootstrapIndices(n);
                double[][] bootX = new double[n][d];
                double[] bootY = new double[n];
                for (int i = 0; i < n; i++) {
                    bootX[i] = data[indices[i]];
                    bootY[i] = y[indices[i]];
                }
                DecisionTree tree = new DecisionTree(max_depth, min_samples_split, "regressor");
                tree.fit(numja.NumJa.array(bootX), bootY);
                return tree;
            });
        }

        List<DecisionTree> results = ParallelUtils.execute(tasks, n_jobs);
        for (int t = 0; t < n_estimators; t++) trees[t] = results.get(t);
        fitted = true;
        return this;
    }

    /** Predict class labels */
    public int[] predict(NDArray X) {
        check();
        if (mode.equals("classifier")) {
            int n = X.getShape()[0];
            int[] pred = new int[n];
            int[][] allPreds = new int[n_estimators][];
            List<Callable<int[]>> tasks = new ArrayList<>();
            for (int t = 0; t < n_estimators; t++) {
                final int currentTree = t;
                tasks.add(() -> trees[currentTree].predict(X));
            }
            List<int[]> res = ParallelUtils.execute(tasks, n_jobs);
            for (int t = 0; t < n_estimators; t++) allPreds[t] = res.get(t);

            for (int i = 0; i < n; i++) {
                // Majority vote
                int maxLabel = 0;
                for (int t = 0; t < n_estimators; t++) if (allPreds[t][i] > maxLabel) maxLabel = allPreds[t][i];
                int[] votes = new int[maxLabel + 1];
                for (int t = 0; t < n_estimators; t++) votes[allPreds[t][i]]++;
                int best = 0;
                for (int c = 1; c <= maxLabel; c++) if (votes[c] > votes[best]) best = c;
                pred[i] = best;
            }
            return pred;
        } else {
            double[] regPred = predictRegression(X);
            int[] pred = new int[regPred.length];
            for (int i = 0; i < regPred.length; i++) pred[i] = (int) Math.round(regPred[i]);
            return pred;
        }
    }

    /** Predict continuous values */
    public double[] predictRegression(NDArray X) {
        check();
        int n = X.getShape()[0];
        double[] pred = new double[n];
        List<Callable<double[]>> tasks = new ArrayList<>();
        for (int t = 0; t < n_estimators; t++) {
            final int currentTree = t;
            tasks.add(() -> trees[currentTree].predictRegression(X));
        }
        List<double[]> res = ParallelUtils.execute(tasks, n_jobs);
        for (int t = 0; t < n_estimators; t++) {
            double[] tp = res.get(t);
            for (int i = 0; i < n; i++) pred[i] += tp[i];
        }
        for (int i = 0; i < n; i++) pred[i] /= n_estimators;
        return pred;
    }

    public double score(NDArray X, int[] y) {
        int[] p = predict(X); int ok = 0;
        for (int i = 0; i < y.length; i++) if (p[i] == y[i]) ok++;
        return (double) ok / y.length;
    }

    public double score(NDArray X, double[] y) {
        double[] p = predictRegression(X);
        double my = 0; for (double v : y) my += v; my /= y.length;
        double sst = 0, ssr = 0;
        for (int i = 0; i < y.length; i++) { sst += (y[i]-my)*(y[i]-my); ssr += (y[i]-p[i])*(y[i]-p[i]); }
        return 1.0 - ssr / sst;
    }

    private int[] bootstrapIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = random.nextInt(n);
        return indices;
    }

    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }
    private double[][] extract2D(NDArray arr) {
        int[] s = arr.getShape(); double[][] r = new double[s[0]][s[1]];
        for (int i = 0; i < s[0]; i++) for (int j = 0; j < s[1]; j++) r[i][j] = arr.get(i, j);
        return r;
    }
}
