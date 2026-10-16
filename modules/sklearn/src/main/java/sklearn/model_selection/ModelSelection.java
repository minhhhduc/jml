package sklearn.model_selection;

import numja.core.NDArray;
import numja.NumJa;
import sklearn.utils.ParallelUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.function.Supplier;

/**
 * Model selection utilities: train/test split and cross-validation
 */
public class ModelSelection {

    // ===== TRAIN/TEST SPLIT (Classification) =====
    public static Object[] trainTestSplit(NDArray X, int[] y, double test_size) {
        return trainTestSplit(X, y, test_size, 42);
    }

    public static Object[] trainTestSplit(NDArray X, int[] y, double test_size, int random_state) {
        double[][] X_data = extract2D(X);
        int n = X_data.length;
        int n_test = (int) Math.round(n * test_size);
        int n_train = n - n_test;
        int[] idx = shuffledIndices(n, random_state);

        double[][] Xtr = new double[n_train][], Xte = new double[n_test][];
        int[] ytr = new int[n_train], yte = new int[n_test];
        for (int i = 0; i < n_train; i++) { Xtr[i] = X_data[idx[i]].clone(); ytr[i] = y[idx[i]]; }
        for (int i = 0; i < n_test; i++) { Xte[i] = X_data[idx[n_train+i]].clone(); yte[i] = y[idx[n_train+i]]; }

        return new Object[]{ NumJa.array(Xtr), NumJa.array(Xte), ytr, yte };
    }

    // ===== TRAIN/TEST SPLIT (Regression) =====
    public static Object[] trainTestSplitRegression(NDArray X, double[] y, double test_size) {
        return trainTestSplitRegression(X, y, test_size, 42);
    }

    public static Object[] trainTestSplitRegression(NDArray X, double[] y, double test_size, int random_state) {
        double[][] X_data = extract2D(X);
        int n = X_data.length;
        int n_test = (int) Math.round(n * test_size);
        int n_train = n - n_test;
        int[] idx = shuffledIndices(n, random_state);

        double[][] Xtr = new double[n_train][], Xte = new double[n_test][];
        double[] ytr = new double[n_train], yte = new double[n_test];
        for (int i = 0; i < n_train; i++) { Xtr[i] = X_data[idx[i]].clone(); ytr[i] = y[idx[i]]; }
        for (int i = 0; i < n_test; i++) { Xte[i] = X_data[idx[n_train+i]].clone(); yte[i] = y[idx[n_train+i]]; }

        return new Object[]{ NumJa.array(Xtr), NumJa.array(Xte), ytr, yte };
    }

    // ===== K-FOLD CROSS VALIDATION =====

    /**
     * Functional interface for models that support fit + score
     */
    public interface Estimator {
        void fit(NDArray X_train, int[] y_train);
        double score(NDArray X_test, int[] y_test);
    }

    public interface EstimatorRegression {
        void fit(NDArray X_train, double[] y_train);
        double score(NDArray X_test, double[] y_test);
    }

    /**
     * K-fold cross-validation for classification
     * @param estimator model implementing Estimator interface
     * @param X feature data
     * @param y labels
     * @param cv number of folds
     * @return array of scores for each fold
     */
    public static double[] crossValScore(Estimator estimator, NDArray X, int[] y, int cv) {
        return crossValScore(() -> estimator, X, y, cv, 1); // Default to sequential to avoid state corruption on single instance
    }

    /**
     * K-fold cross-validation for classification with parallelization.
     * Uses a Supplier to create fresh estimator instances for each fold.
     * @param supplier Supplier creating fresh model instances
     * @param X feature data
     * @param y labels
     * @param cv number of folds
     * @param n_jobs number of threads
     * @return array of scores
     */
    public static double[] crossValScore(Supplier<Estimator> supplier, NDArray X, int[] y, int cv, int n_jobs) {
        double[][] data = extract2D(X);
        int n = data.length;
        int foldSize = n / cv;
        int[] indices = shuffledIndices(n, 42);
        double[] scores = new double[cv];

        List<Callable<Double>> tasks = new ArrayList<>();
        for (int fold = 0; fold < cv; fold++) {
            final int currentFold = fold;
            tasks.add(() -> {
                int testStart = currentFold * foldSize;
                int testEnd = (currentFold == cv - 1) ? n : testStart + foldSize;
                int testN = testEnd - testStart;
                int trainN = n - testN;

                double[][] Xtr = new double[trainN][];
                int[] ytr = new int[trainN];
                double[][] Xte = new double[testN][];
                int[] yte = new int[testN];

                int ti = 0, ei = 0;
                for (int i = 0; i < n; i++) {
                    int idx = indices[i];
                    if (i >= testStart && i < testEnd) {
                        Xte[ei] = data[idx]; yte[ei] = y[idx]; ei++;
                    } else {
                        Xtr[ti] = data[idx]; ytr[ti] = y[idx]; ti++;
                    }
                }

                Estimator foldEstimator = supplier.get();
                foldEstimator.fit(NumJa.array(Xtr), ytr);
                return foldEstimator.score(NumJa.array(Xte), yte);
            });
        }

        List<Double> res = ParallelUtils.execute(tasks, n_jobs);
        for (int fold = 0; fold < cv; fold++) {
            scores[fold] = res.get(fold);
        }

        return scores;
    }

    /**
     * K-fold cross-validation for regression
     */
    public static double[] crossValScore(EstimatorRegression estimator, NDArray X, double[] y, int cv) {
        return crossValScore(() -> estimator, X, y, cv, 1);
    }
    
    public static double[] crossValScore(Supplier<EstimatorRegression> supplier, NDArray X, double[] y, int cv, int n_jobs) {
        double[][] data = extract2D(X);
        int n = data.length;
        int foldSize = n / cv;
        int[] indices = shuffledIndices(n, 42);
        double[] scores = new double[cv];

        List<Callable<Double>> tasks = new ArrayList<>();
        for (int fold = 0; fold < cv; fold++) {
            final int currentFold = fold;
            tasks.add(() -> {
                int testStart = currentFold * foldSize;
                int testEnd = (currentFold == cv - 1) ? n : testStart + foldSize;
                int testN = testEnd - testStart;
                int trainN = n - testN;

                double[][] Xtr = new double[trainN][];
                double[] ytr = new double[trainN];
                double[][] Xte = new double[testN][];
                double[] yte = new double[testN];

                int ti = 0, ei = 0;
                for (int i = 0; i < n; i++) {
                    int idx = indices[i];
                    if (i >= testStart && i < testEnd) {
                        Xte[ei] = data[idx]; yte[ei] = y[idx]; ei++;
                    } else {
                        Xtr[ti] = data[idx]; ytr[ti] = y[idx]; ti++;
                    }
                }

                EstimatorRegression foldEstimator = supplier.get();
                foldEstimator.fit(NumJa.array(Xtr), ytr);
                return foldEstimator.score(NumJa.array(Xte), yte);
            });
        }

        List<Double> res = ParallelUtils.execute(tasks, n_jobs);
        for (int fold = 0; fold < cv; fold++) {
            scores[fold] = res.get(fold);
        }

        return scores;
    }

    // ===== UTILITIES =====
    private static int[] shuffledIndices(int n, int seed) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Random random = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
        return indices;
    }

    private static double[][] extract2D(NDArray arr) {
        int[] s = arr.getShape(); double[][] r = new double[s[0]][s[1]];
        for (int i = 0; i < s[0]; i++) for (int j = 0; j < s[1]; j++) r[i][j] = arr.get(i, j);
        return r;
    }
}
