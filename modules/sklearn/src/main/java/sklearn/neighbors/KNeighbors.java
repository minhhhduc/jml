package sklearn.neighbors;

import numja.core.NDArray;
import numja.NumJa;
import sklearn.utils.ParallelUtils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * K-Nearest Neighbors - Unified Classifier and Regressor
 * Supports both classification (majority vote) and regression (mean)
 */
public class KNeighbors {
    private int n_neighbors;
    private String mode;  // "classifier" or "regressor"
    private int n_jobs;
    private double[][] X_train;
    private int[] y_train_int;
    private double[] y_train_double;
    private boolean fitted = false;

    public KNeighbors() { this(5, "classifier", 0); }
    public KNeighbors(int n_neighbors) { this(n_neighbors, "classifier", 0); }
    public KNeighbors(int n_neighbors, String mode) { this(n_neighbors, mode, 0); }
    
    public KNeighbors(int n_neighbors, String mode, int n_jobs) {
        this.n_neighbors = n_neighbors;
        this.mode = mode;
        this.n_jobs = n_jobs;
    }

    /**
     * Fit for classification (int labels)
     */
    public KNeighbors fit(NDArray X, int[] y) {
        this.X_train = extract2DArray(X);
        this.y_train_int = y.clone();
        this.mode = "classifier";
        this.fitted = true;
        return this;
    }

    /**
     * Fit for regression (double targets)
     */
    public KNeighbors fit(NDArray X, double[] y) {
        this.X_train = extract2DArray(X);
        this.y_train_double = y.clone();
        this.mode = "regressor";
        this.fitted = true;
        return this;
    }

    /**
     * Predict class labels (classifier mode)
     */
    public int[] predict(NDArray X) {
        checkFitted();
        double[][] X_data = extract2DArray(X);
        int[] predictions = new int[X_data.length];
        
        List<Callable<Integer>> tasks = new ArrayList<>();
        for (int i = 0; i < X_data.length; i++) {
            final double[] x = X_data[i];
            tasks.add(() -> {
                int[] neighbors = findNeighborIndices(x);
                if (mode.equals("classifier")) {
                    return majorityVote(neighbors);
                } else {
                    return (int) Math.round(meanValue(neighbors));
                }
            });
        }
        
        List<Integer> res = ParallelUtils.execute(tasks, n_jobs);
        for (int i = 0; i < X_data.length; i++) {
            predictions[i] = res.get(i);
        }
        return predictions;
    }

    /**
     * Predict continuous values (regressor mode)
     */
    public double[] predictRegression(NDArray X) {
        checkFitted();
        double[][] X_data = extract2DArray(X);
        double[] predictions = new double[X_data.length];

        List<Callable<Double>> tasks = new ArrayList<>();
        for (int i = 0; i < X_data.length; i++) {
            final double[] x = X_data[i];
            tasks.add(() -> {
                int[] neighbors = findNeighborIndices(x);
                return meanValue(neighbors);
            });
        }
        
        List<Double> res = ParallelUtils.execute(tasks, n_jobs);
        for (int i = 0; i < X_data.length; i++) {
            predictions[i] = res.get(i);
        }
        return predictions;
    }

    /**
     * Score for classification (accuracy)
     */
    public double score(NDArray X, int[] y) {
        int[] pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (pred[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }

    /**
     * Score for regression (R²)
     */
    public double score(NDArray X, double[] y) {
        double[] pred = predictRegression(X);
        double mean_y = 0;
        for (double v : y) mean_y += v;
        mean_y /= y.length;

        double ss_tot = 0, ss_res = 0;
        for (int i = 0; i < y.length; i++) {
            ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
            ss_res += (y[i] - pred[i]) * (y[i] - pred[i]);
        }
        return 1.0 - (ss_res / ss_tot);
    }

    private int[] findNeighborIndices(double[] x) {
        int n = X_train.length;
        double[] distances = new double[n];
        int[] indices = new int[n];

        for (int i = 0; i < n; i++) {
            distances[i] = euclideanDistance(x, X_train[i]);
            indices[i] = i;
        }

        // Partial sort to find k nearest
        for (int i = 0; i < n_neighbors; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (distances[j] < distances[minIdx]) {
                    minIdx = j;
                }
            }
            // Swap
            double tmpD = distances[i]; distances[i] = distances[minIdx]; distances[minIdx] = tmpD;
            int tmpI = indices[i]; indices[i] = indices[minIdx]; indices[minIdx] = tmpI;
        }

        return Arrays.copyOf(indices, n_neighbors);
    }

    private int majorityVote(int[] neighborIndices) {
        // Find max label
        int maxLabel = 0;
        for (int idx : neighborIndices) {
            if (y_train_int[idx] > maxLabel) maxLabel = y_train_int[idx];
        }

        int[] counts = new int[maxLabel + 1];
        for (int idx : neighborIndices) {
            counts[y_train_int[idx]]++;
        }

        int bestLabel = 0;
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] > counts[bestLabel]) bestLabel = i;
        }
        return bestLabel;
    }

    private double meanValue(int[] neighborIndices) {
        double sum = 0;
        if (mode.equals("regressor")) {
            for (int idx : neighborIndices) {
                sum += y_train_double[idx];
            }
        } else {
            for (int idx : neighborIndices) {
                sum += y_train_int[idx];
            }
        }
        return sum / neighborIndices.length;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private void checkFitted() {
        if (!fitted) throw new IllegalStateException("Model not fitted. Call fit() first.");
    }

    private double[][] extract2DArray(NDArray arr) {
        int[] shape = arr.getShape();
        int rows = shape[0];
        int cols = shape[1];
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = arr.get(i, j);
            }
        }
        return result;
    }
}
