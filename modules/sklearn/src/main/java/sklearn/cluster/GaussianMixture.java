package sklearn.cluster;

import numja.core.NDArray;
import numja.NumJa;
import java.util.Random;

/**
 * Gaussian Mixture Model (GMM) clustering
 * Probabilistic model assuming data is generated from a mixture of Gaussians
 */
public class GaussianMixture {
    private int n_components;
    private int max_iter;
    private double tol;
    private Random random;

    private double[] weights;  // mixing coefficients
    private double[][] means;  // component means
    private double[][][] covariances;  // component covariances
    private double[][] responsibilities;  // responsibilities (E-step output)
    private int n_iter_;

    /**
     * Initialize Gaussian Mixture Model
     * @param n_components number of mixture components
     */
    public GaussianMixture(int n_components) {
        this(n_components, 100, 1e-3);
    }

    /**
     * Initialize Gaussian Mixture Model with parameters
     * @param n_components number of components
     * @param max_iter maximum iterations
     * @param tol tolerance for convergence
     */
    public GaussianMixture(int n_components, int max_iter, double tol) {
        this.n_components = n_components;
        this.max_iter = max_iter;
        this.tol = tol;
        this.random = new Random();
    }

    /**
     * Fit GMM to data using EM algorithm
     * @param X input data (n_samples × n_features)
     * @return this GaussianMixture object
     */
    public GaussianMixture fit(NDArray X) {
        int[] shape = X.getShape();
        int n_samples = shape[0];
        int n_features = shape[1];

        // Extract 2D array
        double[][] data = extract2DArray(X);

        // Initialize parameters
        initializeParameters(data);

        double prev_log_likelihood = Double.NEGATIVE_INFINITY;

        for (int iteration = 0; iteration < max_iter; iteration++) {
            // E-step: compute responsibilities
            eStep(data);

            // M-step: update parameters
            mStep(data);

            // Check convergence
            double log_likelihood = computeLogLikelihood(data);
            if (Math.abs(log_likelihood - prev_log_likelihood) < tol) {
                this.n_iter_ = iteration;
                break;
            }
            prev_log_likelihood = log_likelihood;
            this.n_iter_ = iteration + 1;
        }

        return this;
    }

    /**
     * Predict cluster labels (hard assignment)
     * @param X input data
     * @return cluster labels
     */
    public int[] predict(NDArray X) {
        double[][] data = extract2DArray(X);
        int n_samples = data.length;
        int[] labels = new int[n_samples];

        for (int i = 0; i < n_samples; i++) {
            double max_prob = -1;
            int best_cluster = 0;

            for (int k = 0; k < n_components; k++) {
                double prob = weights[k] * gaussianPDF(data[i], means[k], covariances[k]);
                if (prob > max_prob) {
                    max_prob = prob;
                    best_cluster = k;
                }
            }
            labels[i] = best_cluster;
        }

        return labels;
    }

    /**
     * Get cluster means
     * @return means as NDArray
     */
    public NDArray getMeans() {
        return NumJa.array(means);
    }

    /**
     * Get mixing weights
     * @return weights array
     */
    public double[] getWeights() {
        return weights.clone();
    }

    /**
     * Get number of iterations until convergence
     * @return iterations
     */
    public int getNIter() {
        return n_iter_;
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

    private void initializeParameters(double[][] data) {
        int n_features = data[0].length;
        weights = new double[n_components];
        means = new double[n_components][n_features];
        covariances = new double[n_components][n_features][n_features];
        responsibilities = new double[data.length][n_components];

        // Initialize with K-means++ approach
        for (int k = 0; k < n_components; k++) {
            int idx = random.nextInt(data.length);
            means[k] = data[idx].clone();
            weights[k] = 1.0 / n_components;

            // Initialize covariance as identity
            for (int i = 0; i < n_features; i++) {
                covariances[k][i][i] = 1.0;
            }
        }
    }

    private void eStep(double[][] data) {
        int n_samples = data.length;

        for (int i = 0; i < n_samples; i++) {
            double sum = 0;
            for (int k = 0; k < n_components; k++) {
                responsibilities[i][k] = weights[k] * gaussianPDF(data[i], means[k], covariances[k]);
                sum += responsibilities[i][k];
            }

            // Normalize
            if (sum > 0) {
                for (int k = 0; k < n_components; k++) {
                    responsibilities[i][k] /= sum;
                }
            }
        }
    }

    private void mStep(double[][] data) {
        int n_samples = data.length;
        int n_features = data[0].length;

        // Update weights
        double[] Nk = new double[n_components];
        for (int i = 0; i < n_samples; i++) {
            for (int k = 0; k < n_components; k++) {
                Nk[k] += responsibilities[i][k];
            }
        }

        for (int k = 0; k < n_components; k++) {
            weights[k] = Nk[k] / n_samples;
        }

        // Update means
        for (int k = 0; k < n_components; k++) {
            for (int j = 0; j < n_features; j++) {
                means[k][j] = 0;
                for (int i = 0; i < n_samples; i++) {
                    means[k][j] += responsibilities[i][k] * data[i][j];
                }
                if (Nk[k] > 0) {
                    means[k][j] /= Nk[k];
                }
            }
        }

        // Update covariances
        for (int k = 0; k < n_components; k++) {
            for (int i = 0; i < n_features; i++) {
                for (int j = 0; j < n_features; j++) {
                    covariances[k][i][j] = 0;
                    for (int n = 0; n < n_samples; n++) {
                        double diff_i = data[n][i] - means[k][i];
                        double diff_j = data[n][j] - means[k][j];
                        covariances[k][i][j] += responsibilities[n][k] * diff_i * diff_j;
                    }
                    if (Nk[k] > 0) {
                        covariances[k][i][j] /= Nk[k];
                    }
                }
            }
        }
    }

    private double computeLogLikelihood(double[][] data) {
        double log_likelihood = 0;
        for (int i = 0; i < data.length; i++) {
            double sum = 0;
            for (int k = 0; k < n_components; k++) {
                sum += weights[k] * gaussianPDF(data[i], means[k], covariances[k]);
            }
            if (sum > 0) {
                log_likelihood += Math.log(sum);
            }
        }
        return log_likelihood;
    }

    private double gaussianPDF(double[] x, double[] mean, double[][] cov) {
        int n = x.length;
        double det = computeDeterminant(cov);
        if (det <= 0) det = 1e-10;

        double diff_sum = 0;
        for (int i = 0; i < n; i++) {
            double diff = x[i] - mean[i];
            diff_sum += diff * diff / Math.max(cov[i][i], 1e-10);
        }

        return (1.0 / Math.sqrt(Math.pow(2 * Math.PI, n) * det)) * Math.exp(-0.5 * diff_sum);
    }

    private double computeDeterminant(double[][] matrix) {
        if (matrix.length == 1) return matrix[0][0];
        if (matrix.length == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        return 1.0; // Simplified for higher dimensions
    }
}
