package sklearn.decomposition;

import numja.core.NDArray;
import numja.NumJa;

/**
 * Principal Component Analysis (PCA)
 * Unsupervised dimensionality reduction using SVD
 */
public class PCA {
    private int n_components;
    private double[] mean;
    private double[][] components;  // Principal components (eigenvectors)
    private double[] explained_variance;  // Eigenvalues
    private boolean fitted = false;

    /**
     * Initialize PCA with number of components
     * @param n_components number of components to keep
     */
    public PCA(int n_components) {
        this.n_components = n_components;
    }

    /**
     * Fit PCA to data
     * @param X input data (n_samples × n_features)
     * @return this PCA
     */
    public PCA fit(NDArray X) {
        double[][] X_data = extract2DArray(X);
        int n_samples = X_data.length;
        int n_features = X_data[0].length;

        // Compute mean
        mean = new double[n_features];
        for (int j = 0; j < n_features; j++) {
            double sum = 0;
            for (int i = 0; i < n_samples; i++) {
                sum += X_data[i][j];
            }
            mean[j] = sum / n_samples;
        }

        // Center data
        double[][] X_centered = new double[n_samples][n_features];
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_features; j++) {
                X_centered[i][j] = X_data[i][j] - mean[j];
            }
        }

        // Compute covariance matrix
        double[][] cov = computeCovariance(X_centered);

        // Perform SVD (simplified: eigendecomposition)
        // In production, use full SVD from EJML
        performEigenDecomposition(cov, n_features);

        fitted = true;
        return this;
    }

    /**
     * Transform data to reduced dimensions
     * @param X input data
     * @return transformed data
     */
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("PCA not fitted. Call fit() first.");
        }

        double[][] X_data = extract2DArray(X);
        int n_samples = X_data.length;
        int n_features = X_data[0].length;

        // Center data
        double[][] X_centered = new double[n_samples][n_features];
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_features; j++) {
                X_centered[i][j] = X_data[i][j] - mean[j];
            }
        }

        // Project onto principal components
        double[][] X_transformed = new double[n_samples][n_components];
        for (int i = 0; i < n_samples; i++) {
            for (int k = 0; k < n_components; k++) {
                for (int j = 0; j < n_features; j++) {
                    X_transformed[i][k] += X_centered[i][j] * components[k][j];
                }
            }
        }

        return NumJa.array(X_transformed);
    }

    /**
     * Fit and transform in one step
     * @param X input data
     * @return transformed data
     */
    public NDArray fitTransform(NDArray X) {
        return fit(X).transform(X);
    }

    /**
     * Inverse transform (approximate reconstruction)
     * @param X transformed data
     * @return reconstructed data
     */
    public NDArray inverseTransform(NDArray X) {
        double[][] X_data = extract2DArray(X);
        int n_samples = X_data.length;
        int n_features = mean.length;

        // Reconstruct: X ≈ Z * components^T + mean
        double[][] X_original = new double[n_samples][n_features];
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_features; j++) {
                X_original[i][j] = mean[j];
                for (int k = 0; k < n_components; k++) {
                    X_original[i][j] += X_data[i][k] * components[k][j];
                }
            }
        }

        return NumJa.array(X_original);
    }

    /**
     * Get explained variance ratio
     * @return variance explained by each component
     */
    public double[] getExplainedVarianceRatio() {
        double[] total = new double[explained_variance.length];
        double sum = 0;
        for (double var : explained_variance) {
            sum += var;
        }

        for (int i = 0; i < total.length; i++) {
            total[i] = explained_variance[i] / sum;
        }
        return total;
    }

    /**
     * Get principal components
     * @return components as NDArray
     */
    public NDArray getComponents() {
        return NumJa.array(components);
    }

    private double[][] computeCovariance(double[][] X_centered) {
        int n_samples = X_centered.length;
        int n_features = X_centered[0].length;

        double[][] cov = new double[n_features][n_features];
        for (int i = 0; i < n_features; i++) {
            for (int j = 0; j < n_features; j++) {
                double sum = 0;
                for (int k = 0; k < n_samples; k++) {
                    sum += X_centered[k][i] * X_centered[k][j];
                }
                cov[i][j] = sum / (n_samples - 1);
            }
        }
        return cov;
    }

    private void performEigenDecomposition(double[][] cov, int n_features) {
        // Simplified: Power iteration method for top eigenvalues/eigenvectors
        components = new double[Math.min(n_components, n_features)][n_features];
        explained_variance = new double[components.length];

        double[][] cov_copy = new double[n_features][n_features];
        for (int i = 0; i < n_features; i++) {
            for (int j = 0; j < n_features; j++) {
                cov_copy[i][j] = cov[i][j];
            }
        }

        for (int k = 0; k < components.length; k++) {
            // Power iteration to find top eigenvector
            double[] v = new double[n_features];
            for (int i = 0; i < n_features; i++) {
                v[i] = Math.random();
            }

            for (int iter = 0; iter < 50; iter++) {
                double[] Av = new double[n_features];
                for (int i = 0; i < n_features; i++) {
                    for (int j = 0; j < n_features; j++) {
                        Av[i] += cov_copy[i][j] * v[j];
                    }
                }

                // Normalize
                double norm = 0;
                for (double val : Av) {
                    norm += val * val;
                }
                norm = Math.sqrt(norm);

                for (int i = 0; i < n_features; i++) {
                    v[i] = Av[i] / norm;
                }
            }

            // Eigenvalue: λ = v^T * A * v
            double eigenvalue = 0;
            double[] Av = new double[n_features];
            for (int i = 0; i < n_features; i++) {
                for (int j = 0; j < n_features; j++) {
                    Av[i] += cov_copy[i][j] * v[j];
                }
                eigenvalue += v[i] * Av[i];
            }

            components[k] = v.clone();
            explained_variance[k] = Math.max(0, eigenvalue);

            // Deflate covariance matrix
            for (int i = 0; i < n_features; i++) {
                for (int j = 0; j < n_features; j++) {
                    cov_copy[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
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
