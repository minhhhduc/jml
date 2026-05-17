package sklearn.preprocessing;

import numja.core.NDArray;
import numja.NumJa;

/**
 * StandardScaler - Standardize features by removing mean and scaling to unit variance
 * Transforms features to have mean=0 and std=1
 */
public class StandardScaler {
    private double[] mean;
    private double[] scale;  // standard deviation
    private boolean fitted = false;

    /**
     * Fit the scaler to data
     * @param X input data (n_samples × n_features)
     * @return this StandardScaler
     */
    public StandardScaler fit(NDArray X) {
        int[] shape = X.getShape();
        int n_samples = shape[0];
        int n_features = shape[1];

        // Extract 2D array
        double[][] data = extract2DArray(X);

        mean = new double[n_features];
        scale = new double[n_features];

        // Compute mean
        for (int j = 0; j < n_features; j++) {
            double sum = 0;
            for (int i = 0; i < n_samples; i++) {
                sum += data[i][j];
            }
            mean[j] = sum / n_samples;
        }

        // Compute standard deviation
        for (int j = 0; j < n_features; j++) {
            double sum = 0;
            for (int i = 0; i < n_samples; i++) {
                double diff = data[i][j] - mean[j];
                sum += diff * diff;
            }
            scale[j] = Math.sqrt(sum / n_samples);
            if (scale[j] < 1e-10) scale[j] = 1.0;  // Avoid division by zero
        }

        fitted = true;
        return this;
    }

    /**
     * Transform data using fitted parameters
     * @param X input data
     * @return transformed data
     */
    public NDArray transform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler not fitted. Call fit() first.");
        }

        int[] shape = X.getShape();
        int n_samples = shape[0];
        int n_features = shape[1];

        // Extract 2D array
        double[][] data = extract2DArray(X);

        double[][] transformed = new double[n_samples][n_features];

        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_features; j++) {
                transformed[i][j] = (data[i][j] - mean[j]) / scale[j];
            }
        }

        return NumJa.array(transformed);
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
     * Inverse transform (convert back to original scale)
     * @param X transformed data
     * @return data in original scale
     */
    public NDArray inverseTransform(NDArray X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler not fitted. Call fit() first.");
        }

        int[] shape = X.getShape();
        int n_samples = shape[0];
        int n_features = shape[1];

        // Extract 2D array
        double[][] data = extract2DArray(X);

        double[][] original = new double[n_samples][n_features];

        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_features; j++) {
                original[i][j] = data[i][j] * scale[j] + mean[j];
            }
        }

        return NumJa.array(original);
    }

    /**
     * Get the mean for each feature
     * @return mean array
     */
    public double[] getMean() {
        return mean.clone();
    }

    /**
     * Get the standard deviation for each feature
     * @return scale array
     */
    public double[] getScale() {
        return scale.clone();
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
