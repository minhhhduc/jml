package sklearn.linear_model;

import numja.core.NDArray;
import numja.NumJa;

/**
 * Linear Regression using Least Squares method
 * Solves: y = X * β + ε
 */
public class LinearRegression {
    private double[] coefficients;
    private double intercept;

    /**
     * Fit linear regression model
     * @param X training features (n_samples × n_features)
     * @param y target values
     * @return this LinearRegression
     */
    public LinearRegression fit(NDArray X, double[] y) {
        double[][] X_data = extract2DArray(X);
        int n_samples = X_data.length;
        int n_features = X_data[0].length;

        // Add bias term (column of 1s)
        double[][] X_with_bias = new double[n_samples][n_features + 1];
        for (int i = 0; i < n_samples; i++) {
            X_with_bias[i][0] = 1.0;  // Bias
            for (int j = 0; j < n_features; j++) {
                X_with_bias[i][j + 1] = X_data[i][j];
            }
        }

        // Normal equation: β = (X^T * X)^-1 * X^T * y
        // Compute X^T
        double[][] XT = transpose(X_with_bias);

        // Compute X^T * X
        double[][] XTX = matmul(XT, X_with_bias);

        // Compute X^T * y
        double[] XTy = new double[n_features + 1];
        for (int i = 0; i < n_features + 1; i++) {
            for (int j = 0; j < n_samples; j++) {
                XTy[i] += XT[i][j] * y[j];
            }
        }

        // Solve system: (X^T * X) * β = X^T * y
        double[] beta = gaussianElimination(XTX, XTy);

        this.intercept = beta[0];
        this.coefficients = new double[n_features];
        for (int i = 0; i < n_features; i++) {
            this.coefficients[i] = beta[i + 1];
        }

        return this;
    }

    /**
     * Predict on new data
     * @param X input features
     * @return predicted values
     */
    public double[] predict(NDArray X) {
        double[][] X_data = extract2DArray(X);
        double[] predictions = new double[X_data.length];

        for (int i = 0; i < X_data.length; i++) {
            predictions[i] = intercept;
            for (int j = 0; j < coefficients.length; j++) {
                predictions[i] += coefficients[j] * X_data[i][j];
            }
        }

        return predictions;
    }

    /**
     * Get coefficients
     * @return regression coefficients
     */
    public double[] getCoefficients() {
        return coefficients.clone();
    }

    /**
     * Get intercept
     * @return bias/intercept term
     */
    public double getIntercept() {
        return intercept;
    }

    /**
     * Compute R² score
     * @param X input features
     * @param y actual values
     * @return R² (0 to 1)
     */
    public double score(NDArray X, double[] y) {
        double[] y_pred = predict(X);

        // Compute mean of y
        double y_mean = 0;
        for (double val : y) {
            y_mean += val;
        }
        y_mean /= y.length;

        // SS_tot = Σ(y_i - mean(y))²
        double ss_tot = 0;
        for (double val : y) {
            ss_tot += (val - y_mean) * (val - y_mean);
        }

        // SS_res = Σ(y_i - ŷ_i)²
        double ss_res = 0;
        for (int i = 0; i < y.length; i++) {
            ss_res += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
        }

        // R² = 1 - SS_res/SS_tot
        return 1.0 - (ss_res / ss_tot);
    }

    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    private double[][] matmul(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = B[0].length;
        int common = A[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    private double[] gaussianElimination(double[][] A, double[] b) {
        int n = A.length;
        double[][] aug = new double[n][n + 1];

        // Create augmented matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = A[i][j];
            }
            aug[i][n] = b[i];
        }

        // Forward elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int max_row = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(aug[k][i]) > Math.abs(aug[max_row][i])) {
                    max_row = k;
                }
            }

            // Swap rows
            double[] temp = aug[i];
            aug[i] = aug[max_row];
            aug[max_row] = temp;

            // Eliminate column
            for (int k = i + 1; k < n; k++) {
                if (aug[i][i] != 0) {
                    double factor = aug[k][i] / aug[i][i];
                    for (int j = i; j <= n; j++) {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }
        }

        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= aug[i][j] * x[j];
            }
            if (aug[i][i] != 0) {
                x[i] /= aug[i][i];
            }
        }

        return x;
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
