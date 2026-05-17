package sklearn.linear_model;

import numja.core.NDArray;
import numja.NumJa;

/**
 * Logistic Regression - Binary and Multi-class Classification
 * Uses gradient descent with sigmoid (binary) or softmax (multi-class)
 */
public class LogisticRegression {
    private double[][] weights;  // [n_classes][n_features]
    private double[] bias;
    private double learning_rate;
    private int max_iter;
    private double tol;
    private int n_classes;
    private boolean fitted = false;
    private boolean verbose = false;

    public LogisticRegression() {
        this(0.01, 1000, 1e-4);
    }

    public LogisticRegression(double learning_rate, int max_iter, double tol) {
        this.learning_rate = learning_rate;
        this.max_iter = max_iter;
        this.tol = tol;
    }

    public LogisticRegression setLearningRate(double lr) {
        this.learning_rate = lr;
        return this;
    }

    public LogisticRegression setMaxIter(int iter) {
        this.max_iter = iter;
        return this;
    }

    public LogisticRegression setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /**
     * Fit logistic regression model
     * @param X training features (n_samples × n_features)
     * @param y target labels (0, 1, ..., n_classes-1)
     * @return this LogisticRegression
     */
    public LogisticRegression fit(NDArray X, int[] y) {
        double[][] X_data = extract2DArray(X);
        int n_samples = X_data.length;
        int n_features = X_data[0].length;

        // Determine number of classes
        n_classes = 0;
        for (int label : y) {
            if (label > n_classes) n_classes = label;
        }
        n_classes++;

        weights = new double[n_classes][n_features];
        bias = new double[n_classes];

        // Gradient descent
        double prev_loss = Double.MAX_VALUE;

        for (int iter = 0; iter < max_iter; iter++) {
            double loss = 0;

            // Compute gradients over all samples
            double[][] grad_w = new double[n_classes][n_features];
            double[] grad_b = new double[n_classes];

            for (int i = 0; i < n_samples; i++) {
                double[] probs = computeProbabilities(X_data[i]);

                for (int c = 0; c < n_classes; c++) {
                    double error = probs[c] - (y[i] == c ? 1.0 : 0.0);
                    for (int j = 0; j < n_features; j++) {
                        grad_w[c][j] += error * X_data[i][j];
                    }
                    grad_b[c] += error;
                }

                // Cross-entropy loss
                loss -= Math.log(Math.max(probs[y[i]], 1e-10));
            }

            // Update weights
            for (int c = 0; c < n_classes; c++) {
                for (int j = 0; j < n_features; j++) {
                    weights[c][j] -= learning_rate * grad_w[c][j] / n_samples;
                }
                bias[c] -= learning_rate * grad_b[c] / n_samples;
            }

            loss /= n_samples;
            
            if (verbose && iter % 100 == 0) {
                System.out.println("Iteration " + iter + ", Loss: " + loss);
            }

            if (Math.abs(prev_loss - loss) < tol) break;
            prev_loss = loss;
        }

        fitted = true;
        return this;
    }

    /**
     * Predict class labels
     */
    public int[] predict(NDArray X) {
        checkFitted();
        double[][] X_data = extract2DArray(X);
        int[] predictions = new int[X_data.length];

        for (int i = 0; i < X_data.length; i++) {
            double[] probs = computeProbabilities(X_data[i]);
            predictions[i] = argmax(probs);
        }

        return predictions;
    }

    /**
     * Predict class probabilities
     */
    public double[][] predictProba(NDArray X) {
        checkFitted();
        double[][] X_data = extract2DArray(X);
        double[][] proba = new double[X_data.length][n_classes];

        for (int i = 0; i < X_data.length; i++) {
            proba[i] = computeProbabilities(X_data[i]);
        }

        return proba;
    }

    /**
     * Compute accuracy score
     */
    public double score(NDArray X, int[] y) {
        int[] pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (pred[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }

    public double[][] getCoefficients() { return weights; }
    public double[] getIntercept() { return bias; }

    private double[] computeProbabilities(double[] x) {
        double[] logits = new double[n_classes];
        for (int c = 0; c < n_classes; c++) {
            logits[c] = bias[c];
            for (int j = 0; j < x.length; j++) {
                logits[c] += weights[c][j] * x[j];
            }
        }
        return softmax(logits);
    }

    private double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > max) max = v;

        double[] exp = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < x.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }

    private int argmax(double[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
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
