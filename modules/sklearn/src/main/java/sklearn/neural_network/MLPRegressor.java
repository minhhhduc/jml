package sklearn.neural_network;

import numja.core.NDArray;
import numja.NumJa;
import java.util.Random;

/**
 * Multi-Layer Perceptron (MLP) for Regression
 * Full backpropagation with MSE loss
 */
public class MLPRegressor {
    private int[] layer_sizes;
    private double[][][] weights;
    private double[][] biases;
    private double learning_rate;
    private int max_iter;
    private Random random;
    private String activation;
    private boolean verbose = false;

    public MLPRegressor(int... hidden_layer_sizes) {
        this.learning_rate = 0.01;
        this.max_iter = 200;
        this.random = new Random(42);
        this.activation = "relu";
        this.layer_sizes = new int[hidden_layer_sizes.length + 2];
        this.layer_sizes[0] = -1;
        for (int i = 0; i < hidden_layer_sizes.length; i++) {
            this.layer_sizes[i + 1] = hidden_layer_sizes[i];
        }
        this.layer_sizes[hidden_layer_sizes.length + 1] = 1;
    }

    public MLPRegressor setLearningRate(double lr) {
        this.learning_rate = lr;
        return this;
    }

    public MLPRegressor setActivation(String act) {
        this.activation = act;
        return this;
    }

    public MLPRegressor setMaxIter(int iter) {
        this.max_iter = iter;
        return this;
    }

    public MLPRegressor setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public MLPRegressor fit(NDArray X, double[] y) {
        int n_samples = X.getShape()[0];
        int n_features = X.getShape()[1];

        layer_sizes[0] = n_features;
        initializeWeights();

        double[][] X_data = extract2DArray(X);

        for (int iter = 0; iter < max_iter; iter++) {
            double loss = 0;

            for (int i = 0; i < n_samples; i++) {
                // Forward pass - store z and activations
                double[][] z_values = new double[layer_sizes.length][];
                double[][] activations = new double[layer_sizes.length][];
                activations[0] = X_data[i];
                z_values[0] = X_data[i];

                for (int l = 0; l < layer_sizes.length - 1; l++) {
                    int out_size = layer_sizes[l + 1];
                    double[] z = new double[out_size];

                    for (int j = 0; j < out_size; j++) {
                        z[j] = biases[l][j];
                        for (int k = 0; k < layer_sizes[l]; k++) {
                            z[j] += weights[l][k][j] * activations[l][k];
                        }
                    }
                    z_values[l + 1] = z;

                    if (l == layer_sizes.length - 2) {
                        // Output layer: linear activation for regression
                        activations[l + 1] = z.clone();
                    } else {
                        double[] a = new double[out_size];
                        for (int j = 0; j < out_size; j++) {
                            a[j] = applyActivation(z[j]);
                        }
                        activations[l + 1] = a;
                    }
                }

                double pred = activations[activations.length - 1][0];
                double error = pred - y[i];
                loss += error * error;

                // Backward pass
                double[][] deltas = new double[layer_sizes.length][];

                // Output layer delta: d(MSE)/d(output) = 2*(pred - y) / n
                // Simplified: delta = (pred - y) for SGD
                deltas[layer_sizes.length - 1] = new double[]{error};

                // Hidden layers
                for (int l = layer_sizes.length - 2; l >= 1; l--) {
                    int current_size = layer_sizes[l];
                    int next_size = layer_sizes[l + 1];
                    double[] delta = new double[current_size];

                    for (int j = 0; j < current_size; j++) {
                        double sum = 0;
                        for (int k = 0; k < next_size; k++) {
                            sum += weights[l][j][k] * deltas[l + 1][k];
                        }
                        delta[j] = sum * activationDerivative(z_values[l][j]);
                    }
                    deltas[l] = delta;
                }

                // Update weights and biases
                for (int l = 0; l < layer_sizes.length - 1; l++) {
                    int in_size = layer_sizes[l];
                    int out_size = layer_sizes[l + 1];

                    for (int j = 0; j < in_size; j++) {
                        for (int k = 0; k < out_size; k++) {
                            weights[l][j][k] -= learning_rate * deltas[l + 1][k] * activations[l][j];
                        }
                    }
                    for (int k = 0; k < out_size; k++) {
                        biases[l][k] -= learning_rate * deltas[l + 1][k];
                    }
                }
            }

            if (verbose && iter % 50 == 0) {
                System.out.println("Epoch " + iter + ", MSE Loss: " + (loss / n_samples));
            }
        }

        return this;
    }

    public double[] predict(NDArray X) {
        double[][] X_data = extract2DArray(X);
        double[] predictions = new double[X_data.length];

        for (int i = 0; i < X_data.length; i++) {
            predictions[i] = forwardPass(X_data[i])[0];
        }

        return predictions;
    }

    public double score(NDArray X, double[] y) {
        double[] pred = predict(X);
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

    private double[] forwardPass(double[] x) {
        double[] current = x;
        for (int l = 0; l < layer_sizes.length - 1; l++) {
            int out_size = layer_sizes[l + 1];
            double[] z = new double[out_size];

            for (int j = 0; j < out_size; j++) {
                z[j] = biases[l][j];
                for (int k = 0; k < current.length; k++) {
                    z[j] += weights[l][k][j] * current[k];
                }
            }

            if (l == layer_sizes.length - 2) {
                current = z;
            } else {
                double[] a = new double[out_size];
                for (int j = 0; j < out_size; j++) {
                    a[j] = applyActivation(z[j]);
                }
                current = a;
            }
        }
        return current;
    }

    private void initializeWeights() {
        int n_layers = layer_sizes.length;
        weights = new double[n_layers - 1][][];
        biases = new double[n_layers - 1][];

        for (int l = 0; l < n_layers - 1; l++) {
            int in_size = layer_sizes[l];
            int out_size = layer_sizes[l + 1];

            double scale = activation.equals("relu") ?
                Math.sqrt(2.0 / in_size) : Math.sqrt(1.0 / in_size);

            weights[l] = new double[in_size][out_size];
            biases[l] = new double[out_size];

            for (int i = 0; i < in_size; i++) {
                for (int j = 0; j < out_size; j++) {
                    weights[l][i][j] = (random.nextDouble() - 0.5) * 2 * scale;
                }
            }
        }
    }

    private double applyActivation(double x) {
        switch (activation) {
            case "sigmoid": return Activations.sigmoid(x);
            case "tanh": return Activations.tanh(x);
            default: return Activations.relu(x);
        }
    }

    private double activationDerivative(double x) {
        switch (activation) {
            case "sigmoid": return Activations.sigmoidDerivative(x);
            case "tanh": return Activations.tanhDerivative(x);
            default: return Activations.reluDerivative(x);
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
