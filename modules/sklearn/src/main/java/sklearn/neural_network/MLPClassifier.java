package sklearn.neural_network;

import numja.core.NDArray;
import numja.NumJa;
import java.util.Random;

/**
 * Multi-Layer Perceptron (MLP) - Feedforward Neural Network
 * Supports arbitrary number of hidden layers with full backpropagation training
 */
public class MLPClassifier {
    private int[] layer_sizes;
    private double[][][] weights;  // weights[l][i][j]
    private double[][] biases;
    private double learning_rate;
    private int max_iter;
    private Random random;
    private String activation;
    private boolean verbose = false;

    public MLPClassifier(int... hidden_layer_sizes) {
        this.learning_rate = 0.01;
        this.max_iter = 200;
        this.random = new Random(42);
        this.activation = "relu";
        this.layer_sizes = new int[hidden_layer_sizes.length + 2];
        this.layer_sizes[0] = -1;
        for (int i = 0; i < hidden_layer_sizes.length; i++) {
            this.layer_sizes[i + 1] = hidden_layer_sizes[i];
        }
        this.layer_sizes[hidden_layer_sizes.length + 1] = -1;
    }

    public MLPClassifier setLearningRate(double lr) {
        this.learning_rate = lr;
        return this;
    }

    public MLPClassifier setActivation(String act) {
        this.activation = act;
        return this;
    }

    public MLPClassifier setMaxIter(int iter) {
        this.max_iter = iter;
        return this;
    }

    public MLPClassifier setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public MLPClassifier fit(NDArray X, int[] y) {
        int n_samples = X.getShape()[0];
        int n_features = X.getShape()[1];
        int n_classes = findMaxLabel(y) + 1;

        layer_sizes[0] = n_features;
        layer_sizes[layer_sizes.length - 1] = n_classes;
        initializeWeights();

        double[][] X_data = extract2DArray(X);

        for (int iter = 0; iter < max_iter; iter++) {
            double loss = 0;

            for (int i = 0; i < n_samples; i++) {
                // Forward pass
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
                        activations[l + 1] = Activations.softmax(z);
                    } else {
                        double[] a = new double[out_size];
                        for (int j = 0; j < out_size; j++) {
                            a[j] = applyActivation(z[j]);
                        }
                        activations[l + 1] = a;
                    }
                }

                // Compute cross-entropy loss
                double[] output = activations[activations.length - 1];
                for (int k = 0; k < n_classes; k++) {
                    if (k == y[i]) {
                        loss -= Math.log(Math.max(output[k], 1e-10));
                    }
                }

                // Backward pass - full backpropagation
                double[][] deltas = new double[layer_sizes.length][];

                // Output layer delta: softmax + cross-entropy => delta = output - one_hot
                double[] output_delta = new double[n_classes];
                for (int k = 0; k < n_classes; k++) {
                    output_delta[k] = output[k] - (k == y[i] ? 1.0 : 0.0);
                }
                deltas[layer_sizes.length - 1] = output_delta;

                // Hidden layers delta
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
                System.out.println("Epoch " + iter + ", Loss: " + (loss / n_samples));
            }
        }

        return this;
    }

    public int[] predict(NDArray X) {
        double[][] X_data = extract2DArray(X);
        int[] predictions = new int[X_data.length];

        for (int i = 0; i < X_data.length; i++) {
            double[] output = forwardPass(X_data[i]);
            predictions[i] = argmax(output);
        }

        return predictions;
    }

    public double[][] predictProba(NDArray X) {
        double[][] X_data = extract2DArray(X);
        double[][] proba = new double[X_data.length][];

        for (int i = 0; i < X_data.length; i++) {
            proba[i] = forwardPass(X_data[i]);
        }

        return proba;
    }

    public double score(NDArray X, int[] y) {
        int[] pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (pred[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
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
                current = Activations.softmax(z);
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

    private int findMaxLabel(int[] labels) {
        int max = 0;
        for (int label : labels) {
            if (label > max) max = label;
        }
        return max;
    }

    private int argmax(double[] arr) {
        int max_idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[max_idx]) max_idx = i;
        }
        return max_idx;
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
