package sklearn.neural_network;

import numja.core.NDArray;
import numja.NumJa;
import java.util.Random;

/**
 * Autoencoder - Unsupervised learning for dimensionality reduction and feature learning
 * Architecture: Input → Encoder(hidden→latent) → Decoder(hidden→output)
 * Full backpropagation through both encoder and decoder
 */
public class Autoencoder {
    private int input_dim;
    private int encoding_dim;
    private int hidden_dim;
    private double learning_rate;
    private int max_iter;
    private Random random;

    // Encoder weights
    private double[][] enc_w1, enc_w2;
    private double[] enc_b1, enc_b2;

    // Decoder weights
    private double[][] dec_w1, dec_w2;
    private double[] dec_b1, dec_b2;

    public Autoencoder(int input_dim, int encoding_dim) {
        this(input_dim, encoding_dim, 64);
    }

    public Autoencoder(int input_dim, int encoding_dim, int hidden_dim) {
        this.input_dim = input_dim;
        this.encoding_dim = encoding_dim;
        this.hidden_dim = hidden_dim;
        this.learning_rate = 0.001;
        this.max_iter = 100;
        this.random = new Random(42);
        initializeWeights();
    }

    public Autoencoder setLearningRate(double lr) {
        this.learning_rate = lr;
        return this;
    }

    public Autoencoder setMaxIter(int iter) {
        this.max_iter = iter;
        return this;
    }

    public Autoencoder fit(NDArray X) {
        double[][] X_data = extract2DArray(X);
        int n_samples = X_data.length;

        for (int iter = 0; iter < max_iter; iter++) {
            double loss = 0;

            for (int i = 0; i < n_samples; i++) {
                // Forward: encode
                double[] enc_h_z = linearTransform(X_data[i], enc_w1, enc_b1, hidden_dim, input_dim);
                double[] enc_h = reluArray(enc_h_z);
                double[] z_latent = linearTransform(enc_h, enc_w2, enc_b2, encoding_dim, hidden_dim);

                // Forward: decode
                double[] dec_h_z = linearTransform(z_latent, dec_w1, dec_b1, hidden_dim, encoding_dim);
                double[] dec_h = reluArray(dec_h_z);
                double[] output_z = linearTransform(dec_h, dec_w2, dec_b2, input_dim, hidden_dim);
                double[] output = sigmoidArray(output_z);

                // MSE loss
                double[] error = new double[input_dim];
                for (int j = 0; j < input_dim; j++) {
                    error[j] = output[j] - X_data[i][j];
                    loss += error[j] * error[j];
                }

                // Backprop through decoder
                // d_output = error * sigmoid'(output_z)
                double[] d_output = new double[input_dim];
                for (int j = 0; j < input_dim; j++) {
                    d_output[j] = error[j] * output[j] * (1 - output[j]);
                }

                // Update dec_w2, dec_b2
                for (int j = 0; j < hidden_dim; j++) {
                    for (int k = 0; k < input_dim; k++) {
                        dec_w2[j][k] -= learning_rate * d_output[k] * dec_h[j];
                    }
                }
                for (int k = 0; k < input_dim; k++) {
                    dec_b2[k] -= learning_rate * d_output[k];
                }

                // d_dec_h = dec_w2^T * d_output * relu'(dec_h_z)
                double[] d_dec_h = new double[hidden_dim];
                for (int j = 0; j < hidden_dim; j++) {
                    double sum = 0;
                    for (int k = 0; k < input_dim; k++) {
                        sum += dec_w2[j][k] * d_output[k];
                    }
                    d_dec_h[j] = sum * (dec_h_z[j] > 0 ? 1.0 : 0.0);
                }

                // Update dec_w1, dec_b1
                for (int j = 0; j < encoding_dim; j++) {
                    for (int k = 0; k < hidden_dim; k++) {
                        dec_w1[j][k] -= learning_rate * d_dec_h[k] * z_latent[j];
                    }
                }
                for (int k = 0; k < hidden_dim; k++) {
                    dec_b1[k] -= learning_rate * d_dec_h[k];
                }

                // d_z_latent = dec_w1^T * d_dec_h (linear activation in latent)
                double[] d_z_latent = new double[encoding_dim];
                for (int j = 0; j < encoding_dim; j++) {
                    double sum = 0;
                    for (int k = 0; k < hidden_dim; k++) {
                        sum += dec_w1[j][k] * d_dec_h[k];
                    }
                    d_z_latent[j] = sum;
                }

                // Update enc_w2, enc_b2
                for (int j = 0; j < hidden_dim; j++) {
                    for (int k = 0; k < encoding_dim; k++) {
                        enc_w2[j][k] -= learning_rate * d_z_latent[k] * enc_h[j];
                    }
                }
                for (int k = 0; k < encoding_dim; k++) {
                    enc_b2[k] -= learning_rate * d_z_latent[k];
                }

                // d_enc_h = enc_w2^T * d_z_latent * relu'(enc_h_z)
                double[] d_enc_h = new double[hidden_dim];
                for (int j = 0; j < hidden_dim; j++) {
                    double sum = 0;
                    for (int k = 0; k < encoding_dim; k++) {
                        sum += enc_w2[j][k] * d_z_latent[k];
                    }
                    d_enc_h[j] = sum * (enc_h_z[j] > 0 ? 1.0 : 0.0);
                }

                // Update enc_w1, enc_b1
                for (int j = 0; j < input_dim; j++) {
                    for (int k = 0; k < hidden_dim; k++) {
                        enc_w1[j][k] -= learning_rate * d_enc_h[k] * X_data[i][j];
                    }
                }
                for (int k = 0; k < hidden_dim; k++) {
                    enc_b1[k] -= learning_rate * d_enc_h[k];
                }
            }

            if (iter % 20 == 0) {
                System.out.println("Autoencoder Epoch " + iter + ", Loss: " + (loss / n_samples));
            }
        }

        return this;
    }

    public NDArray transform(NDArray X) {
        double[][] X_data = extract2DArray(X);
        double[][] encoded = new double[X_data.length][encoding_dim];
        for (int i = 0; i < X_data.length; i++) {
            encoded[i] = encode(X_data[i]);
        }
        return NumJa.array(encoded);
    }

    public NDArray inverseTransform(NDArray Z) {
        double[][] Z_data = extract2DArray(Z);
        double[][] decoded = new double[Z_data.length][input_dim];
        for (int i = 0; i < Z_data.length; i++) {
            decoded[i] = decode(Z_data[i]);
        }
        return NumJa.array(decoded);
    }

    public NDArray fitTransform(NDArray X) {
        fit(X);
        return transform(X);
    }

    private double[] encode(double[] x) {
        double[] h = reluArray(linearTransform(x, enc_w1, enc_b1, hidden_dim, input_dim));
        return linearTransform(h, enc_w2, enc_b2, encoding_dim, hidden_dim);
    }

    private double[] decode(double[] z) {
        double[] h = reluArray(linearTransform(z, dec_w1, dec_b1, hidden_dim, encoding_dim));
        return sigmoidArray(linearTransform(h, dec_w2, dec_b2, input_dim, hidden_dim));
    }

    private double[] linearTransform(double[] input, double[][] w, double[] b, int out_dim, int in_dim) {
        double[] output = new double[out_dim];
        for (int j = 0; j < out_dim; j++) {
            output[j] = b[j];
            for (int i = 0; i < in_dim; i++) {
                output[j] += w[i][j] * input[i];
            }
        }
        return output;
    }

    private double[] reluArray(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.max(0, x[i]);
        }
        return result;
    }

    private double[] sigmoidArray(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Activations.sigmoid(x[i]);
        }
        return result;
    }

    private void initializeWeights() {
        double scale_enc = Math.sqrt(2.0 / input_dim);
        double scale_dec = Math.sqrt(2.0 / encoding_dim);

        enc_w1 = randomMatrix(input_dim, hidden_dim, scale_enc);
        enc_b1 = new double[hidden_dim];
        enc_w2 = randomMatrix(hidden_dim, encoding_dim, scale_enc);
        enc_b2 = new double[encoding_dim];

        dec_w1 = randomMatrix(encoding_dim, hidden_dim, scale_dec);
        dec_b1 = new double[hidden_dim];
        dec_w2 = randomMatrix(hidden_dim, input_dim, scale_dec);
        dec_b2 = new double[input_dim];
    }

    private double[][] randomMatrix(int rows, int cols, double scale) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (random.nextDouble() - 0.5) * 2 * scale;
            }
        }
        return matrix;
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
