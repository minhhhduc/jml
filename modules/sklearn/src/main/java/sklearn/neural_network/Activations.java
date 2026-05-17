package sklearn.neural_network;

/**
 * Activation functions for neural networks
 */
public class Activations {
    
    /**
     * Sigmoid activation function: 1 / (1 + exp(-x))
     * @param x input value
     * @return sigmoid(x)
     */
    public static double sigmoid(double x) {
        if (x > 500) return 1.0;  // Prevent overflow
        if (x < -500) return 0.0;
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Derivative of sigmoid
     * @param x input value
     * @return sigmoid'(x)
     */
    public static double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }
    
    /**
     * ReLU (Rectified Linear Unit): max(0, x)
     * @param x input value
     * @return relu(x)
     */
    public static double relu(double x) {
        return Math.max(0, x);
    }
    
    /**
     * Derivative of ReLU
     * @param x input value
     * @return relu'(x)
     */
    public static double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
    
    /**
     * Tanh (Hyperbolic Tangent) activation
     * @param x input value
     * @return tanh(x)
     */
    public static double tanh(double x) {
        return Math.tanh(x);
    }
    
    /**
     * Derivative of Tanh
     * @param x input value
     * @return tanh'(x)
     */
    public static double tanhDerivative(double x) {
        double t = Math.tanh(x);
        return 1 - t * t;
    }
    
    /**
     * Linear activation (identity): x
     * @param x input value
     * @return x
     */
    public static double linear(double x) {
        return x;
    }
    
    /**
     * Derivative of linear
     * @param x input value
     * @return 1.0
     */
    public static double linearDerivative(double x) {
        return 1.0;
    }
    
    /**
     * Softmax for multi-class classification
     * @param x input array
     * @return softmax probabilities
     */
    public static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double val : x) {
            if (val > max) max = val;
        }
        
        double[] exp = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }
        
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = exp[i] / sum;
        }
        return result;
    }
}
