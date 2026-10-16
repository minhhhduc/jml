package sklearn;

import numja.NumJa;
import numja.core.NDArray;
import sklearn.cluster.KMeans;
import sklearn.cluster.GaussianMixture;
import sklearn.preprocessing.StandardScaler;
import sklearn.neural_network.Activations;
import sklearn.neural_network.MLPClassifier;
import sklearn.neural_network.MLPRegressor;
import sklearn.neural_network.Autoencoder;
import sklearn.decomposition.PCA;
import sklearn.linear_model.LinearRegression;
import sklearn.metrics.Metrics;
import sklearn.model_selection.ModelSelection;

/**
 * Test cases for sklearn module
 */
public class TestSklearn {

    public static void main(String[] args) {
        System.out.println("=== SKLearn Module Tests ===\n");

        testActivations();
        System.out.println();
        testKMeans();
        System.out.println();
        testGaussianMixture();
        System.out.println();
        testStandardScaler();
        System.out.println();
        testMLP();
        System.out.println();
        testMLPRegressor();
        System.out.println();
        testAutoencoder();
        System.out.println();
        testPCA();
        System.out.println();
        testLinearRegression();
        System.out.println();
        testMetrics();
        System.out.println();
        testTrainTestSplit();
    }

    private static void testActivations() {
        System.out.println("--- Activation Functions ---");
        double x = 1.0;
        System.out.println("sigmoid(" + x + ") = " + Activations.sigmoid(x));
        System.out.println("relu(" + x + ") = " + Activations.relu(x));
        System.out.println("tanh(" + x + ") = " + Activations.tanh(x));
        
        double[] test = {1.0, 2.0, 3.0};
        double[] softmax = Activations.softmax(test);
        System.out.println("softmax = " + java.util.Arrays.toString(softmax));
    }

    private static void testKMeans() {
        System.out.println("--- KMeans Clustering ---");

        // Create sample data: 3 clusters
        double[][] data = {
            {1.0, 1.0}, {1.2, 1.1}, {0.9, 1.0},
            {5.0, 5.0}, {5.1, 5.0}, {5.0, 5.1},
            {10.0, 1.0}, {10.1, 1.0}, {10.0, 1.1}
        };

        NDArray X = NumJa.array(data);

        // Fit KMeans
        KMeans kmeans = new KMeans(3);
        kmeans.fit(X);

        int[] labels = kmeans.getLabels();
        System.out.println("Cluster labels: " + java.util.Arrays.toString(labels));
        System.out.println("Iterations: " + kmeans.getNIter());
    }

    private static void testGaussianMixture() {
        System.out.println("--- Gaussian Mixture Model ---");

        // Create sample data
        double[][] data = {
            {1.0, 1.0}, {1.2, 1.1}, {0.9, 1.0},
            {5.0, 5.0}, {5.1, 5.0}, {5.0, 5.1},
            {10.0, 1.0}, {10.1, 1.0}, {10.0, 1.1}
        };

        NDArray X = NumJa.array(data);

        // Fit GMM
        GaussianMixture gmm = new GaussianMixture(3);
        gmm.fit(X);

        int[] labels = gmm.predict(X);
        double[] weights = gmm.getWeights();

        System.out.println("Cluster labels: " + java.util.Arrays.toString(labels));
        System.out.println("Mixing weights: " + java.util.Arrays.toString(weights));
        System.out.println("Iterations: " + gmm.getNIter());
    }

    private static void testStandardScaler() {
        System.out.println("--- StandardScaler Preprocessing ---");

        // Create sample data
        double[][] data = {
            {1.0, 100.0},
            {2.0, 200.0},
            {3.0, 300.0},
            {4.0, 400.0},
            {5.0, 500.0}
        };

        NDArray X = NumJa.array(data);

        // Fit and transform
        StandardScaler scaler = new StandardScaler();
        NDArray X_scaled = scaler.fitTransform(X);

        System.out.println("Original shape: " + java.util.Arrays.toString(X.getShape()));
        System.out.println("Scaled shape: " + java.util.Arrays.toString(X_scaled.getShape()));
        System.out.println("Mean after scaling: " + java.util.Arrays.toString(scaler.getMean()));
    }

    private static void testMLP() {
        System.out.println("--- Multi-Layer Perceptron (Neural Network) ---");

        // Simple classification dataset
        double[][] X_train = {
            {0.0, 0.0}, {0.1, 0.1}, {1.0, 1.0}, {0.9, 1.1},
            {0.0, 1.0}, {0.1, 0.9}, {1.0, 0.0}, {0.9, 0.1}
        };
        int[] y_train = {0, 0, 1, 1, 2, 2, 3, 3};

        NDArray X = NumJa.array(X_train);

        // Create and train MLP
        MLPClassifier mlp = new MLPClassifier(32, 16)
            .setActivation("relu")
            .setLearningRate(0.01)
            .setMaxIter(5);

        System.out.println("Training MLP...");
        mlp.fit(X, y_train);

        int[] predictions = mlp.predict(X);
        System.out.println("Predictions: " + java.util.Arrays.toString(predictions));
    }

    private static void testAutoencoder() {
        System.out.println("--- Autoencoder (Unsupervised Learning) ---");

        // Create sample data
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0},
            {1.1, 2.1, 3.1, 4.1},
            {5.0, 6.0, 7.0, 8.0},
            {5.1, 6.1, 7.1, 8.1}
        };

        NDArray X = NumJa.array(data);

        // Create and train autoencoder
        Autoencoder ae = new Autoencoder(4, 2, 32)
            .setLearningRate(0.001)
            .setMaxIter(5);

        System.out.println("Training Autoencoder...");
        ae.fit(X);

        // Encode to latent space
        NDArray encoded = ae.transform(X);
        System.out.println("Encoded shape: " + java.util.Arrays.toString(encoded.getShape()));

        // Reconstruct
        NDArray reconstructed = ae.inverseTransform(encoded);
        System.out.println("Reconstructed shape: " + java.util.Arrays.toString(reconstructed.getShape()));
    }

    private static void testPCA() {
        System.out.println("--- Principal Component Analysis (PCA) ---");

        // Create sample data with correlation
        double[][] data = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0},
            {4.0, 8.0, 12.0},
            {5.0, 10.0, 15.0}
        };

        NDArray X = NumJa.array(data);

        // Fit PCA
        PCA pca = new PCA(2);
        NDArray X_reduced = pca.fitTransform(X);

        System.out.println("Original shape: " + java.util.Arrays.toString(X.getShape()));
        System.out.println("Reduced shape: " + java.util.Arrays.toString(X_reduced.getShape()));
        System.out.println("Explained variance ratio: " + 
            java.util.Arrays.toString(pca.getExplainedVarianceRatio()));

        // Reconstruct
        NDArray X_reconstructed = pca.inverseTransform(X_reduced);
        System.out.println("Reconstructed shape: " + java.util.Arrays.toString(X_reconstructed.getShape()));
    }

    private static void testMLPRegressor() {
        System.out.println("--- MLP Regressor (Neural Network for Regression) ---");

        // Create sample regression data
        double[][] X_train = {
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        };
        double[] y_train = {2.0, 4.0, 6.0, 8.0, 10.0};

        NDArray X = NumJa.array(X_train);

        // Create and train regressor
        MLPRegressor mlp = new MLPRegressor(16)
            .setActivation("relu")
            .setLearningRate(0.01)
            .setMaxIter(5);

        System.out.println("Training MLPRegressor...");
        mlp.fit(X, y_train);

        double[] predictions = mlp.predict(X);
        System.out.println("Predictions: " + java.util.Arrays.toString(predictions));
    }

    private static void testLinearRegression() {
        System.out.println("--- Linear Regression ---");

        // Create sample data
        double[][] X_train = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0},
            {5.0, 6.0}
        };
        double[] y_train = {3.0, 5.0, 7.0, 9.0, 11.0};

        NDArray X = NumJa.array(X_train);

        // Fit linear regression
        LinearRegression lr = new LinearRegression();
        lr.fit(X, y_train);

        double[] predictions = lr.predict(X);
        System.out.println("Predictions: " + java.util.Arrays.toString(predictions));
        System.out.println("Coefficients: " + java.util.Arrays.toString(lr.getCoefficients()));
        System.out.println("Intercept: " + lr.getIntercept());
        System.out.println("R² Score: " + lr.score(X, y_train));
    }

    private static void testMetrics() {
        System.out.println("--- Model Evaluation Metrics ---");

        // Binary classification metrics
        int[] y_true = {0, 0, 1, 1, 1, 0, 1, 1, 0, 1};
        int[] y_pred = {0, 1, 1, 1, 1, 0, 1, 0, 0, 1};

        System.out.println("Accuracy: " + Metrics.accuracyScore(y_true, y_pred));
        System.out.println("Precision: " + Metrics.precisionScore(y_true, y_pred));
        System.out.println("Recall: " + Metrics.recallScore(y_true, y_pred));
        System.out.println("F1 Score: " + Metrics.f1Score(y_true, y_pred));

        Metrics.printConfusionMatrix(Metrics.confusionMatrix(y_true, y_pred));

        // Regression metrics
        double[] y_true_reg = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] y_pred_reg = {1.1, 2.1, 2.9, 4.0, 4.9};

        System.out.println("\nRegression Metrics:");
        System.out.println("MSE: " + Metrics.meanSquaredError(y_true_reg, y_pred_reg));
        System.out.println("RMSE: " + Metrics.rmse(y_true_reg, y_pred_reg));
        System.out.println("MAE: " + Metrics.meanAbsoluteError(y_true_reg, y_pred_reg));
        System.out.println("R² Score: " + Metrics.r2Score(y_true_reg, y_pred_reg));
    }

    private static void testTrainTestSplit() {
        System.out.println("--- Train Test Split ---");

        // Create sample data
        double[][] X_data = {
            {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0},
            {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}, {8.0, 8.0}
        };
        int[] y = {0, 0, 0, 0, 1, 1, 1, 1};

        NDArray X = NumJa.array(X_data);

        // Split 80/20
        Object[] split = ModelSelection.trainTestSplit(X, y, 0.2, 42);
        NDArray X_train = (NDArray) split[0];
        NDArray X_test = (NDArray) split[1];
        int[] y_train = (int[]) split[2];
        int[] y_test = (int[]) split[3];

        System.out.println("Train shape: " + java.util.Arrays.toString(X_train.getShape()));
        System.out.println("Test shape: " + java.util.Arrays.toString(X_test.getShape()));
        System.out.println("Train labels: " + java.util.Arrays.toString(y_train));
        System.out.println("Test labels: " + java.util.Arrays.toString(y_test));
    }
}
