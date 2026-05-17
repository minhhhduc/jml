package sklearn.datasets;

import numja.core.NDArray;
import numja.NumJa;
import pandas.Pandas;
import pandas.DataFrame;
import sklearn.preprocessing.LabelEncoder;
import java.io.File;
import java.util.Random;

/**
 * Utility functions to load standard datasets or generate synthetic data.
 */
public class Datasets {

    /**
     * Load and return the Iris dataset (classification).
     */
    public static Bunch loadIris() {
        File file = new File("dist/datasets/iris.csv");
        if (!file.exists()) {
            file = new File("datasets/iris.csv");
        }
        if (!file.exists()) {
            file = new File("../dist/datasets/iris.csv");
        }
        if (!file.exists()) {
            file = new File("../datasets/iris.csv");
        }
        if (!file.exists()) {
            file = new File("../../dist/datasets/iris.csv");
        }
        if (!file.exists()) {
            file = new File("../../datasets/iris.csv");
        }
        if (!file.exists()) {
            throw new RuntimeException("Iris dataset (iris.csv) could not be located in standard paths.");
        }

        DataFrame df;
        try {
            df = Pandas.read_csv(file.getAbsolutePath(), new Pandas.ReadCsvOptions().header(0));
        } catch (java.io.IOException e) {
            throw new RuntimeException("Error reading Iris dataset file: " + e.getMessage(), e);
        }
        double[][] xArr = df.drop("species").toArray();
        NDArray X = NumJa.array(xArr);

        LabelEncoder le = new LabelEncoder();
        String[] species = new String[df.shape()[0]];
        for (int i = 0; i < species.length; i++) {
            species[i] = df.getColumn("species").getString(i);
        }
        le.fit(species);
        int[] y = le.transform(species);

        String[] featureNames = {"sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"};
        String[] targetNames = {"setosa", "versicolor", "virginica"};

        return new Bunch(X, y, featureNames, targetNames);
    }

    /**
     * Generate isotropic Gaussian blobs for clustering.
     */
    public static Bunch makeBlobs(int nSamples, int nFeatures, int centers, double clusterStd, int randomState) {
        Random r = new Random(randomState);
        double[][] centersCoords = new double[centers][nFeatures];
        for (int i = 0; i < centers; i++) {
            for (int j = 0; j < nFeatures; j++) {
                centersCoords[i][j] = (r.nextDouble() - 0.5) * 20.0; // Random center coordinates
            }
        }

        double[][] data = new double[nSamples][nFeatures];
        int[] target = new int[nSamples];

        for (int i = 0; i < nSamples; i++) {
            int centerIdx = r.nextInt(centers);
            target[i] = centerIdx;
            for (int j = 0; j < nFeatures; j++) {
                double gaussianNoise = r.nextGaussian() * clusterStd;
                data[i][j] = centersCoords[centerIdx][j] + gaussianNoise;
            }
        }

        String[] featureNames = new String[nFeatures];
        for (int j = 0; j < nFeatures; j++) featureNames[j] = "feature_" + j;

        String[] targetNames = new String[centers];
        for (int i = 0; i < centers; i++) targetNames[i] = "blob_" + i;

        return new Bunch(NumJa.array(data), target, featureNames, targetNames);
    }

    /**
     * Generate a random regression problem.
     */
    public static Bunch makeRegression(int nSamples, int nFeatures, double noise, int randomState) {
        Random r = new Random(randomState);
        double[] coef = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            coef[j] = (r.nextDouble() - 0.5) * 100.0; // Ground truth weights
        }

        double[][] X = new double[nSamples][nFeatures];
        double[] y = new double[nSamples];

        for (int i = 0; i < nSamples; i++) {
            double sum = 0;
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = r.nextGaussian();
                sum += X[i][j] * coef[j];
            }
            double noiseVal = r.nextGaussian() * noise;
            y[i] = sum + noiseVal;
        }

        String[] featureNames = new String[nFeatures];
        for (int j = 0; j < nFeatures; j++) featureNames[j] = "feature_" + j;

        return new Bunch(NumJa.array(X), y, featureNames);
    }
}
