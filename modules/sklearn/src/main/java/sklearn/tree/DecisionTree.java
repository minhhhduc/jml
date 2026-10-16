package sklearn.tree;

import numja.core.NDArray;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Decision Tree - Unified Classifier and Regressor
 * CART algorithm with Gini impurity (classification) or MSE (regression)
 */
public class DecisionTree {
    private int max_depth;
    private int min_samples_split;
    private String mode; // "classifier" or "regressor"
    private Node root;
    private int n_features;
    private boolean fitted = false;

    private static class Node {
        int featureIndex = -1;
        double threshold;
        double value;       // prediction value (class label or mean)
        int classLabel;     // for classifier
        Node left, right;
        boolean isLeaf = false;
    }

    public DecisionTree() { this(10, 2, "classifier"); }

    public DecisionTree(int max_depth, int min_samples_split, String mode) {
        this.max_depth = max_depth;
        this.min_samples_split = min_samples_split;
        this.mode = mode;
    }

    /** Fit for classification */
    public DecisionTree fit(NDArray X, int[] y) {
        double[][] data = extract2D(X);
        n_features = data[0].length;
        double[] yDouble = new double[y.length];
        for (int i = 0; i < y.length; i++) yDouble[i] = y[i];
        this.mode = "classifier";
        root = buildTree(data, yDouble, 0);
        fitted = true;
        return this;
    }

    /** Fit for regression */
    public DecisionTree fit(NDArray X, double[] y) {
        double[][] data = extract2D(X);
        n_features = data[0].length;
        this.mode = "regressor";
        root = buildTree(data, y, 0);
        fitted = true;
        return this;
    }

    /** Predict class labels (classifier) */
    public int[] predict(NDArray X) {
        check(); double[][] data = extract2D(X);
        int[] pred = new int[data.length];
        for (int i = 0; i < data.length; i++) pred[i] = (int) predictSample(root, data[i]);
        return pred;
    }

    /** Predict continuous values (regressor) */
    public double[] predictRegression(NDArray X) {
        check(); double[][] data = extract2D(X);
        double[] pred = new double[data.length];
        for (int i = 0; i < data.length; i++) pred[i] = predictSample(root, data[i]);
        return pred;
    }

    public double score(NDArray X, int[] y) {
        int[] p = predict(X); int ok = 0;
        for (int i = 0; i < y.length; i++) if (p[i] == y[i]) ok++;
        return (double) ok / y.length;
    }

    public double score(NDArray X, double[] y) {
        double[] p = predictRegression(X);
        double my = 0; for (double v : y) my += v; my /= y.length;
        double sst = 0, ssr = 0;
        for (int i = 0; i < y.length; i++) { sst += (y[i]-my)*(y[i]-my); ssr += (y[i]-p[i])*(y[i]-p[i]); }
        return 1.0 - ssr / sst;
    }

    private double predictSample(Node node, double[] x) {
        if (node.isLeaf) return mode.equals("classifier") ? node.classLabel : node.value;
        if (x[node.featureIndex] <= node.threshold) return predictSample(node.left, x);
        return predictSample(node.right, x);
    }

    private Node buildTree(double[][] X, double[] y, int depth) {
        Node node = new Node();
        int n = X.length;

        // Leaf conditions
        if (depth >= max_depth || n < min_samples_split || allSame(y)) {
            node.isLeaf = true;
            if (mode.equals("classifier")) {
                node.classLabel = majorityClass(y);
                node.value = node.classLabel;
            } else {
                double sum = 0; for (double v : y) sum += v;
                node.value = sum / n;
            }
            return node;
        }

        // Find best split
        double bestScore = Double.MAX_VALUE;
        int bestFeature = 0;
        double bestThreshold = 0;

        for (int f = 0; f < n_features; f++) {
            // Get unique sorted values for thresholds
            double[] vals = new double[n];
            for (int i = 0; i < n; i++) vals[i] = X[i][f];
            Arrays.sort(vals);

            for (int t = 0; t < n - 1; t++) {
                if (vals[t] == vals[t + 1]) continue;
                double threshold = (vals[t] + vals[t + 1]) / 2.0;
                double score = computeSplitScore(X, y, f, threshold);
                if (score < bestScore) {
                    bestScore = score;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }

        // If no good split found, make leaf
        if (bestScore == Double.MAX_VALUE) {
            node.isLeaf = true;
            if (mode.equals("classifier")) { node.classLabel = majorityClass(y); node.value = node.classLabel; }
            else { double s = 0; for (double v : y) s += v; node.value = s / n; }
            return node;
        }

        node.featureIndex = bestFeature;
        node.threshold = bestThreshold;

        // Split data
        int leftCount = 0;
        for (int i = 0; i < n; i++) if (X[i][bestFeature] <= bestThreshold) leftCount++;

        double[][] leftX = new double[leftCount][n_features];
        double[] leftY = new double[leftCount];
        double[][] rightX = new double[n - leftCount][n_features];
        double[] rightY = new double[n - leftCount];
        int li = 0, ri = 0;

        for (int i = 0; i < n; i++) {
            if (X[i][bestFeature] <= bestThreshold) {
                leftX[li] = X[i]; leftY[li] = y[i]; li++;
            } else {
                rightX[ri] = X[i]; rightY[ri] = y[i]; ri++;
            }
        }

        node.left = buildTree(leftX, leftY, depth + 1);
        node.right = buildTree(rightX, rightY, depth + 1);
        return node;
    }

    private double computeSplitScore(double[][] X, double[] y, int feature, double threshold) {
        int n = X.length, leftN = 0, rightN = 0;
        double leftScore = 0, rightScore = 0;

        // Count splits
        for (int i = 0; i < n; i++) {
            if (X[i][feature] <= threshold) leftN++; else rightN++;
        }
        if (leftN == 0 || rightN == 0) return Double.MAX_VALUE;

        double[] leftY = new double[leftN], rightY = new double[rightN];
        int li = 0, ri = 0;
        for (int i = 0; i < n; i++) {
            if (X[i][feature] <= threshold) leftY[li++] = y[i]; else rightY[ri++] = y[i];
        }

        if (mode.equals("classifier")) {
            leftScore = giniImpurity(leftY);
            rightScore = giniImpurity(rightY);
        } else {
            leftScore = mse(leftY);
            rightScore = mse(rightY);
        }

        return (leftN * leftScore + rightN * rightScore) / n;
    }

    private double giniImpurity(double[] y) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (double v : y) counts.merge((int) v, 1, Integer::sum);
        double gini = 1.0;
        for (int c : counts.values()) { double p = (double) c / y.length; gini -= p * p; }
        return gini;
    }

    private double mse(double[] y) {
        double mean = 0; for (double v : y) mean += v; mean /= y.length;
        double sse = 0; for (double v : y) sse += (v - mean) * (v - mean);
        return sse / y.length;
    }

    private int majorityClass(double[] y) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (double v : y) counts.merge((int) v, 1, Integer::sum);
        int best = 0, bestCount = -1;
        for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
            if (e.getValue() > bestCount) { bestCount = e.getValue(); best = e.getKey(); }
        }
        return best;
    }

    private boolean allSame(double[] y) {
        for (int i = 1; i < y.length; i++) if (y[i] != y[0]) return false;
        return true;
    }

    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }

    private double[][] extract2D(NDArray arr) {
        int[] s = arr.getShape(); double[][] r = new double[s[0]][s[1]];
        for (int i = 0; i < s[0]; i++) for (int j = 0; j < s[1]; j++) r[i][j] = arr.get(i, j);
        return r;
    }
}
