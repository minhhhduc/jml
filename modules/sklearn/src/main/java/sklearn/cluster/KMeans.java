package sklearn.cluster;

import numja.core.NDArray;
import numja.NumJa;
import java.util.Random;

/**
 * K-Means clustering algorithm
 * Partitions data into k clusters by minimizing within-cluster variance
 */
public class KMeans {
    private NDArray centroids;
    private int[] labels;
    private int n_clusters;
    private int max_iter;
    private double tol;
    private Random random;
    private int n_iter_;

    /**
     * Initialize KMeans with number of clusters
     * @param n_clusters number of clusters
     */
    public KMeans(int n_clusters) {
        this(n_clusters, 300, 1e-4);
    }

    /**
     * Initialize KMeans with parameters
     * @param n_clusters number of clusters
     * @param max_iter maximum iterations
     * @param tol tolerance for convergence
     */
    public KMeans(int n_clusters, int max_iter, double tol) {
        this.n_clusters = n_clusters;
        this.max_iter = max_iter;
        this.tol = tol;
        this.random = new Random();
    }

    /**
     * Fit KMeans to data
     * @param X input data (n_samples × n_features)
     * @return this KMeans object
     */
    public KMeans fit(NDArray X) {
        int[] shape = X.getShape();
        int n_samples = shape[0];
        int n_features = shape[1];

        // Extract 2D array from NDArray
        double[][] data = extract2DArray(X);

        // Single centroid mirror for the whole fit (extracted once, updated in place)
        double[][] centroidsData = new double[n_clusters][n_features];
        for (int k = 0; k < n_clusters; k++) {
            int idx = random.nextInt(n_samples);
            centroidsData[k] = data[idx].clone();
        }

        // Iterate
        this.labels = new int[n_samples];
        double[][] sums = new double[n_clusters][n_features];
        int[] counts = new int[n_clusters];
        double prev_inertia = Double.MAX_VALUE;

        for (int iteration = 0; iteration < max_iter; iteration++) {
            // Single sample pass: assign labels, accumulate pre-update inertia,
            // and per-cluster sums from the same distance computation.
            double inertia = assignAndInertia(data, centroidsData, sums, counts);

            // Compute means in place (zero row preserved for empty clusters).
            for (int k = 0; k < n_clusters; k++) {
                if (counts[k] > 0) {
                    for (int j = 0; j < n_features; j++) {
                        sums[k][j] /= counts[k];
                    }
                }
            }

            // Check convergence against pre-update inertia (bit-identical trajectory).
            if (Math.abs(prev_inertia - inertia) < tol) {
                this.n_iter_ = iteration;
                break;
            }
            prev_inertia = inertia;
            // Update mirror in place.
            for (int k = 0; k < n_clusters; k++) {
                System.arraycopy(sums[k], 0, centroidsData[k], 0, n_features);
            }
            this.n_iter_ = iteration + 1;
        }

        this.centroids = NumJa.array(centroidsData);

        return this;
    }

    /**
     * Predict cluster labels for new data
     * @param X input data
     * @return cluster labels
     */
    public int[] predict(NDArray X) {
        double[][] data = extract2DArray(X);
        int n_samples = data.length;
        int[] labels = new int[n_samples];

        double[][] centroids_data = extract2DArray(centroids);

        for (int i = 0; i < n_samples; i++) {
            double min_dist = Double.MAX_VALUE;
            int closest_centroid = 0;

            for (int k = 0; k < n_clusters; k++) {
                double dist = euclideanDistance(data[i], centroids_data[k]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = k;
                }
            }
            labels[i] = closest_centroid;
        }

        return labels;
    }

    /**
     * Get the cluster centroids
     * @return centroids as NDArray
     */
    public NDArray getCentroids() {
        return centroids;
    }

    /**
     * Get cluster labels for the fitted data
     * @return cluster labels
     */
    public int[] getLabels() {
        return labels;
    }

    /**
     * Get number of iterations until convergence
     * @return iterations
     */
    public int getNIter() {
        return n_iter_;
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

    private double assignAndInertia(double[][] data, double[][] centroidsData, double[][] sums, int[] counts) {
        for (int k = 0; k < n_clusters; k++) {
            counts[k] = 0;
            for (int j = 0; j < sums[k].length; j++) {
                sums[k][j] = 0.0;
            }
        }

        double inertia = 0;
        for (int i = 0; i < data.length; i++) {
            double min_dist = Double.MAX_VALUE;
            int closest = 0;

            for (int k = 0; k < n_clusters; k++) {
                double dist = euclideanDistance(data[i], centroidsData[k]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = k;
                }
            }
            labels[i] = closest;
            inertia += min_dist * min_dist;
            for (int j = 0; j < data[i].length; j++) {
                sums[closest][j] += data[i][j];
            }
            counts[closest]++;
        }

        return inertia;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
