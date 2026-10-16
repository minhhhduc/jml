package sklearn.cluster;

import numja.core.NDArray;
import java.util.*;

/**
 * DBSCAN - Density-Based Spatial Clustering of Applications with Noise
 * Finds core samples of high density and expands clusters from them
 */
public class DBSCAN {
    private double eps;
    private int minSamples;
    private int[] labels;
    private List<Integer> coreIndices;
    private boolean fitted = false;

    public DBSCAN() { this(0.5, 5); }
    public DBSCAN(double eps, int minSamples) {
        this.eps = eps;
        this.minSamples = minSamples;
    }

    public DBSCAN setEps(double eps) { this.eps = eps; return this; }
    public DBSCAN setMinSamples(int m) { this.minSamples = m; return this; }

    public DBSCAN fit(NDArray X) {
        double[][] data = extract2D(X);
        int n = data.length;
        labels = new int[n];
        Arrays.fill(labels, -1); // -1 = unvisited/noise
        coreIndices = new ArrayList<>();

        // Find neighbors for each point
        List<List<Integer>> neighborhoods = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> neighbors = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (euclidean(data[i], data[j]) <= eps) neighbors.add(j);
            }
            neighborhoods.add(neighbors);
            if (neighbors.size() >= minSamples) coreIndices.add(i);
        }

        int clusterId = 0;
        for (int i = 0; i < n; i++) {
            if (labels[i] != -1) continue;
            if (neighborhoods.get(i).size() < minSamples) continue; // noise

            // Expand cluster
            labels[i] = clusterId;
            Queue<Integer> queue = new LinkedList<>(neighborhoods.get(i));
            Set<Integer> visited = new HashSet<>();
            visited.add(i);

            while (!queue.isEmpty()) {
                int q = queue.poll();
                if (visited.contains(q)) continue;
                visited.add(q);

                if (labels[q] == -1) labels[q] = clusterId; // was noise, now border
                if (labels[q] != -1 && labels[q] != clusterId) continue;
                labels[q] = clusterId;

                if (neighborhoods.get(q).size() >= minSamples) {
                    for (int nb : neighborhoods.get(q)) {
                        if (!visited.contains(nb)) queue.add(nb);
                    }
                }
            }
            clusterId++;
        }

        fitted = true;
        return this;
    }

    public int[] getLabels() { check(); return labels.clone(); }
    public List<Integer> getCoreIndices() { check(); return new ArrayList<>(coreIndices); }
    public int getNumClusters() {
        check(); int max = -1;
        for (int l : labels) if (l > max) max = l;
        return max + 1;
    }

    private double euclidean(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) { double d = a[i] - b[i]; sum += d * d; }
        return Math.sqrt(sum);
    }

    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }
    private double[][] extract2D(NDArray arr) {
        int[] s = arr.getShape(); double[][] r = new double[s[0]][s[1]];
        for (int i = 0; i < s[0]; i++) for (int j = 0; j < s[1]; j++) r[i][j] = arr.get(i, j);
        return r;
    }
}
