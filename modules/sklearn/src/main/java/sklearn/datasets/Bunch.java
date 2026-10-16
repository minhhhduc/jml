package sklearn.datasets;

import numja.core.NDArray;

/**
 * A container object exposing keys as attributes, matching Scikit-Learn's Bunch.
 */
public class Bunch {
    private final NDArray data;
    private final int[] target;
    private final double[] targetRegression;
    private final String[] featureNames;
    private final String[] targetNames;

    public Bunch(NDArray data, int[] target, String[] featureNames, String[] targetNames) {
        this.data = data;
        this.target = target;
        this.targetRegression = null;
        this.featureNames = featureNames;
        this.targetNames = targetNames;
    }

    public Bunch(NDArray data, double[] targetRegression, String[] featureNames) {
        this.data = data;
        this.target = null;
        this.targetRegression = targetRegression;
        this.featureNames = featureNames;
        this.targetNames = null;
    }

    public NDArray getData() {
        return data;
    }

    public int[] getTarget() {
        if (target == null) throw new IllegalStateException("This Bunch contains regression targets.");
        return target;
    }

    public double[] getTargetRegression() {
        if (targetRegression == null) throw new IllegalStateException("This Bunch contains classification targets.");
        return targetRegression;
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    public String[] getTargetNames() {
        return targetNames;
    }
}
