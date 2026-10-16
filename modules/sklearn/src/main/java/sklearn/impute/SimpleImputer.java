package sklearn.impute;

import numja.core.NDArray;
import numja.NumJa;

import java.util.HashMap;
import java.util.Map;

/**
 * Imputation transformer for completing missing values.
 */
public class SimpleImputer {
    private String strategy;
    private double missing_values;
    private double[] statistics_; // The imputation fill value for each feature
    private boolean fitted = false;

    public SimpleImputer() { this("mean", Double.NaN); }
    public SimpleImputer(String strategy) { this(strategy, Double.NaN); }
    public SimpleImputer(String strategy, double missing_values) {
        this.strategy = strategy;
        this.missing_values = missing_values;
    }

    public SimpleImputer fit(NDArray X) {
        int[] shape = X.getShape();
        int n = shape[0];
        int d = shape[1];
        
        statistics_ = new double[d];
        for (int j = 0; j < d; j++) {
            if (strategy.equals("mean")) {
                double sum = 0;
                int count = 0;
                for (int i = 0; i < n; i++) {
                    double v = X.get(i, j);
                    if (!isMissing(v)) { sum += v; count++; }
                }
                statistics_[j] = (count > 0) ? sum / count : 0.0;
            } else if (strategy.equals("most_frequent")) {
                Map<Double, Integer> counts = new HashMap<>();
                for (int i = 0; i < n; i++) {
                    double v = X.get(i, j);
                    if (!isMissing(v)) counts.put(v, counts.getOrDefault(v, 0) + 1);
                }
                double mostFreq = 0.0;
                int maxCount = -1;
                for (Map.Entry<Double, Integer> e : counts.entrySet()) {
                    if (e.getValue() > maxCount) { maxCount = e.getValue(); mostFreq = e.getKey(); }
                }
                statistics_[j] = mostFreq;
            } else if (strategy.equals("constant")) {
                statistics_[j] = 0.0; // Default constant
            } else {
                throw new IllegalArgumentException("Unsupported strategy: " + strategy);
            }
        }
        fitted = true;
        return this;
    }

    public NDArray transform(NDArray X) {
        if (!fitted) throw new IllegalStateException("Not fitted");
        int[] shape = X.getShape();
        int n = shape[0];
        int d = shape[1];

        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double v = X.get(i, j);
                if (isMissing(v)) {
                    out[i][j] = statistics_[j];
                } else {
                    out[i][j] = v;
                }
            }
        }
        return NumJa.array(out);
    }

    public NDArray fit_transform(NDArray X) {
        return fit(X).transform(X);
    }

    private boolean isMissing(double v) {
        if (Double.isNaN(missing_values)) return Double.isNaN(v);
        return v == missing_values;
    }
}
