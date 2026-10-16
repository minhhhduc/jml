package sklearn.preprocessing;

import numja.core.NDArray;
import numja.NumJa;

import java.util.ArrayList;
import java.util.List;

/**
 * Generate polynomial and interaction features.
 * Generate a new feature matrix consisting of all polynomial combinations
 * of the features with degree less than or equal to the specified degree.
 */
public class PolynomialFeatures {
    private int degree;
    private boolean include_bias;

    public PolynomialFeatures() { this(2, true); }
    public PolynomialFeatures(int degree) { this(degree, true); }
    public PolynomialFeatures(int degree, boolean include_bias) {
        this.degree = degree;
        this.include_bias = include_bias;
    }

    public PolynomialFeatures fit(NDArray X) {
        // Nothing to fit for polynomial features
        return this;
    }

    public NDArray transform(NDArray X) {
        int[] shape = X.getShape();
        int n = shape[0];
        int d = shape[1];

        // Only support degree 2 for simplicity
        if (degree > 2) throw new UnsupportedOperationException("Only degree 2 is currently supported");

        int out_features = (include_bias ? 1 : 0) + d;
        if (degree >= 2) {
            out_features += (d * (d + 1)) / 2;
        }

        double[][] out = new double[n][out_features];

        for (int i = 0; i < n; i++) {
            int c = 0;
            if (include_bias) out[i][c++] = 1.0;

            // degree 1
            for (int j = 0; j < d; j++) out[i][c++] = X.get(i, j);

            // degree 2
            if (degree >= 2) {
                for (int j = 0; j < d; j++) {
                    for (int k = j; k < d; k++) {
                        out[i][c++] = X.get(i, j) * X.get(i, k);
                    }
                }
            }
        }
        return NumJa.array(out);
    }

    public NDArray fit_transform(NDArray X) {
        return transform(X);
    }
}
