package sklearn.preprocessing;

import numja.core.NDArray;
import numja.NumJa;
import java.util.*;

/**
 * OneHotEncoder - Encode categorical features as a one-hot numeric array
 * Converts integer labels to binary vector representation
 */
public class OneHotEncoder {
    private int n_categories;
    private boolean fitted = false;

    public OneHotEncoder() {}

    public OneHotEncoder fit(int[] labels) {
        n_categories = 0;
        for (int l : labels) if (l > n_categories) n_categories = l;
        n_categories++;
        fitted = true;
        return this;
    }

    public NDArray transform(int[] labels) {
        check();
        double[][] result = new double[labels.length][n_categories];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] < 0 || labels[i] >= n_categories)
                throw new IllegalArgumentException("Label out of range: " + labels[i]);
            result[i][labels[i]] = 1.0;
        }
        return NumJa.array(result);
    }

    public NDArray fitTransform(int[] labels) { return fit(labels).transform(labels); }

    public int[] inverseTransform(NDArray encoded) {
        check();
        double[][] data = extract2D(encoded);
        int[] result = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            int maxIdx = 0;
            for (int j = 1; j < n_categories; j++) {
                if (data[i][j] > data[i][maxIdx]) maxIdx = j;
            }
            result[i] = maxIdx;
        }
        return result;
    }

    public int getNumCategories() { return n_categories; }
    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }

    private double[][] extract2D(NDArray arr) {
        int[] s = arr.getShape(); double[][] r = new double[s[0]][s[1]];
        for (int i = 0; i < s[0]; i++) for (int j = 0; j < s[1]; j++) r[i][j] = arr.get(i, j);
        return r;
    }
}
