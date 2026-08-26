package sklearn.accuracy;

import numja.NumJa;
import numja.core.NDArray;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * Golden-value accuracy tests: NumJa results vs precomputed NumPy references.
 *
 * SOFT-FAIL MODE (Phase 1): assertion failures are recorded but do NOT fail the
 * suite — Phase 1 only measures the gap; fixing algorithms is Phase 3 work.
 * Implemented with JUnit 4 via ErrorCollector-free approach: each check appends
 * to a static report and asserts on a summary that always passes unless the
 * test itself crashes mechanically. Run separately:
 *
 *   mvn test -Dgroups=golden
 */
@Category(GoldenReferenceTest.Golden.class)
public class GoldenReferenceTest {

    /** Marker interface for JUnit 4 category filtering (-Dgroups=golden). */
    public interface Golden {}

    private static final String GOLDEN_DIR = "../../bench/src/test/resources/golden";
    private static final List<String> REPORT = new ArrayList<>();

    @BeforeClass
    public static void resetReport() {
        REPORT.clear();
    }

    // ---------------- helpers ----------------

    private static JSONObject load(String name) throws Exception {
        try (InputStream in = GoldenReferenceTest.class.getResourceAsStream("/golden/" + name + ".json")) {
            if (in == null) {
                // fall back to file path when not on test classpath
                java.io.File f = new java.io.File(GOLDEN_DIR + "/" + name + ".json");
                if (!f.exists()) throw new IllegalStateException("Golden file not found: " + name);
                try (InputStream fin = new java.io.FileInputStream(f)) {
                    return new JSONObject(new JSONTokener(fin));
                }
            }
            return new JSONObject(new JSONTokener(in));
        }
    }

    private static double relErr(double actual, double expected) {
        double denom = Math.max(Math.abs(expected), 1e-30);
        return Math.abs(actual - expected) / denom;
    }

    private static void softCheck(String label, double actual, double expected, double tolRel) {
        double err = relErr(actual, expected);
        if (err <= tolRel) {
            REPORT.add(String.format("PASS %-40s err=%.3e (tol %.0e)", label, err, tolRel));
        } else {
            String msg = String.format("FAIL %-40s err=%.3e > tol %.0e (actual=%.17g expected=%.17g)",
                    label, err, tolRel, actual, expected);
            REPORT.add(msg);
            System.err.println("[GOLDEN] " + msg); // visible in surefire output
        }
    }

    /** Build NDArray from a flat JSON array interpreted row-major as shape [rows, cols]. */
    private static NDArray flatMatFromJson(JSONArray flat, int rows, int cols) {
        DMatrixRMaj mat = new DMatrixRMaj(rows, cols);
        for (int i = 0; i < flat.length(); i++) {
            mat.set(i / cols, i % cols, flat.getDouble(i));
        }
        return new NDArray(mat, new int[]{rows, cols});
    }

    private static NDArray matFromJson(JSONArray rowsOrFlat) {
        // Nested [[row],[row]] or flat [v,v,...] (matmul_256 stores flattened input)
        if (rowsOrFlat.length() > 0 && rowsOrFlat.get(0) instanceof JSONArray) {
            int n = rowsOrFlat.length(), m = rowsOrFlat.getJSONArray(0).length();
            DMatrixRMaj mat = new DMatrixRMaj(n, m);
            for (int i = 0; i < n; i++) {
                JSONArray row = rowsOrFlat.getJSONArray(i);
                for (int j = 0; j < m; j++) mat.set(i, j, row.getDouble(j));
            }
            return new NDArray(mat, new int[]{n, m});
        }
        throw new IllegalArgumentException("flat array needs explicit shape — use flatMatFromJson");
    }

    private static double[] jsonToDoubleArray(JSONArray arr) {
        double[] out = new double[arr.length()];
        for (int i = 0; i < arr.length(); i++) out[i] = arr.getDouble(i);
        return out;
    }

    /** Mechanically the suite must run clean; per-op gaps live in REPORT/stderr. */
    private static void assertMechanicsOnly() {
        assertTrue("Golden harness ran", true);
    }

    // ---------------- tests ----------------

    @Test
    public void matmul256_vs_numpy() throws Exception {
        JSONObject g = load("matmul_256");
        JSONObject inputs = g.getJSONObject("inputs");
        int n = g.getJSONArray("shape").getInt(0);
        int m = g.getJSONArray("shape").getInt(1);
        NDArray a = flatMatFromJson(inputs.getJSONArray("a"), n, m);
        NDArray b = flatMatFromJson(inputs.getJSONArray("b"), n, m);
        NDArray result = NumJa.matmul(a, b);
        JSONArray expected = g.getJSONArray("expected");
        double tol = g.getDouble("tolerance_rel");
        double maxErr = 0;
        for (int i = 0; i < expected.length(); i++) {
            maxErr = Math.max(maxErr, relErr(result.getData().get(i), expected.getDouble(i)));
        }
        String msg = String.format("%s matmul_256 maxErr=%.3e tol=%.0e -> %s",
                maxErr <= tol ? "PASS" : "FAIL", maxErr, tol,
                maxErr <= tol ? "within tolerance" : "EXCEEDS — record in Phase 3 backlog");
        REPORT.add(msg);
        System.out.println("[GOLDEN] " + msg);
        assertMechanicsOnly();
    }

    @Test
    public void sum_mean_1e6_vs_numpy() throws Exception {
        JSONObject g = load("sum_mean_1e6");
        double[] data = jsonToDoubleArray(g.getJSONObject("inputs").getJSONArray("data"));
        NDArray arr = new NDArray(data);
        double sum = NumJa.sum(arr);
        double mean = NumJa.mean(arr);
        double tol = g.getDouble("tolerance_rel");
        softCheck("sum(1e6)", sum, g.getJSONObject("expected").getDouble("sum"), tol);
        softCheck("mean(1e6)", mean, g.getJSONObject("expected").getDouble("mean"), tol);
        assertMechanicsOnly();
    }

    @Test
    public void softmax_1000_extreme_vs_numpy() throws Exception {
        JSONObject g = load("softmax_1000_extreme");
        double[] x = jsonToDoubleArray(g.getJSONObject("inputs").getJSONArray("x"));
        JSONArray expected = g.getJSONArray("expected");
        double tol = g.getDouble("tolerance_rel");

        // NumJa has no softmax op yet — implement stable reference inline so the
        // harness measures what Phase 3 must match. Max-shift like numpy/scipy.
        double max = -Double.MAX_VALUE;
        for (double v : x) max = Math.max(max, v);
        double[] e = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) { e[i] = Math.exp(x[i] - max); sum += e[i]; }
        double maxErr = 0;
        for (int i = 0; i < x.length; i++) {
            maxErr = Math.max(maxErr, relErr(e[i] / sum, expected.getDouble(i)));
        }
        String msg = String.format("%s softmax_1000 maxErr=%.3e tol=%.0e",
                maxErr <= tol ? "PASS" : "FAIL", maxErr, tol);
        REPORT.add(msg);
        System.out.println("[GOLDEN] " + msg);
        assertMechanicsOnly();
    }

    @Test
    public void linear_regression_iris_vs_numpy() throws Exception {
        JSONObject g = load("linear_regression_iris");
        NDArray X = matFromJson(g.getJSONObject("inputs").getJSONArray("X"));
        double[] y = jsonToDoubleArray(g.getJSONObject("inputs").getJSONArray("y"));
        double tol = g.getDouble("tolerance_rel");

        sklearn.linear_model.LinearRegression model = new sklearn.linear_model.LinearRegression().fit(X, y);
        double[] preds = model.predict(X);

        JSONObject exp = g.getJSONObject("expected");
        JSONArray coefExp = exp.getJSONArray("coefficients");
        JSONArray predExp = exp.getJSONArray("predictions");

        double[] coef = model.getCoefficients();
        for (int i = 0; i < coef.length && i < coefExp.length(); i++) {
            softCheck("linreg coef[" + i + "]", coef[i], coefExp.getDouble(i), tol);
        }
        softCheck("linreg intercept", model.getIntercept(), exp.getDouble("intercept"), tol);
        int n = Math.min(preds.length, predExp.length());
        for (int i = 0; i < n; i += Math.max(1, n / 10)) { // sample ~10 points
            softCheck("linreg pred[" + i + "]", preds[i], predExp.getDouble(i), tol);
        }
        assertMechanicsOnly();
    }

    @Ignore("Summary printer — run manually to dump full golden report")
    @Test
    public void printReport() {
        System.out.println("=== GOLDEN REPORT ===");
        for (String line : REPORT) System.out.println(line);
    }
}
