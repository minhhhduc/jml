package sklearn.accuracy;

import numja.NumJa;
import numja.core.NDArray;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Golden-value accuracy tests: NumJa results vs precomputed NumPy references.
 *
 * SOFT-FAIL MODE (Phase 1): assertion failures are recorded but do NOT fail the
 * suite — Phase 1 only measures the gap; fixing algorithms is Phase 3 work.
 * Each check appends a PASS/FAIL line to a report that is flushed to
 * target/golden-report.txt after every test (survives the JVM) and echoed to
 * stdout. The suite only fails if the harness itself crashes mechanically.
 *
 * Run: mvn -pl modules/sklearn -am test -Dtest=GoldenReferenceTest
 *
 * Inputs are regenerated from each file's "seed" via java.util.Random —
 * identical to what scripts/golden/generate_golden.py used (Java-compatible
 * LCG), so golden files store expected values only, not raw inputs.
 */
@Category(GoldenReferenceTest.Golden.class)
public class GoldenReferenceTest {

    /** Marker interface for JUnit 4 category filtering. */
    public interface Golden {}

    private static final String GOLDEN_DIR = "../../bench/src/test/resources/golden";
    private static final List<String> REPORT = new ArrayList<>();

    @BeforeClass
    public static void resetReport() {
        REPORT.clear();
        REPORT.add("Golden accuracy report (NumJa vs NumPy), soft-fail mode");
    }

    // ---------------- helpers ----------------

    private static JSONObject load(String name) throws Exception {
        try (InputStream in = GoldenReferenceTest.class.getResourceAsStream("/golden/" + name + ".json")) {
            if (in != null) return new JSONObject(new JSONTokener(in));
        }
        // golden files live in bench/src/test/resources, not on this module's test classpath
        File f = new File(GOLDEN_DIR + "/" + name + ".json");
        if (!f.exists()) throw new IllegalStateException("Golden file not found: " + name);
        try (InputStream fin = new FileInputStream(f)) {
            return new JSONObject(new JSONTokener(fin));
        }
    }

    private static double relErr(double actual, double expected) {
        double denom = Math.max(Math.abs(expected), 1e-30);
        return Math.abs(actual - expected) / denom;
    }

    /** Record a PASS/FAIL line; never throws (soft-fail). */
    private static void check(String label, double actual, double expected, double tolRel) {
        double err = relErr(actual, expected);
        boolean pass = err <= tolRel;
        String msg = String.format("%s %-40s err=%.3e (tol %.0e)", pass ? "PASS" : "FAIL", label, err, tolRel);
        if (!pass) {
            msg += String.format(" actual=%.17g expected=%.17g", actual, expected);
        }
        REPORT.add(msg);
        System.out.println("[GOLDEN] " + msg);
    }

    /** Same as {@link #check} for pre-aggregated max-error values. */
    private static void checkMaxErr(String label, double maxErr, double tolRel) {
        boolean pass = maxErr <= tolRel;
        String msg = String.format("%s %-40s maxErr=%.3e (tol %.0e)", pass ? "PASS" : "FAIL", label, maxErr, tolRel);
        REPORT.add(msg);
        System.out.println("[GOLDEN] " + msg);
    }

    private static void flushReport() {
        try (PrintWriter w = new PrintWriter("target/golden-report.txt", "UTF-8")) {
            for (String line : REPORT) w.println(line);
        } catch (Exception e) {
            System.err.println("[GOLDEN] could not write report: " + e.getMessage());
        }
    }

    private static void assertMechanicsOnly() {
        flushReport();
        assertTrue(true); // soft-fail: per-op gaps live in REPORT/stderr/target/golden-report.txt
    }

    /** Regenerate inputs exactly as generate_golden.py did (java.util.Random LCG). */
    private static Random seeded(long seed) {
        return new Random(seed);
    }

    /** Read the 4 numeric columns of dist/datasets/iris.csv as [rows][4]. */
    private static double[][] readIrisFeatures() throws Exception {
        File f = new File("../../dist/datasets/iris.csv");
        if (!f.exists()) throw new IllegalStateException("iris.csv not found: " + f.getPath());
        List<double[]> rows = new ArrayList<>();
        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(f))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                rows.add(new double[]{
                        Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                        Double.parseDouble(parts[2]), Double.parseDouble(parts[3])});
            }
        }
        return rows.toArray(new double[0][]);
    }

    private static double[] jsonToDoubleArray(JSONArray arr) {
        double[] out = new double[arr.length()];
        for (int i = 0; i < arr.length(); i++) out[i] = arr.getDouble(i);
        return out;
    }

    // ---------------- tests ----------------

    @Test
    public void matmul256_vs_numpy() throws Exception {
        JSONObject g = load("matmul_256");
        long seed = g.getLong("seed");
        int n = g.getJSONArray("shape").getInt(0);
        Random rng = seeded(seed);
        // draw order must match generate_golden.py: ALL of 'a' first, then ALL of 'b'
        double[][] av = new double[n][n], bv = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) av[i][j] = rng.nextDouble();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) bv[i][j] = rng.nextDouble();
        NDArray result = NumJa.matmul(new NDArray(av), new NDArray(bv));
        JSONArray expected = g.getJSONArray("expected");
        double tol = g.getDouble("tolerance_rel");
        double maxErr = 0;
        for (int i = 0; i < expected.length(); i++) {
            maxErr = Math.max(maxErr, relErr(result.getData().get(i), expected.getDouble(i)));
        }
        checkMaxErr("matmul_256 maxErr", maxErr, tol);
        assertMechanicsOnly();
    }

    @Test
    public void sum_mean_vs_numpy() throws Exception {
        JSONObject g = load("sum_mean");
        long seed = g.getLong("seed");
        Random rng = seeded(seed);
        double[] data = new double[1_000_000];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextDouble() * 2.0 - 1.0;
        NDArray arr = new NDArray(data);
        double tol = g.getDouble("tolerance_rel");
        check("sum(1e6)", NumJa.sum(arr), g.getJSONObject("expected").getDouble("sum"), tol);
        check("mean(1e6)", NumJa.mean(arr), g.getJSONObject("expected").getDouble("mean"), tol);
        assertMechanicsOnly();
    }

    /**
     * REFERENCE-ONLY CHECK: NumJa has no softmax op yet, so this diffs an inline
     * stable implementation against NumPy to validate the harness + tolerance.
     * It is NOT a NumJa accuracy measurement — when numja gains a softmax op,
     * switch this test to call it.
     */
    @Test
    public void softmax_extreme_vs_numpy() throws Exception {
        JSONObject g = load("softmax_extreme");
        long seed = g.getLong("seed");
        Random rng = seeded(seed);
        double[] x = new double[1000];
        for (int i = 0; i < x.length; i++) x[i] = rng.nextDouble() * 1400.0 - 700.0;
        JSONArray expected = g.getJSONArray("expected");

        double max = -Double.MAX_VALUE;
        for (double v : x) max = Math.max(max, v);
        double[] e = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) { e[i] = Math.exp(x[i] - max); sum += e[i]; }
        double maxErr = 0;
        for (int i = 0; i < x.length; i++) {
            maxErr = Math.max(maxErr, relErr(e[i] / sum, expected.getDouble(i)));
        }
        checkMaxErr("softmax_1000 [reference-only] maxErr", maxErr, tol(g));
        assertMechanicsOnly();
    }

    @Test
    public void linear_regression_iris_vs_numpy() throws Exception {
        JSONObject g = load("linear_regression_iris");
        // inputs come from dist/datasets/iris.csv (see "input_source" in the golden file)
        double[][] raw = readIrisFeatures();
        int n = raw.length;
        double[][] xv = new double[n][3];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(raw[i], 0, xv[i], 0, 3);
            y[i] = raw[i][3];
        }
        NDArray X = new NDArray(xv);
        double tol = g.getDouble("tolerance_rel");

        sklearn.linear_model.LinearRegression model = new sklearn.linear_model.LinearRegression().fit(X, y);
        double[] preds = model.predict(X);

        JSONObject exp = g.getJSONObject("expected");
        JSONArray coefExp = exp.getJSONArray("coefficients");
        JSONArray predExp = exp.getJSONArray("predictions");

        double[] coef = model.getCoefficients();
        for (int i = 0; i < coef.length && i < coefExp.length(); i++) {
            check("linreg coef[" + i + "]", coef[i], coefExp.getDouble(i), tol);
        }
        check("linreg intercept", model.getIntercept(), exp.getDouble("intercept"), tol);
        int np = Math.min(preds.length, predExp.length());
        for (int i = 0; i < np; i += Math.max(1, np / 10)) { // sample ~10 points
            check("linreg pred[" + i + "]", preds[i], predExp.getDouble(i), tol);
        }
        assertMechanicsOnly();
    }

    private static double tol(JSONObject g) {
        return g.getDouble("tolerance_rel");
    }

    @Ignore("Summary printer — run manually to dump full golden report")
    @Test
    public void printReport() {
        System.out.println("=== GOLDEN REPORT ===");
        for (String line : REPORT) System.out.println(line);
    }
}
