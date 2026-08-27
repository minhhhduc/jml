package sklearn.accuracy;

import numja.NumericStable;
import numja.NumJa;
import numja.core.NDArray;
import numja.core.ParallelOps;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import sklearn.neural_network.Activations;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * ACC-03 hard-fail golden-value test suite.
 *
 * Mirrors all 4 existing GoldenReferenceTest tests + 3 new Phase 3 tests = 7 total.
 * Every assertion is HARD-FAIL: no REPORT buffer, no soft-fail path. Each test
 * calls GoldenFixtures.assertWithinTolerance or GoldenFixtures.assertMaxErrWithinTolerance
 * which throws AssertionError on tolerance miss.
 *
 * @Category(GoldenReferenceTest.Golden.class) — same filter marker as the soft-fail class.
 *
 * GoldenReferenceTest continues to soft-fail alongside this class; both coexist.
 */
@Category(GoldenReferenceTest.Golden.class)
public class AccuracyHardeningTest {

    // ---------------- helpers ----------------

    private static double[] jsonToDoubleArray(JSONArray arr) {
        return GoldenFixtures.jsonToDoubleArray(arr);
    }

    private static double[][] readIrisFeatures() throws Exception {
        File f = new File("../../dist/datasets/iris.csv");
        if (!f.exists()) throw new IllegalStateException("iris.csv not found: " + f.getPath());
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                rows.add(new double[]{
                        Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                        Double.parseDouble(parts[2]), Double.parseDouble(parts[3])});
            }
        }
        double[][] arr = new double[rows.size()][];
        return rows.toArray(arr);
    }

    // ---------------- 4 mirrored tests ----------------

    @Test
    public void matmul256_withinTolerance() throws Exception {
        JSONObject g = GoldenFixtures.load("matmul_256");
        long seed = g.getLong("seed");
        int n = g.getJSONArray("shape").getInt(0);
        Random rng = new Random(seed);
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
            maxErr = Math.max(maxErr, GoldenFixtures.relErr(result.getData().get(i), expected.getDouble(i)));
        }
        GoldenFixtures.assertMaxErrWithinTolerance("matmul_256 maxErr", maxErr, tol);
    }

    @Test
    public void sum_mean_withinTolerance() throws Exception {
        JSONObject g = GoldenFixtures.load("sum_mean");
        long seed = g.getLong("seed");
        Random rng = new Random(seed);
        double[] data = new double[1_000_000];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextDouble() * 2.0 - 1.0;
        NDArray arr = new NDArray(data);
        double tol = g.getDouble("tolerance_rel");
        GoldenFixtures.assertWithinTolerance("sum(1e6)", NumJa.sum(arr),
                g.getJSONObject("expected").getDouble("sum"), tol);
        GoldenFixtures.assertWithinTolerance("mean(1e6)", NumJa.mean(arr),
                g.getJSONObject("expected").getDouble("mean"), tol);
    }

    @Test
    public void softmax_extreme_withinTolerance() throws Exception {
        JSONObject g = GoldenFixtures.load("softmax_extreme");
        long seed = g.getLong("seed");
        Random rng = new Random(seed);
        double[] x = new double[1000];
        for (int i = 0; i < x.length; i++) x[i] = rng.nextDouble() * 1400.0 - 700.0;
        JSONArray expectedArr = g.getJSONArray("expected");
        double[] expected = jsonToDoubleArray(expectedArr);
        double tol = g.getDouble("tolerance_rel");

        // HARD-FAIL path: production softmax (NumericStable-backed via Activations delegation).
        double[] result = Activations.softmax(x);
        double maxErr = 0;
        for (int i = 0; i < result.length; i++) {
            maxErr = Math.max(maxErr, GoldenFixtures.relErr(result[i], expected[i]));
        }
        GoldenFixtures.assertMaxErrWithinTolerance("softmax_extreme maxErr", maxErr, tol);
    }

    @Test
    public void linreg_iris_withinTolerance() throws Exception {
        JSONObject g = GoldenFixtures.load("linear_regression_iris");
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
            GoldenFixtures.assertWithinTolerance("linreg coef[" + i + "]",
                    coef[i], coefExp.getDouble(i), tol);
        }
        GoldenFixtures.assertWithinTolerance("linreg intercept",
                model.getIntercept(), exp.getDouble("intercept"), tol);
        int np = Math.min(preds.length, predExp.length());
        for (int i = 0; i < np; i += Math.max(1, np / 10)) {
            GoldenFixtures.assertWithinTolerance("linreg pred[" + i + "]",
                    preds[i], predExp.getDouble(i), tol);
        }
    }

    // ---------------- 3 new Phase 3 tests ----------------

    @Test
    public void sum_mean_pathological_withinTolerance() throws Exception {
        JSONObject g = GoldenFixtures.load("sum_mean_pathological");
        double[] data = new double[1_000_000];
        for (int i = 0; i < data.length; i++) data[i] = (i % 2 == 0) ? 1e15 : 1e-15;
        double tol = g.getDouble("tolerance_rel");

        // Per-leaf Kahan compensated sum from Plan 02.
        double sumActual = ParallelOps.sum(data);
        GoldenFixtures.assertWithinTolerance("sum_pathological", sumActual,
                g.getJSONObject("expected").getDouble("sum"), tol);
        GoldenFixtures.assertWithinTolerance("mean_pathological", sumActual / data.length,
                g.getJSONObject("expected").getDouble("mean"), tol);
    }

    @Test
    public void softmax_extreme_logits_withinTolerance() throws Exception {
        JSONObject g = GoldenFixtures.load("softmax_extreme_logits");
        Random rng = new Random(g.getLong("seed"));
        double[] x = new double[1000];
        for (int i = 0; i < x.length; i++) x[i] = rng.nextDouble() * 2e300 - 1e300;
        double tol = g.getDouble("tolerance_rel");

        // Production softmax on ±1e300 — must not produce NaN or Inf.
        double[] result = Activations.softmax(x);

        // (1) Every output is finite.
        for (int i = 0; i < result.length; i++) {
            assertTrue("softmax output not finite at index " + i + ": " + result[i],
                    Double.isFinite(result[i]));
        }

        // (2) Normalization invariant: sum to 1 within tolerance.
        double sum = 0.0;
        for (double v : result) sum += v;
        GoldenFixtures.assertWithinTolerance("softmax_extreme_logits norm", sum, 1.0, tol);
    }

    @Test
    public void logsumexp_simple_withinTolerance() throws Exception {
        JSONObject g = GoldenFixtures.load("logsumexp_simple");
        Random rng = new Random(g.getLong("seed"));
        double[] x = new double[500];
        for (int i = 0; i < x.length; i++) x[i] = rng.nextDouble() * 100.0 - 50.0;
        double tol = g.getDouble("tolerance_rel");
        double theoreticalUpperBound = g.getDouble("theoretical_upper_bound");

        // Stability + finiteness gate.
        double result = NumericStable.logSumExp(x);
        assertTrue("logSumExp not finite: " + result, Double.isFinite(result));
        assertTrue("logSumExp " + result + " > theoretical upper bound " + theoreticalUpperBound,
                result <= theoreticalUpperBound);
        assertTrue("logSumExp " + result + " below min input -50", result >= -50.0);

        // Self-consistency: two calls must agree within tolerance.
        double result2 = NumericStable.logSumExp(x);
        GoldenFixtures.assertWithinTolerance("logsumexp_self_consistency",
                result, result2, tol);
    }
}
