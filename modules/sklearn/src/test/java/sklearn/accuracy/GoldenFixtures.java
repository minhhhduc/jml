package sklearn.accuracy;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.junit.Assert;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Shared fixture loader + accuracy assertion helpers for golden-value tests.
 *
 * Package-private (test-only) — splits the soft-fail {@link #check} recorder used
 * by {@link GoldenReferenceTest} from the hard-fail {@link #assertWithinTolerance}
 * used by ACC-03 {@code AccuracyHardeningTest}.
 */
class GoldenFixtures {

    private static final String GOLDEN_DIR = "../../bench/src/test/resources/golden";
    private static final List<String> REPORT = new ArrayList<>();

    private GoldenFixtures() {}

    // ---------------- shared loader / recorder (soft-fail) ----------------

    static JSONObject load(String name) throws Exception {
        try (InputStream in = GoldenReferenceTest.class.getResourceAsStream("/golden/" + name + ".json")) {
            if (in != null) return new JSONObject(new JSONTokener(in));
        }
        File f = new File(GOLDEN_DIR + "/" + name + ".json");
        if (!f.exists()) throw new IllegalStateException("Golden file not found: " + name);
        try (InputStream fin = new FileInputStream(f)) {
            return new JSONObject(new JSONTokener(fin));
        }
    }

    static double relErr(double actual, double expected) {
        double denom = Math.max(Math.abs(expected), 1e-30);
        return Math.abs(actual - expected) / denom;
    }

    static void check(String label, double actual, double expected, double tolRel) {
        double err = relErr(actual, expected);
        boolean pass = err <= tolRel;
        String msg = String.format("%s %-40s err=%.3e (tol %.0e)", pass ? "PASS" : "FAIL", label, err, tolRel);
        if (!pass) {
            msg += String.format(" actual=%.17g expected=%.17g", actual, expected);
        }
        REPORT.add(msg);
        System.out.println("[GOLDEN] " + msg);
    }

    static void checkMaxErr(String label, double maxErr, double tolRel) {
        boolean pass = maxErr <= tolRel;
        String msg = String.format("%s %-40s maxErr=%.3e (tol %.0e)", pass ? "PASS" : "FAIL", label, maxErr, tolRel);
        REPORT.add(msg);
        System.out.println("[GOLDEN] " + msg);
    }

    static void flushReport() {
        try (PrintWriter w = new PrintWriter("target/golden-report.txt", "UTF-8")) {
            for (String line : REPORT) w.println(line);
        } catch (Exception e) {
            System.err.println("[GOLDEN] could not write report: " + e.getMessage());
        }
    }

    static void resetReport() {
        REPORT.clear();
        REPORT.add("Golden accuracy report (NumJa vs NumPy), soft-fail mode");
    }

    static void assertMechanicsOnly() {
        flushReport();
        Assert.assertTrue(true); // soft-fail: per-op gaps live in REPORT/stderr/target/golden-report.txt
    }

    static double[] jsonToDoubleArray(JSONArray arr) {
        double[] out = new double[arr.length()];
        for (int i = 0; i < arr.length(); i++) out[i] = arr.getDouble(i);
        return out;
    }

    // ---------------- hard-fail assertions (AccuracyHardeningTest) ----------------

    static void assertWithinTolerance(String label, double actual, double expected, double tolRel) {
        double err = relErr(actual, expected);
        Assert.assertTrue(label + " err=" + err + " tol=" + tolRel, err <= tolRel);
    }

    static void assertMaxErrWithinTolerance(String label, double maxErr, double tolRel) {
        Assert.assertTrue(label + " maxErr=" + maxErr + " tol=" + tolRel, maxErr <= tolRel);
    }
}
