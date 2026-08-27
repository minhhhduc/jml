package numja;

/**
 * Max-shift stable numerical primitives for softmax / log-softmax / log-sum-exp.
 *
 * All three methods subtract the input max before any {@link Math#exp} call so
 * the largest exponent is {@code 0} — finite inputs always produce finite outputs.
 * No allocation beyond the returned array (logSumExp returns a scalar).
 */
public final class NumericStable {

    private NumericStable() {}

    /**
     * Max-shift stable softmax: subtract max before exp so result is finite for any finite input.
     */
    public static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double val : x) {
            if (val > max) max = val;
        }

        double[] exp = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }

        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = exp[i] / sum;
        }
        return result;
    }

    /**
     * Max-shift stable log-softmax: returns log probabilities without the underflow of softmax(x) -> log().
     */
    public static double[] logSoftmax(double[] x) {
        double denom = logSumExp(x);
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = (x[i] - denom);
        }
        return result;
    }

    /**
     * Max-shift stable log(sum(exp(x))): returns -Inf if x is empty; otherwise max + log(sum(exp(x - max))).
     */
    public static double logSumExp(double[] x) {
        if (x.length == 0) return Double.NEGATIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > max) max = v;
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += Math.exp(x[i] - max);
        }
        return max + Math.log(sum);
    }
}
