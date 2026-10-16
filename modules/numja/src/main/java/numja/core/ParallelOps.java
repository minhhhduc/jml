package numja.core;

import numja.config.ThreadPoolConfig;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * ForkJoin-backed elementwise + reduce utility for {@code double[]}.
 * Threshold-gated: small arrays use the sequential loop, large arrays submit a
 * {@link RecursiveAction} tree to the singleton FJP from {@link ThreadPoolConfig}.
 * Public-API-free: callers see no new entry points.
 */
public final class ParallelOps {

    /** Minimum array size that takes the parallel path. */
    public static final int THRESHOLD = 100_000;

    /** Leaf-chunk size inside the RecursiveAction tree. */
    static final int LEAF_CUTOFF = 16_384;

    /**
     * Test-only threshold override. When >= 0, replaces THRESHOLD in the gate check.
     * Package-private: only same-package tests may set it. Tests MUST call
     * {@link #resetThresholdForTesting()} in {@code @After} to avoid leaking state.
     */
    static volatile int testThresholdOverride = -1;

    private ParallelOps() {}

    /** Force the threshold to {@code n} for the next calls. Tests only. */
    static void setThresholdForTesting(int n) {
        testThresholdOverride = n;
    }

    /** Reset the threshold override back to the default (-1 = use THRESHOLD). Tests only. */
    static void resetThresholdForTesting() {
        testThresholdOverride = -1;
    }

    private static int gate() {
        int t = testThresholdOverride;
        return t >= 0 ? t : THRESHOLD;
    }

    /**
     * Elementwise binary op over two {@code double[]} arrays, writing into {@code out}.
     * Caller guarantees {@code a.length == b.length == out.length}.
     */
    public static void elementwiseBinary(double[] a, double[] b, double[] out, DoubleBinaryOperator op) {
        final int n = a.length;
        if (n < gate()) {
            for (int i = 0; i < n; i++) out[i] = op.applyAsDouble(a[i], b[i]);
            return;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        pool.invoke(new ElementwiseTask(a, b, out, op, 0, n));
    }

    /**
     * Elementwise unary op over one {@code double[]} array, writing into {@code out}.
     * Caller guarantees {@code in.length == out.length}.
     */
    public static void elementwiseUnary(double[] in, double[] out, DoubleUnaryOperator op) {
        final int n = in.length;
        if (n < gate()) {
            for (int i = 0; i < n; i++) out[i] = op.applyAsDouble(in[i]);
            return;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        pool.invoke(new UnaryTask(in, out, op, 0, n));
    }

    /**
     * Elementwise binary op with a broadcast {@code scalar}, writing into {@code out}.
     * Caller guarantees {@code a.length == out.length}.
     */
    public static void scalarBinary(double[] a, double scalar, double[] out, DoubleBinaryOperator op) {
        final int n = a.length;
        if (n < gate()) {
            for (int i = 0; i < n; i++) out[i] = op.applyAsDouble(a[i], scalar);
            return;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        pool.invoke(new ScalarBinaryTask(a, scalar, out, op, 0, n));
    }

    /**
     * Sum all elements of {@code data}. Per-leaf Kahan compensated; tree merge naive.
     */
    public static double sum(double[] data) {
        final int n = data.length;
        if (n < gate()) {
            double s = 0.0, c = 0.0;
            for (int i = 0; i < n; i++) {
                double y = data[i] - c;
                double t = s + y;
                c = (t - s) - y;
                s = t;
            }
            return s;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        SumTask root = new SumTask(data, 0, n);
        pool.invoke(root);
        return root.total;
    }

    /**
     * Log-sum-exp compensated product: sign * exp(sum(log(|x|))) with per-leaf Kahan on
     * the log-accumulation. Tree merge is naive (sumLog adds). Empty input = 1.0; any
     * zero anywhere = 0.0 (short-circuit). Final exp is a single amplification step.
     */
    public static double prod(double[] data) {
        final int n = data.length;
        if (n == 0) return 1.0;
        if (n < gate()) {
            double sign = 1.0, sumLog = 0.0, c = 0.0;
            for (int i = 0; i < n; i++) {
                double v = data[i];
                if (v == 0.0) return 0.0;
                if (v < 0.0) sign = -sign;
                double abs = v < 0.0 ? -v : v;
                double y = Math.log(abs) - c;
                double t = sumLog + y;
                c = (t - sumLog) - y;
                sumLog = t;
            }
            return sign * Math.exp(sumLog);
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        ProdTask root = new ProdTask(data, 0, n);
        pool.invoke(root);
        if (root.sign == 0.0) return 0.0;
        return root.sign * Math.exp(root.sumLog);
    }

    /**
     * Minimum element of {@code data}.
     */
    public static double min(double[] data) {
        final int n = data.length;
        if (n < gate()) {
            double m = Double.MAX_VALUE;
            for (int i = 0; i < n; i++) if (data[i] < m) m = data[i];
            return m;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        MinTask root = new MinTask(data, 0, n);
        pool.invoke(root);
        return root.best;
    }

    /**
     * Maximum element of {@code data}.
     */
    public static double max(double[] data) {
        final int n = data.length;
        if (n < gate()) {
            double m = -Double.MAX_VALUE;
            for (int i = 0; i < n; i++) if (data[i] > m) m = data[i];
            return m;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        MaxTask root = new MaxTask(data, 0, n);
        pool.invoke(root);
        return root.best;
    }

    private static final class ElementwiseTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] a, b, out;
        private final DoubleBinaryOperator op;
        private final int lo, hi;

        ElementwiseTask(double[] a, double[] b, double[] out, DoubleBinaryOperator op, int lo, int hi) {
            this.a = a; this.b = b; this.out = out; this.op = op; this.lo = lo; this.hi = hi;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                for (int i = lo; i < hi; i++) out[i] = op.applyAsDouble(a[i], b[i]);
                return;
            }
            int mid = (lo + hi) >>> 1;
            invokeAll(new ElementwiseTask(a, b, out, op, lo, mid),
                      new ElementwiseTask(a, b, out, op, mid, hi));
        }
    }

    private static final class UnaryTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] in, out;
        private final DoubleUnaryOperator op;
        private final int lo, hi;

        UnaryTask(double[] in, double[] out, DoubleUnaryOperator op, int lo, int hi) {
            this.in = in; this.out = out; this.op = op; this.lo = lo; this.hi = hi;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                for (int i = lo; i < hi; i++) out[i] = op.applyAsDouble(in[i]);
                return;
            }
            int mid = (lo + hi) >>> 1;
            invokeAll(new UnaryTask(in, out, op, lo, mid),
                      new UnaryTask(in, out, op, mid, hi));
        }
    }

    private static final class ScalarBinaryTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] a;
        private final double scalar;
        private final double[] out;
        private final DoubleBinaryOperator op;
        private final int lo, hi;

        ScalarBinaryTask(double[] a, double scalar, double[] out, DoubleBinaryOperator op, int lo, int hi) {
            this.a = a; this.scalar = scalar; this.out = out; this.op = op; this.lo = lo; this.hi = hi;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                for (int i = lo; i < hi; i++) out[i] = op.applyAsDouble(a[i], scalar);
                return;
            }
            int mid = (lo + hi) >>> 1;
            invokeAll(new ScalarBinaryTask(a, scalar, out, op, lo, mid),
                      new ScalarBinaryTask(a, scalar, out, op, mid, hi));
        }
    }

    private static final class SumTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] data;
        private final int lo, hi;
        double total;

        SumTask(double[] data, int lo, int hi) {
            this.data = data; this.lo = lo; this.hi = hi;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                // ponytail: Kahan compensation, ~5% leaf overhead vs naive, error O(log n * eps).
                double s = 0.0, c = 0.0;
                for (int i = lo; i < hi; i++) {
                    double y = data[i] - c;
                    double t = s + y;
                    c = (t - s) - y;
                    s = t;
                }
                this.total = s;
                return;
            }
            int mid = (lo + hi) >>> 1;
            SumTask left = new SumTask(data, lo, mid);
            SumTask right = new SumTask(data, mid, hi);
            invokeAll(left, right);
            this.total = left.total + right.total;
        }
    }

    private static final class ProdTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] data;
        private final int lo, hi;
        double sign;    //  1.0 positive, -1.0 negative, 0.0 zero short-circuit
        double sumLog;  // accumulated sum of log(|x|), Kahan-compensated
        double c;       // Kahan compensation for sumLog

        ProdTask(double[] data, int lo, int hi) {
            this.data = data; this.lo = lo; this.hi = hi;
            this.sign = 1.0; this.sumLog = 0.0; this.c = 0.0;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                for (int i = lo; i < hi; i++) {
                    double v = data[i];
                    if (v == 0.0) { this.sign = 0.0; this.sumLog = 0.0; this.c = 0.0; return; }
                    if (v < 0.0) this.sign = -this.sign;
                    double abs = v < 0.0 ? -v : v;
                    double y = Math.log(abs) - this.c;
                    double t = this.sumLog + y;
                    this.c = (t - this.sumLog) - y;
                    this.sumLog = t;
                }
                return;
            }
            int mid = (lo + hi) >>> 1;
            ProdTask left = new ProdTask(data, lo, mid);
            ProdTask right = new ProdTask(data, mid, hi);
            invokeAll(left, right);
            this.sign = left.sign * right.sign;
            this.sumLog = left.sumLog + right.sumLog;
        }
    }

    private static final class MinTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] data;
        private final int lo, hi;
        double best;

        MinTask(double[] data, int lo, int hi) {
            this.data = data; this.lo = lo; this.hi = hi;
            // Explicit identity init — Java default 0.0 is wrong for min over
            // all-positive data. Defends WR-05 invariant: best is always the
            // correct identity before compute() overwrites it.
            this.best = Double.MAX_VALUE;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                double m = best;
                for (int i = lo; i < hi; i++) if (data[i] < m) m = data[i];
                this.best = m;
                return;
            }
            int mid = (lo + hi) >>> 1;
            MinTask left = new MinTask(data, lo, mid);
            MinTask right = new MinTask(data, mid, hi);
            invokeAll(left, right);
            this.best = Math.min(left.best, right.best);
        }
    }

    private static final class MaxTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] data;
        private final int lo, hi;
        double best;

        MaxTask(double[] data, int lo, int hi) {
            this.data = data; this.lo = lo; this.hi = hi;
            // Explicit identity init — Java default 0.0 is wrong for max over
            // all-negative data. Defends WR-05 invariant: best is always the
            // correct identity before compute() overwrites it.
            this.best = -Double.MAX_VALUE;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                double m = best;
                for (int i = lo; i < hi; i++) if (data[i] > m) m = data[i];
                this.best = m;
                return;
            }
            int mid = (lo + hi) >>> 1;
            MaxTask left = new MaxTask(data, lo, mid);
            MaxTask right = new MaxTask(data, mid, hi);
            invokeAll(left, right);
            this.best = Math.max(left.best, right.best);
        }
    }
}
