package numja.core;

import numja.config.ThreadPoolConfig;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.function.DoubleBinaryOperator;

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

    private ParallelOps() {}

    /**
     * Elementwise binary op over two {@code double[]} arrays, writing into {@code out}.
     * Caller guarantees {@code a.length == b.length == out.length}.
     */
    public static void elementwiseBinary(double[] a, double[] b, double[] out, DoubleBinaryOperator op) {
        final int n = a.length;
        if (n < THRESHOLD) {
            for (int i = 0; i < n; i++) out[i] = op.applyAsDouble(a[i], b[i]);
            return;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        pool.invoke(new ElementwiseTask(a, b, out, op, 0, n));
    }

    /**
     * Sum all elements of {@code data}. Deterministic left-to-right tree merge.
     */
    public static double sum(double[] data) {
        final int n = data.length;
        if (n < THRESHOLD) {
            double s = 0.0;
            for (int i = 0; i < n; i++) s += data[i];
            return s;
        }
        ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
        SumTask root = new SumTask(data, 0, n);
        pool.invoke(root);
        return root.total;
    }

    /**
     * Minimum element of {@code data}.
     */
    public static double min(double[] data) {
        final int n = data.length;
        if (n < THRESHOLD) {
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
        if (n < THRESHOLD) {
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
                double s = 0.0;
                for (int i = lo; i < hi; i++) s += data[i];
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

    private static final class MinTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final double[] data;
        private final int lo, hi;
        double best;

        MinTask(double[] data, int lo, int hi) {
            this.data = data; this.lo = lo; this.hi = hi;
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                double m = Double.MAX_VALUE;
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
        }

        @Override
        protected void compute() {
            if (hi - lo <= LEAF_CUTOFF) {
                double m = -Double.MAX_VALUE;
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
