package numja.core;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Hardware-abstraction spine for numja's hot-path numeric kernels.
 *
 * <p>This interface is the swap point for future GPU/TPU backends. The default
 * CPU implementation lives in {@link CpuThreadBackend}; selection is owned by
 * {@link BackendSelector}. Phase 5 wires the abstraction only — no caller of
 * {@link ParallelOps} or {@link ArrayOps} is dispatched through this interface
 * yet (that wiring lives in 05-02).
 *
 * <p>Op signatures mirror the {@link ParallelOps} statics 1:1, plus a flat-array
 * matmul that takes raw {@code double[]} + dims so a future GPU backend can read
 * the raw data without going through {@code DMatrixRMaj}.
 */
public interface ComputeBackend {

    /** Elementwise binary op over two {@code double[]}, writing into {@code out}. */
    void elementwiseBinary(double[] a, double[] b, double[] out, DoubleBinaryOperator op);

    /** Elementwise unary op over one {@code double[]}, writing into {@code out}. */
    void elementwiseUnary(double[] in, double[] out, DoubleUnaryOperator op);

    /** Elementwise binary op with a broadcast {@code scalar}, writing into {@code out}. */
    void scalarBinary(double[] a, double scalar, double[] out, DoubleBinaryOperator op);

    /** Sum all elements (compensated per-leaf). */
    double sum(double[] data);

    /** Product (log-sum-exp compensated). */
    double prod(double[] data);

    /** Minimum element. */
    double min(double[] data);

    /** Maximum element. */
    double max(double[] data);

    /**
     * Matrix multiply on raw row-major {@code double[]} arrays. Caller owns
     * {@code out} allocation. Matches {@link ArrayOps#dot}'s shape: a is
     * {@code aRows x aCols}, b is {@code bRows x bCols}, requires
     * {@code aCols == bRows}; output is {@code aRows x bCols}.
     */
    void matmul(double[] a, int aRows, int aCols,
                double[] b, int bRows, int bCols,
                double[] out);
}
