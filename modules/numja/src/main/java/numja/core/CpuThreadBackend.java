package numja.core;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Default {@link ComputeBackend} — singleton delegating 1:1 to the existing
 * {@link ParallelOps} / EJML paths.
 *
 * <p>Zero behavior change: every elementwise/reduce call routes through the
 * same {@link ParallelOps} static that the rest of numja already calls, so the
 * threshold gate, ForkJoinPool routing, Kahan sum, and log-sum-exp prod
 * implementations are reused verbatim. The {@link #matmul} body is the
 * row-major double[] equivalent of {@link ArrayOps#dot} (same EJML
 * {@code CommonOps_DDRM.mult} call) so a future GPU backend has a flat-array
 * swap point without going through {@code DMatrixRMaj}.
 */
public final class CpuThreadBackend implements ComputeBackend {

    private static final CpuThreadBackend INSTANCE = new CpuThreadBackend();

    private CpuThreadBackend() {}

    /** Returns the singleton CPU backend. */
    public static CpuThreadBackend getInstance() {
        return INSTANCE;
    }

    @Override
    public void elementwiseBinary(double[] a, double[] b, double[] out, DoubleBinaryOperator op) {
        ParallelOps.elementwiseBinary(a, b, out, op);
    }

    @Override
    public void elementwiseUnary(double[] in, double[] out, DoubleUnaryOperator op) {
        ParallelOps.elementwiseUnary(in, out, op);
    }

    @Override
    public void scalarBinary(double[] a, double scalar, double[] out, DoubleBinaryOperator op) {
        ParallelOps.scalarBinary(a, scalar, out, op);
    }

    @Override
    public double sum(double[] data) {
        return ParallelOps.sum(data);
    }

    @Override
    public double prod(double[] data) {
        return ParallelOps.prod(data);
    }

    @Override
    public double min(double[] data) {
        return ParallelOps.min(data);
    }

    @Override
    public double max(double[] data) {
        return ParallelOps.max(data);
    }

    @Override
    public void matmul(double[] a, int aRows, int aCols,
                       double[] b, int bRows, int bCols,
                       double[] out) {
        DMatrixRMaj ma = new DMatrixRMaj(aRows, aCols, true, a);
        DMatrixRMaj mb = new DMatrixRMaj(bRows, bCols, true, b);
        DMatrixRMaj mo = new DMatrixRMaj(aRows, bCols);
        CommonOps_DDRM.mult(ma, mb, mo);
        System.arraycopy(mo.data, 0, out, 0, aRows * bCols);
    }
}
