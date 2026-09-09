package numja.core;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.dense.row.decomposition.svd.SvdImplicitQrDecompose_DDRM;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import numja.config.ThreadPoolConfig;

/**
 * Array creation and element-wise operations
 */
public class ArrayOps {
    
    private static ThreadPoolConfig config = ThreadPoolConfig.getInstance();
    
    /**
     * Create array from double values
     */
    public static NDArray array(double... values) {
        return new NDArray(values);
    }
    
    /**
     * Create array from 2D double array
     */
    public static NDArray array(double[][] values) {
        return new NDArray(values);
    }
    
    /**
     * Create array of zeros
     */
    public static NDArray zeros(int rows, int cols) {
        DMatrixRMaj data = new DMatrixRMaj(rows, cols);
        return new NDArray(data, new int[]{rows, cols});
    }
    
    /**
     * Create 1D array of zeros
     */
    public static NDArray zeros(int size) {
        DMatrixRMaj data = new DMatrixRMaj(size, 1);
        return new NDArray(data, new int[]{size});
    }
    
    /**
     * Create array of ones
     */
    public static NDArray ones(int rows, int cols) {
        DMatrixRMaj data = new DMatrixRMaj(rows, cols);
        CommonOps_DDRM.fill(data, 1.0);
        return new NDArray(data, new int[]{rows, cols});
    }
    
    /**
     * Create 1D array of ones
     */
    public static NDArray ones(int size) {
        DMatrixRMaj data = new DMatrixRMaj(size, 1);
        CommonOps_DDRM.fill(data, 1.0);
        return new NDArray(data, new int[]{size});
    }

    /**
     * Create uninitialized array
     */
    public static NDArray empty(int rows, int cols) {
        DMatrixRMaj data = new DMatrixRMaj(rows, cols);
        return new NDArray(data, new int[]{rows, cols});
    }

    /**
     * Create 1D uninitialized array
     */
    public static NDArray empty(int size) {
        DMatrixRMaj data = new DMatrixRMaj(size, 1);
        return new NDArray(data, new int[]{size});
    }
    
    /**
     * Create identity matrix
     */
    public static NDArray eye(int n) {
        DMatrixRMaj data = CommonOps_DDRM.identity(n);
        return new NDArray(data, new int[]{n, n});
    }
    
    /**
     * Create array with evenly spaced values
     */
    public static NDArray arange(double start, double stop, double step) {
        int size = (int) Math.ceil((stop - start) / step);
        double[] data = new double[size];
        
        for (int i = 0; i < size; i++) {
            data[i] = start + i * step;
        }
        
        return new NDArray(data);
    }
    
    /**
     * Create array with num evenly spaced values
     */
    public static NDArray linspace(double start, double stop, int num) {
        double[] data = new double[num];
        double step = (stop - start) / (num - 1);
        
        for (int i = 0; i < num; i++) {
            data[i] = start + i * step;
        }
        
        return new NDArray(data);
    }
    
    /**
     * Create full array
     */
    public static NDArray full(int rows, int cols, double value) {
        DMatrixRMaj data = new DMatrixRMaj(rows, cols);
        CommonOps_DDRM.fill(data, value);
        return new NDArray(data, new int[]{rows, cols});
    }
    
    /**
     * Element-wise addition
     */
    public static NDArray add(NDArray a, NDArray b) {
        return a.add(b);
    }
    
    /**
     * Element-wise subtraction
     */
    public static NDArray subtract(NDArray a, NDArray b) {
        return a.subtract(b);
    }
    
    /**
     * Element-wise multiplication
     */
    public static NDArray multiply(NDArray a, NDArray b) {
        return a.multiply(b);
    }
    
    /**
     * Element-wise division
     */
    public static NDArray divide(NDArray a, NDArray b) {
        return a.divide(b);
    }
    
    /**
     * Element-wise power
     */
    public static NDArray power(NDArray a, double exponent) {
        return a.power(exponent);
    }
    
    /**
     * Absolute value
     */
    public static NDArray abs(NDArray a) {
        return a.abs();
    }
    
    /**
     * Square root
     */
    public static NDArray sqrt(NDArray a) {
        return a.sqrt();
    }
    
    /**
     * Exponential
     */
    public static NDArray exp(NDArray a) {
        return a.exp();
    }
    
    /**
     * Natural logarithm
     */
    public static NDArray log(NDArray a) {
        return a.log();
    }
    
    /**
     * Sine
     */
    public static NDArray sin(NDArray a) {
        return a.sin();
    }
    
    /**
     * Cosine
     */
    public static NDArray cos(NDArray a) {
        return a.cos();
    }

    /**
     * Tangent
     */
    public static NDArray tan(NDArray a) {
        return a.tan();
    }
    
    /**
     * Dot product / Matrix multiplication. Routes through
     * {@link BackendSelector#get()}().{@code matmul} so future GPU backends can
     * be swapped in without touching this call site. The selected backend's
     * matmul body is byte-equivalent to the prior inline EJML path
     * ({@code CpuThreadBackend.matmul} wraps {@code CommonOps_DDRM.mult} on
     * raw arrays), so the numerics are bit-identical.
     */
    public static NDArray dot(NDArray a, NDArray b) {
        DMatrixRMaj ma = a.getData();
        DMatrixRMaj mb = b.getData();
        int aRows = ma.numRows, aCols = ma.numCols, bRows = mb.numRows, bCols = mb.numCols;
        DMatrixRMaj result = new DMatrixRMaj(aRows, bCols);
        BackendSelector.get().matmul(ma.data, aRows, aCols, mb.data, bRows, bCols, result.data);
        return new NDArray(result, new int[]{aRows, bCols});
    }
    
    /**
     * Sum all elements
     */
    public static double sum(NDArray a) {
        return a.sum();
    }
    
    /**
     * Mean
     */
    public static double mean(NDArray a) {
        return a.mean();
    }
    
    /**
     * Standard deviation
     */
    public static double std(NDArray a) {
        double mean = a.mean();
        double sumSquaredDiff = 0;
        DMatrixRMaj data = a.getData();
        
        for (int i = 0; i < data.numRows * data.numCols; i++) {
            double diff = data.data[i] - mean;
            sumSquaredDiff += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDiff / a.getSize());
    }
    
    /**
     * Variance
     */
    public static double var(NDArray a) {
        double std = std(a);
        return std * std;
    }
    
    /**
     * Minimum
     */
    public static double min(NDArray a) {
        return a.min();
    }
    
    /**
     * Maximum
     */
    public static double max(NDArray a) {
        return a.max();
    }
    
    /**
     * Index of minimum
     */
    public static int argmin(NDArray a) {
        DMatrixRMaj data = a.getData();
        int minIdx = 0;
        double minVal = data.data[0];
        
        for (int i = 1; i < data.numRows * data.numCols; i++) {
            if (data.data[i] < minVal) {
                minVal = data.data[i];
                minIdx = i;
            }
        }
        
        return minIdx;
    }
    
    /**
     * Index of maximum
     */
    public static int argmax(NDArray a) {
        DMatrixRMaj data = a.getData();
        int maxIdx = 0;
        double maxVal = data.data[0];
        
        for (int i = 1; i < data.numRows * data.numCols; i++) {
            if (data.data[i] > maxVal) {
                maxVal = data.data[i];
                maxIdx = i;
            }
        }
        
        return maxIdx;
    }

    /**
     * Product of all elements
     */
    public static double prod(NDArray a) {
        return a.prod();
    }
}

