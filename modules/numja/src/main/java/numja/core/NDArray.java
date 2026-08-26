package numja.core;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;

import java.util.Arrays;

/**
 * N-dimensional array class for NumJa
 * Wraps EJML matrices with additional optimization and thread management
 */
public class NDArray {
    private DMatrixRMaj data;
    private int[] shape;
    private int ndim;
    
    /**
     * Create NDArray from 1D data
     */
    public NDArray(double... data) {
        this.data = new DMatrixRMaj(data.length, 1, true, data);
        this.shape = new int[]{data.length};
        this.ndim = 1;
    }
    
    /**
     * Create NDArray from 2D data
     */
    public NDArray(double[][] data) {
        this.data = new DMatrixRMaj(data);
        this.shape = new int[]{data.length, data[0].length};
        this.ndim = 2;
    }
    
    /**
     * Create NDArray from matrix
     */
    public NDArray(DMatrixRMaj data, int[] shape) {
        this.data = data;
        this.shape = shape;
        this.ndim = shape.length;
    }
    
    /**
     * Get array shape
     */
    public int[] getShape() {
        return shape.clone();
    }
    
    /**
     * Get number of dimensions
     */
    public int getNdim() {
        return ndim;
    }
    
    /**
     * Get total number of elements
     */
    public int getSize() {
        return data.numRows * data.numCols;
    }
    
    /**
     * Get underlying matrix
     */
    public DMatrixRMaj getData() {
        return data;
    }
    
    /**
     * Get element at index
     */
    public double get(int... indices) {
        if (ndim == 1) {
            return data.get(indices[0], 0);
        } else if (ndim == 2) {
            return data.get(indices[0], indices[1]);
        }
        throw new IllegalArgumentException("Unsupported indexing for " + ndim + "D array");
    }
    
    /**
     * Set element at index
     */
    public void set(double value, int... indices) {
        if (ndim == 1) {
            data.set(indices[0], 0, value);
        } else if (ndim == 2) {
            data.set(indices[0], indices[1], value);
        } else {
            throw new IllegalArgumentException("Unsupported indexing for " + ndim + "D array");
        }
    }
    
    /**
     * Get transpose
     */
    public NDArray getTranspose() {
        DMatrixRMaj result = new DMatrixRMaj(data.numCols, data.numRows);
        CommonOps_DDRM.transpose(data, result);
        
        int[] newShape = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            newShape[i] = shape[ndim - 1 - i];
        }
        return new NDArray(result, newShape);
    }
    
    /**
     * Reshape array
     */
    public NDArray reshape(int... newShape) {
        int size = 1;
        for (int dim : newShape) {
            size *= dim;
        }
        if (size != getSize()) {
            throw new IllegalArgumentException("Cannot reshape array of size " + getSize() + 
                " to shape " + Arrays.toString(newShape));
        }
        
        DMatrixRMaj reshaped = new DMatrixRMaj(data);
        return new NDArray(reshaped, newShape);
    }
    
    /**
     * Flatten array
     */
    public NDArray flatten() {
        double[] flat = new double[getSize()];
        for (int i = 0; i < getSize(); i++) {
            flat[i] = data.data[i];
        }
        return new NDArray(flat);
    }
    
    /**
     * Copy array
     */
    public NDArray copy() {
        DMatrixRMaj copy = new DMatrixRMaj(data.numRows, data.numCols);
        copy.setTo(data);
        return new NDArray(copy, shape.clone());
    }
    
    /**
     * Element-wise addition
     */
    public NDArray add(NDArray other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException("Shape mismatch: " +
                Arrays.toString(shape) + " vs " + Arrays.toString(other.shape));
        }

        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            CommonOps_DDRM.add(data, other.data, result);
        } else {
            ParallelOps.elementwiseBinary(data.data, other.data.data, result.data, Double::sum);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise addition with scalar
     */
    public NDArray add(double scalar) {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            result.setTo(data);
            for (int i = 0; i < n; i++) {
                result.data[i] += scalar;
            }
        } else {
            result.setTo(data);
            ParallelOps.scalarBinary(data.data, scalar, result.data, (x, s) -> x + s);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise subtraction
     */
    public NDArray subtract(NDArray other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            CommonOps_DDRM.subtract(data, other.data, result);
        } else {
            ParallelOps.elementwiseBinary(data.data, other.data.data, result.data, (x, y) -> x - y);
        }
        return new NDArray(result, shape.clone());
    }
    
    /**
     * Element-wise subtraction with scalar
     */
    public NDArray subtract(double scalar) {
        return add(-scalar);
    }
    
    /**
     * Element-wise multiplication (Hadamard product)
     */
    public NDArray multiply(NDArray other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            CommonOps_DDRM.elementMult(data, other.data, result);
        } else {
            ParallelOps.elementwiseBinary(data.data, other.data.data, result.data, (x, y) -> x * y);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise multiplication with scalar
     */
    public NDArray multiply(double scalar) {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            CommonOps_DDRM.scale(scalar, data, result);
        } else {
            ParallelOps.scalarBinary(data.data, scalar, result.data, (x, s) -> x * s);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise division
     */
    public NDArray divide(NDArray other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = data.data[i] / other.data.data[i];
            }
        } else {
            ParallelOps.elementwiseBinary(data.data, other.data.data, result.data, (x, y) -> x / y);
        }
        return new NDArray(result, shape.clone());
    }
    
    /**
     * Element-wise division with scalar
     */
    public NDArray divide(double scalar) {
        return multiply(1.0 / scalar);
    }
    
    /**
     * Element-wise power
     */
    public NDArray power(double exponent) {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.pow(data.data[i], exponent);
            }
        } else {
            result.setTo(data);
            ParallelOps.elementwiseUnary(data.data, result.data, d -> Math.pow(d, exponent));
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise absolute value
     */
    public NDArray abs() {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.abs(data.data[i]);
            }
        } else {
            ParallelOps.elementwiseUnary(data.data, result.data, Math::abs);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise square root
     */
    public NDArray sqrt() {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.sqrt(data.data[i]);
            }
        } else {
            ParallelOps.elementwiseUnary(data.data, result.data, Math::sqrt);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise exponential
     */
    public NDArray exp() {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.exp(data.data[i]);
            }
        } else {
            ParallelOps.elementwiseUnary(data.data, result.data, Math::exp);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise natural logarithm
     */
    public NDArray log() {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.log(data.data[i]);
            }
        } else {
            ParallelOps.elementwiseUnary(data.data, result.data, Math::log);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise sine
     */
    public NDArray sin() {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.sin(data.data[i]);
            }
        } else {
            ParallelOps.elementwiseUnary(data.data, result.data, Math::sin);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise cosine
     */
    public NDArray cos() {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.cos(data.data[i]);
            }
        } else {
            ParallelOps.elementwiseUnary(data.data, result.data, Math::cos);
        }
        return new NDArray(result, shape.clone());
    }

    /**
     * Element-wise tangent
     */
    public NDArray tan() {
        DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
        int n = data.numRows * data.numCols;
        if (n < ParallelOps.THRESHOLD) {
            for (int i = 0; i < n; i++) {
                result.data[i] = Math.tan(data.data[i]);
            }
        } else {
            ParallelOps.elementwiseUnary(data.data, result.data, Math::tan);
        }
        return new NDArray(result, shape.clone());
    }
    
    /**
     * Sum all elements
     */
    public double sum() {
        double sum = 0;
        for (int i = 0; i < data.numRows * data.numCols; i++) {
            sum += data.data[i];
        }
        return sum;
    }
    
    /**
     * Calculate mean
     */
    public double mean() {
        return sum() / getSize();
    }
    
    /**
     * Calculate minimum value
     */
    public double min() {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < data.numRows * data.numCols; i++) {
            if (data.data[i] < min) {
                min = data.data[i];
            }
        }
        return min;
    }
    
    /**
     * Calculate maximum value
     */
    public double max() {
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < data.numRows * data.numCols; i++) {
            if (data.data[i] > max) {
                max = data.data[i];
            }
        }
        return max;
    }

    /**
     * Product of all elements
     */
    public double prod() {
        double product = 1.0;
        for (int i = 0; i < data.numRows * data.numCols; i++) {
            product *= data.data[i];
        }
        return product;
    }
    
    /**
     * Negation
     */
    public NDArray negate() {
        return multiply(-1.0);
    }
    
    /**
     * Convert to a flat double array
     */
    public double[] toDoubleArray() {
        int size = data.numRows * data.numCols;
        double[] out = new double[size];
        System.arraycopy(data.data, 0, out, 0, size);
        return out;
    }

    /**
     * Convert to a 2D double array
     */
    public double[][] toArray() {
        if (ndim == 1) {
            double[][] out = new double[shape[0]][1];
            for (int i = 0; i < shape[0]; i++) {
                out[i][0] = data.get(i, 0);
            }
            return out;
        } else if (ndim == 2) {
            double[][] out = new double[shape[0]][shape[1]];
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    out[i][j] = data.get(i, j);
                }
            }
            return out;
        }
        throw new IllegalStateException("Cannot convert " + ndim + "D array to 2D double[][]");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NDArray(shape=").append(Arrays.toString(shape)).append(", data=\n");
        for (int i = 0; i < Math.min(5, data.numRows); i++) {
            sb.append("[");
            for (int j = 0; j < Math.min(5, data.numCols); j++) {
                sb.append(String.format("%.2f", data.get(i, j)));
                if (j < Math.min(5, data.numCols) - 1) sb.append(", ");
            }
            if (data.numCols > 5) sb.append(", ...");
            sb.append("]\n");
        }
        if (data.numRows > 5) sb.append("...\n");
        sb.append(")");
        return sb.toString();
    }
}

