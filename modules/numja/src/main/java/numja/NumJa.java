package numja;

import numja.core.NDArray;
import numja.core.ArrayOps;
import numja.linalg.*;
import numja.config.ThreadPoolConfig;

/**
 * NumJa - NumPy-like Array Library for Java
 * Main entry point for the library
 */
public class NumJa {
    
    // Version
    public static final String VERSION = "0.1.0";
    
    // Global thread config
    private static final ThreadPoolConfig threadConfig = ThreadPoolConfig.getInstance();
    
    // Array creation methods
    /**
     * Create a 1D array from individual values
     * @param values the values for the array
     * @return a new NDArray with the given values
     */
    public static NDArray array(double... values) {
        return ArrayOps.array(values);
    }
    
    /**
     * Create a 2D array from a 2D array of values
     * @param values the 2D array of values
     * @return a new NDArray
     */
    public static NDArray array(double[][] values) {
        return ArrayOps.array(values);
    }
    
    /**
     * Create a matrix of zeros with given dimensions
     * @param rows number of rows
     * @param cols number of columns
     * @return a new NDArray filled with zeros
     */
    public static NDArray zeros(int rows, int cols) {
        return ArrayOps.zeros(rows, cols);
    }
    
    /**
     * Create a 1D array of zeros
     * @param size length of the array
     * @return a new NDArray filled with zeros
     */
    public static NDArray zeros(int size) {
        return ArrayOps.zeros(size);
    }
    
    /**
     * Create a matrix of ones with given dimensions
     * @param rows number of rows
     * @param cols number of columns
     * @return a new NDArray filled with ones
     */
    public static NDArray ones(int rows, int cols) {
        return ArrayOps.ones(rows, cols);
    }
    
    /**
     * Create a 1D array of ones
     * @param size length of the array
     * @return a new NDArray filled with ones
     */
    public static NDArray ones(int size) {
        return ArrayOps.ones(size);
    }

    /**
     * Create a matrix with uninitialized values
     * @param rows number of rows
     * @param cols number of columns
     * @return a new NDArray with undefined values
     */
    public static NDArray empty(int rows, int cols) {
        return ArrayOps.empty(rows, cols);
    }

    /**
     * Create a 1D array with uninitialized values
     * @param size length of the array
     * @return a new NDArray with undefined values
     */
    public static NDArray empty(int size) {
        return ArrayOps.empty(size);
    }
    
    /**
     * Create an identity matrix
     * @param n dimension of the identity matrix
     * @return a new n×n identity matrix
     */
    public static NDArray eye(int n) {
        return ArrayOps.eye(n);
    }
    
    /**
     * Create an array with evenly spaced values
     * @param start starting value
     * @param stop ending value (exclusive)
     * @param step step size
     * @return a new 1D NDArray with values from start to stop with given step
     */
    public static NDArray arange(double start, double stop, double step) {
        return ArrayOps.arange(start, stop, step);
    }
    
    /**
     * Create an array with evenly spaced values
     * @param start starting value
     * @param stop ending value
     * @param num number of samples to generate
     * @return a new 1D NDArray with num evenly spaced values
     */
    public static NDArray linspace(double start, double stop, int num) {
        return ArrayOps.linspace(start, stop, num);
    }
    
    /**
     * Create a matrix filled with a specific value
     * @param rows number of rows
     * @param cols number of columns
     * @param value the value to fill
     * @return a new NDArray filled with the given value
     */
    public static NDArray full(int rows, int cols, double value) {
        return ArrayOps.full(rows, cols, value);
    }
    
    // Element-wise operations
    /**
     * Element-wise addition of two arrays
     * @param a first array
     * @param b second array
     * @return a new NDArray with element-wise sum
     */
    public static NDArray add(NDArray a, NDArray b) {
        return ArrayOps.add(a, b);
    }
    
    /**
     * Element-wise subtraction
     * @param a first array
     * @param b second array
     * @return a new NDArray with element-wise difference
     */
    public static NDArray subtract(NDArray a, NDArray b) {
        return ArrayOps.subtract(a, b);
    }
    
    /**
     * Element-wise multiplication
     * @param a first array
     * @param b second array
     * @return a new NDArray with element-wise product
     */
    public static NDArray multiply(NDArray a, NDArray b) {
        return ArrayOps.multiply(a, b);
    }
    
    /**
     * Element-wise division
     * @param a first array
     * @param b second array
     * @return a new NDArray with element-wise quotient
     */
    public static NDArray divide(NDArray a, NDArray b) {
        return ArrayOps.divide(a, b);
    }
    
    /**
     * Raise each element to a power
     * @param a input array
     * @param exponent power to raise to
     * @return a new NDArray with each element raised to the exponent
     */
    public static NDArray power(NDArray a, double exponent) {
        return ArrayOps.power(a, exponent);
    }
    
    /**
     * Absolute value of each element
     * @param a input array
     * @return a new NDArray with absolute values
     */
    public static NDArray abs(NDArray a) {
        return ArrayOps.abs(a);
    }
    
    /**
     * Square root of each element
     * @param a input array
     * @return a new NDArray with square roots
     */
    public static NDArray sqrt(NDArray a) {
        return ArrayOps.sqrt(a);
    }
    
    /**
     * Exponential function (e^x) for each element
     * @param a input array
     * @return a new NDArray with exponential values
     */
    public static NDArray exp(NDArray a) {
        return ArrayOps.exp(a);
    }
    
    /**
     * Natural logarithm of each element
     * @param a input array
     * @return a new NDArray with logarithm values
     */
    public static NDArray log(NDArray a) {
        return ArrayOps.log(a);
    }
    
    /**
     * Sine function for each element
     * @param a input array (in radians)
     * @return a new NDArray with sine values
     */
    public static NDArray sin(NDArray a) {
        return ArrayOps.sin(a);
    }
    
    /**
     * Cosine function for each element
     * @param a input array (in radians)
     * @return a new NDArray with cosine values
     */
    public static NDArray cos(NDArray a) {
        return ArrayOps.cos(a);
    }

    /**
     * Tangent function for each element
     * @param a input array (in radians)
     * @return a new NDArray with tangent values
     */
    public static NDArray tan(NDArray a) {
        return ArrayOps.tan(a);
    }
    
    // Reductions
    /**
     * Matrix multiplication (dot product)
     * @param a first array
     * @param b second array
     * @return a new NDArray with the result of matrix multiplication
     */
    public static NDArray dot(NDArray a, NDArray b) {
        return ArrayOps.dot(a, b);
    }

    /**
     * Matrix multiplication (alias for dot)
     * @param a first array
     * @param b second array
     * @return a new NDArray with the result of matrix multiplication
     */
    public static NDArray matmul(NDArray a, NDArray b) {
        return ArrayOps.dot(a, b);
    }
    
    /**
     * Sum of all elements
     * @param a input array
     * @return the sum of all elements
     */
    public static double sum(NDArray a) {
        return ArrayOps.sum(a);
    }
    
    /**
     * Mean (average) of all elements
     * @param a input array
     * @return the arithmetic mean
     */
    public static double mean(NDArray a) {
        return ArrayOps.mean(a);
    }
    
    /**
     * Standard deviation of all elements
     * @param a input array
     * @return the standard deviation
     */
    public static double std(NDArray a) {
        return ArrayOps.std(a);
    }
    
    /**
     * Variance of all elements
     * @param a input array
     * @return the variance
     */
    public static double var(NDArray a) {
        return ArrayOps.var(a);
    }
    
    /**
     * Minimum value in the array
     * @param a input array
     * @return the minimum value
     */
    public static double min(NDArray a) {
        return ArrayOps.min(a);
    }
    
    /**
     * Maximum value in the array
     * @param a input array
     * @return the maximum value
     */
    public static double max(NDArray a) {
        return ArrayOps.max(a);
    }
    
    /**
     * Index of the minimum value
     * @param a input array
     * @return the index of the minimum value
     */
    public static int argmin(NDArray a) {
        return ArrayOps.argmin(a);
    }
    
    /**
     * Index of the maximum value
     * @param a input array
     * @return the index of the maximum value
     */
    public static int argmax(NDArray a) {
        return ArrayOps.argmax(a);
    }

    /**
     * Product of all elements
     * @param a input array
     * @return the product of all elements
     */
    public static double prod(NDArray a) {
        return ArrayOps.prod(a);
    }
    
    // Linear Algebra
    /**
     * Matrix inverse
     * @param a input square matrix
     * @return the inverse of the matrix
     */
    public static NDArray inv(NDArray a) {
        return LinAlg.inv(a);
    }
    
    /**
     * Determinant of a matrix
     * @param a input square matrix
     * @return the determinant
     */
    public static double det(NDArray a) {
        return LinAlg.det(a);
    }
    
    /**
     * Rank of a matrix
     * @param a input matrix
     * @return the rank
     */
    public static int matrixRank(NDArray a) {
        return LinAlg.matrixRank(a);
    }

    /**
     * Rank of a matrix (Python-style naming)
     * @param a input matrix
     * @return the rank
     */
    public static int matrix_rank(NDArray a) {
        return LinAlg.matrix_rank(a);
    }
    
    /**
     * Trace of a matrix (sum of diagonal elements)
     * @param a input square matrix
     * @return the trace
     */
    public static double trace(NDArray a) {
        return LinAlg.trace(a);
    }
    
    /**
     * Solve linear system Ax = b
     * @param A coefficient matrix
     * @param b right-hand side vector
     * @return the solution vector x
     */
    public static NDArray solve(NDArray A, NDArray b) {
        return LinAlg.solve(A, b);
    }

    /**
     * Least squares solution for overdetermined system
     * @param A coefficient matrix
     * @param b right-hand side vector
     * @return Lstsq object containing results
     */
    public static Lstsq lstsq(NDArray A, NDArray b) {
        return LinAlg.lstsq(A, b);
    }
    
    /**
     * Singular Value Decomposition
     * @param a input matrix
     * @return SVD object containing U, S, V matrices
     */
    public static SVD svd(NDArray a) {
        return LinAlg.svd(a);
    }
    
    /**
     * QR decomposition
     * @param a input matrix
     * @return QR object containing Q and R matrices
     */
    public static QR qr(NDArray a) {
        return LinAlg.qr(a);
    }
    
    /**
     * Eigenvalue decomposition
     * @param a input square matrix
     * @return Eigen object containing eigenvalues and eigenvectors
     */
    public static Eigen eig(NDArray a) {
        return LinAlg.eig(a);
    }
    
    public static NDArray cholesky(NDArray a) {
        return LinAlg.cholesky(a);
    }

    public static NDArray cholesky(NDArray a, boolean lower) {
        return LinAlg.cholesky(a, lower);
    }
    
    public static double norm(NDArray a) {
        return LinAlg.norm(a);
    }
    
    public static double norm1(NDArray a) {
        return LinAlg.norm1(a);
    }
    
    public static double normInf(NDArray a) {
        return LinAlg.normInf(a);
    }
    
    public static NDArray pinv(NDArray a) {
        return LinAlg.pinv(a);
    }
    
    public static NDArray matrixPower(NDArray a, int n) {
        return LinAlg.matrixPower(a, n);
    }
    
    // Thread configuration
    public static ThreadPoolConfig getThreadConfig() {
        return threadConfig;
    }
    
    public static void setNumThreads(Integer numThreads) {
        threadConfig.setThreads(numThreads);
    }
    
    public static int getNumThreads() {
        return threadConfig.getCurrentThreads();
    }

    public static int get_num_threads() {
        return threadConfig.get_threads();
    }

    public static void set_num_threads(Integer numThreads) {
        threadConfig.set_threads(numThreads);
    }

    public static ThreadPoolConfig get_config() {
        return threadConfig;
    }
    
    public static String getVersion() {
        return VERSION;
    }
    
    public static void main(String[] args) {
        System.out.println("NumJa v" + VERSION);
        System.out.println("NumPy-like Array Library with Automatic Thread Optimization");
        System.out.println();
        System.out.println("Thread Configuration:");
        System.out.println(threadConfig.getConfigDict());
    }
}

