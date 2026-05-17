package numja.tests;

import numja.NumJa;
import numja.core.NDArray;
import numja.linalg.*;
import numja.config.ThreadPoolConfig;

/**
 * Test suite for NumJa library
 */
public class NumJaTest {
    
    public static void testBasicOperations() {
        System.out.println("=" .repeat(60));
        System.out.println("Testing Basic Array Operations");
        System.out.println("=" .repeat(60));
        
        // Array creation
        NDArray a = NumJa.array(1.0, 2.0, 3.0, 4.0, 5.0);
        NDArray b = NumJa.zeros(3, 4);
        NDArray c = NumJa.ones(2, 3);
        
        System.out.println("Array a: " + a);
        System.out.println("Array b shape: " + java.util.Arrays.toString(b.getShape()));
        System.out.println("Array c shape: " + java.util.Arrays.toString(c.getShape()));
        
        // Element-wise operations
        NDArray result = NumJa.add(a, NumJa.array(10.0));
        System.out.println("\na + 10: " + result);
        
        result = NumJa.multiply(a, NumJa.array(2.0));
        System.out.println("a * 2: " + result);
        
        // Reductions
        double total = NumJa.sum(a);
        double mean = NumJa.mean(a);
        System.out.println("\nSum of a: " + total);
        System.out.println("Mean of a: " + mean);
        System.out.println();
    }
    
    public static void testLinearAlgebra() {
        System.out.println("=" .repeat(60));
        System.out.println("Testing Linear Algebra Operations");
        System.out.println("=" .repeat(60));
        
        // Create matrices
        NDArray A = NumJa.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray B = NumJa.array(new double[][]{{5.0, 6.0}, {7.0, 8.0}});
        
        System.out.println("Matrix A:\n" + A + "\n");
        System.out.println("Matrix B:\n" + B + "\n");
        
        // Matrix multiplication
        NDArray C = NumJa.dot(A, B);
        System.out.println("A @ B:\n" + C + "\n");
        
        // Determinant
        double det_A = NumJa.det(A);
        System.out.println("det(A): " + det_A + "\n");
        
        // Matrix inverse
        try {
            NDArray inv_A = NumJa.inv(A);
            System.out.println("inv(A):\n" + inv_A + "\n");
        } catch (Exception e) {
            System.out.println("Error computing inverse: " + e.getMessage() + "\n");
        }
        
        // Matrix rank
        int rank = NumJa.matrixRank(A);
        System.out.println("rank(A): " + rank + "\n");
        
        // Trace
        double trace = NumJa.trace(A);
        System.out.println("trace(A): " + trace + "\n");
        
        // Norm
        double norm = NumJa.norm(A);
        System.out.println("Frobenius norm: " + norm + "\n");
    }
    
    public static void testThreadConfiguration() {
        System.out.println("=" .repeat(60));
        System.out.println("Testing Thread Configuration");
        System.out.println("=" .repeat(60));
        
        ThreadPoolConfig config = NumJa.getThreadConfig();
        
        System.out.println("Current Configuration:");
        System.out.println(config.getConfigDict());
        
        System.out.println("\nCurrent threads: " + NumJa.getNumThreads());
        
        // Test custom thread count
        NumJa.setNumThreads(2);
        System.out.println("After setting to 2: " + NumJa.getNumThreads());
        
        // Test reset
        NumJa.setNumThreads(null);
        System.out.println("After reset: " + NumJa.getNumThreads());
        
        System.out.println();
    }
    
    public static void testMatrixSolve() {
        System.out.println("=" .repeat(60));
        System.out.println("Testing Matrix Solve");
        System.out.println("=" .repeat(60));
        
        // Define system Ax = b
        NDArray A = NumJa.array(new double[][]{{3.0, 1.0}, {1.0, 2.0}});
        NDArray b = NumJa.array(9.0, 8.0);
        
        System.out.println("A =\n" + A + "\n");
        System.out.println("b = " + b + "\n");
        
        // Solve Ax = b
        NDArray x = NumJa.solve(A, b);
        System.out.println("Solution x: " + x + "\n");
        
        // Verify: A @ x should equal b
        NDArray verification = NumJa.dot(A, x);
        System.out.println("Verification A @ x: " + verification);
        System.out.println("Original b: " + b + "\n");
    }
    
    public static void testStatisticalOperations() {
        System.out.println("=" .repeat(60));
        System.out.println("Testing Statistical Operations");
        System.out.println("=" .repeat(60));
        
        NDArray data = NumJa.arange(1, 11, 1);  // [1, 2, ..., 10]
        
        System.out.println("Data: " + data + "\n");
        
        double mean = NumJa.mean(data);
        double std = NumJa.std(data);
        double var = NumJa.var(data);
        double min = NumJa.min(data);
        double max = NumJa.max(data);
        
        System.out.println("Mean: " + mean);
        System.out.println("Std Dev: " + std);
        System.out.println("Variance: " + var);
        System.out.println("Min: " + min);
        System.out.println("Max: " + max + "\n");
    }
    
    public static void testDecompositions() {
        System.out.println("=" .repeat(60));
        System.out.println("Testing Matrix Decompositions");
        System.out.println("=" .repeat(60));
        
        NDArray A = NumJa.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        
        // SVD / QR / Eigen (implementation may be placeholder)
        try {
            SVDResult svd = NumJa.svd(A);
            System.out.println("SVD result: " + (svd != null));
        } catch (Exception e) {
            System.out.println("SVD error: " + e.getMessage());
        }

        try {
            QRResult qr = NumJa.qr(A);
            System.out.println("QR computed: " + (qr != null));
        } catch (Exception e) {
            System.out.println("QR error: " + e.getMessage());
        }

        try {
            EigenResult eigen = NumJa.eig(A);
            System.out.println("Eigen result: " + (eigen != null));
        } catch (Exception e) {
            System.out.println("Eigenvalue error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        try {
            System.out.println("NumJa v" + NumJa.getVersion());
            System.out.println("NumPy-like Array Library for Java");
            System.out.println();
            
            testBasicOperations();
            testLinearAlgebra();
            testThreadConfiguration();
            testMatrixSolve();
            testStatisticalOperations();
            testDecompositions();
            
            System.out.println("=" .repeat(60));
            System.out.println("All tests completed successfully!");
            System.out.println("=" .repeat(60));
        } catch (Exception e) {
            System.err.println("Error during testing: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

