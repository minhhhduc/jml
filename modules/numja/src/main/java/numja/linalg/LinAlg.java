package numja.linalg;

import numja.core.NDArray;
import numja.config.ThreadPoolConfig;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.NormOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.*;

/**
 * Linear Algebra operations using EJML (BLAS/LAPACK equivalent)
 */
public class LinAlg {
    
    private static ThreadPoolConfig config = ThreadPoolConfig.getInstance();
    
    /**
     * Matrix inverse
     */
    public static NDArray inv(NDArray a) {
        DMatrixRMaj input = a.getData();
        if (input.numRows != input.numCols) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        
        DMatrixRMaj result = new DMatrixRMaj(input.numRows, input.numCols);
        if (!CommonOps_DDRM.invert(input, result)) {
            throw new RuntimeException("Matrix is singular and cannot be inverted");
        }
        
        return new NDArray(result, new int[]{input.numRows, input.numCols});
    }
    
    /**
     * Determinant
     */
    public static double det(NDArray a) {
        DMatrixRMaj input = a.getData();
        if (input.numRows != input.numCols) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        
        return CommonOps_DDRM.det(input);
    }
    
    /**
     * Matrix rank
     */
    public static int matrixRank(NDArray a) {
        DMatrixRMaj input = a.getData();
        
        SingularValueDecomposition_F64<DMatrixRMaj> svd = 
            DecompositionFactory_DDRM.svd(input.numRows, input.numCols, true, true, false);
        
        if (!svd.decompose(input)) {
            throw new RuntimeException("SVD decomposition failed");
        }
        
        double[] singularValues = svd.getSingularValues();
        double threshold = Math.max(input.numRows, input.numCols) * singularValues[0] * 1e-15;
        int rank = 0;
        
        for (int i = 0; i < svd.numberOfSingularValues(); i++) {
            if (singularValues[i] > threshold) {
                rank++;
            }
        }
        
        return rank;
    }

    /**
     * Python-style alias for matrix rank.
     */
    public static int matrix_rank(NDArray a) {
        return matrixRank(a);
    }
    
    /**
     * Matrix trace (sum of diagonal)
     */
    public static double trace(NDArray a) {
        DMatrixRMaj input = a.getData();
        double trace = 0;
        int minDim = Math.min(input.numRows, input.numCols);
        
        for (int i = 0; i < minDim; i++) {
            trace += input.get(i, i);
        }
        
        return trace;
    }
    
    /**
     * Solve linear system Ax = b
     */
    public static NDArray solve(NDArray A, NDArray b) {
        DMatrixRMaj matrixA = A.getData();
        DMatrixRMaj vectorB = b.getData();
        
        if (matrixA.numRows != matrixA.numCols) {
            throw new IllegalArgumentException("A must be square");
        }
        if (matrixA.numRows != vectorB.numRows) {
            throw new IllegalArgumentException("Dimensions do not match");
        }
        
        DMatrixRMaj result = new DMatrixRMaj(vectorB.numRows, vectorB.numCols);
        
        if (!CommonOps_DDRM.solve(matrixA, vectorB, result)) {
            throw new RuntimeException("Matrix is singular");
        }
        
        return new NDArray(result, b.getShape());
    }

    /**
     * Least squares solution similar to SciPy's lstsq.
     */
    public static Lstsq lstsq(NDArray A, NDArray b) {
        DMatrixRMaj matrixA = A.getData();
        DMatrixRMaj vectorB = b.getData();

        SVD svd = svd(A);
        NDArray pseudoInverse = pinv(A);
        DMatrixRMaj pinvData = pseudoInverse.getData();
        DMatrixRMaj sol = new DMatrixRMaj(pinvData.numRows, vectorB.numCols);
        CommonOps_DDRM.mult(pinvData, vectorB, sol);

        double[] singularValues = svd.getSingularValues();
        int rank = 0;
        double threshold = Math.max(matrixA.numRows, matrixA.numCols) * singularValues[0] * 1e-15;
        for (double s : singularValues) {
            if (s > threshold) {
                rank++;
            }
        }

        NDArray solution = new NDArray(sol, new int[]{sol.numRows, sol.numCols});
        NDArray residuals = null;
        if (matrixA.numRows > matrixA.numCols) {
            DMatrixRMaj ax = new DMatrixRMaj(matrixA.numRows, vectorB.numCols);
            CommonOps_DDRM.mult(matrixA, sol, ax);
            DMatrixRMaj r = new DMatrixRMaj(matrixA.numRows, vectorB.numCols);
            CommonOps_DDRM.subtract(vectorB, ax, r);
            residuals = new NDArray(r, new int[]{r.numRows, r.numCols});
        }

        return new Lstsq(solution, residuals, rank, singularValues);
    }
    
    /**
     * Singular Value Decomposition
     * Returns U, S, V^T
     */
    public static SVD svd(NDArray a) {
        DMatrixRMaj input = a.getData();
        
        SingularValueDecomposition_F64<DMatrixRMaj> svd = 
            DecompositionFactory_DDRM.svd(input.numRows, input.numCols, true, true, false);
        
        if (!svd.decompose(input)) {
            throw new RuntimeException("SVD decomposition failed");
        }
        
        DMatrixRMaj U = svd.getU(null, false);
        DMatrixRMaj V = svd.getV(null, false);
        
        // Get singular values
        double[] singularValues = svd.getSingularValues();
        
        // V^T
        DMatrixRMaj Vt = new DMatrixRMaj(V.numCols, V.numRows);
        CommonOps_DDRM.transpose(V, Vt);
        
        return new SVD(
            new NDArray(U, new int[]{U.numRows, U.numCols}),
            singularValues,
            new NDArray(Vt, new int[]{Vt.numRows, Vt.numCols})
        );
    }
    
    /**
     * QR Decomposition
     * Returns Q, R
     */
    public static QR qr(NDArray a) {
        DMatrixRMaj input = a.getData();
        
        QRDecomposition<DMatrixRMaj> qr = 
            DecompositionFactory_DDRM.qr(input.numRows, input.numCols);
        
        if (!qr.decompose(input)) {
            throw new RuntimeException("QR decomposition failed");
        }
        
        DMatrixRMaj Q = qr.getQ(null, false);
        DMatrixRMaj R = qr.getR(null, false);
        
        return new QR(
            new NDArray(Q, new int[]{Q.numRows, Q.numCols}),
            new NDArray(R, new int[]{R.numRows, R.numCols})
        );
    }
    
    /**
     * Eigenvalue Decomposition
     * Returns eigenvalues and eigenvectors
     */
    public static Eigen eig(NDArray a) {
        DMatrixRMaj input = a.getData();
        
        if (input.numRows != input.numCols) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        
        EigenDecomposition_F64<DMatrixRMaj> eigen = 
            DecompositionFactory_DDRM.eig(input.numRows, true);
        
        if (!eigen.decompose(input)) {
            throw new RuntimeException("Eigenvalue decomposition failed");
        }
        
        int n = input.numRows;
        double[] realParts = new double[n];
        double[] imagParts = new double[n];
        
        for (int i = 0; i < n; i++) {
            realParts[i] = eigen.getEigenvalue(i).real;
            imagParts[i] = eigen.getEigenvalue(i).imaginary;
        }
        
        // For real matrices, we focus on real eigenvalues
        DMatrixRMaj eigenvectors = new DMatrixRMaj(n, n);
        for (int i = 0; i < n; i++) {
            DMatrixRMaj v = eigen.getEigenVector(i);
            if (v != null) {
                for (int j = 0; j < n; j++) {
                    eigenvectors.set(j, i, v.get(j, 0));
                }
            }
        }
        
        return new Eigen(
            realParts,
            new NDArray(eigenvectors, new int[]{n, n})
        );
    }
    
    /**
     * Cholesky Decomposition (for positive definite matrices)
     */
    public static NDArray cholesky(NDArray a) {
        DMatrixRMaj input = a.getData();
        
        if (input.numRows != input.numCols) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        
        CholeskyDecomposition_F64<DMatrixRMaj> chol = 
            DecompositionFactory_DDRM.chol(input.numRows, true);
        
        if (!chol.decompose(input)) {
            throw new RuntimeException("Matrix is not positive definite");
        }
        
        DMatrixRMaj result = chol.getT(null);
        return new NDArray(result, new int[]{result.numRows, result.numCols});
    }

    /**
     * Python-style alias for Cholesky decomposition.
     */
    public static NDArray cholesky(NDArray a, boolean lower) {
        NDArray result = cholesky(a);
        if (lower) {
            return result;
        }
        DMatrixRMaj upper = new DMatrixRMaj(result.getData().numRows, result.getData().numCols);
        CommonOps_DDRM.transpose(result.getData(), upper);
        return new NDArray(upper, new int[]{upper.numRows, upper.numCols});
    }
    
    /**
     * Frobenius norm
     */
    public static double norm(NDArray a) {
        return NormOps_DDRM.normF(a.getData());
    }
    
    /**
     * 1-norm (maximum absolute column sum)
     */
    public static double norm1(NDArray a) {
        DMatrixRMaj input = a.getData();
        double maxColSum = 0;
        
        for (int j = 0; j < input.numCols; j++) {
            double colSum = 0;
            for (int i = 0; i < input.numRows; i++) {
                colSum += Math.abs(input.get(i, j));
            }
            maxColSum = Math.max(maxColSum, colSum);
        }
        
        return maxColSum;
    }
    
    /**
     * Infinity norm (maximum absolute row sum)
     */
    public static double normInf(NDArray a) {
        DMatrixRMaj input = a.getData();
        double maxRowSum = 0;
        
        for (int i = 0; i < input.numRows; i++) {
            double rowSum = 0;
            for (int j = 0; j < input.numCols; j++) {
                rowSum += Math.abs(input.get(i, j));
            }
            maxRowSum = Math.max(maxRowSum, rowSum);
        }
        
        return maxRowSum;
    }
    
    /**
     * Pseudo-inverse (Moore-Penrose)
     */
    public static NDArray pinv(NDArray a) {
        DMatrixRMaj input = a.getData();
        
        // Use SVD for pseudo-inverse
        SVD svd = svd(a);
        DMatrixRMaj V = svd.getV().getData();
        DMatrixRMaj U = svd.getU().getData();
        double[] s = svd.getSingularValues();
        
        // Create diagonal matrix of 1/sigma
        DMatrixRMaj Sinv = new DMatrixRMaj(input.numCols, input.numRows);
        double threshold = Math.max(input.numRows, input.numCols) * s[0] * 1e-15;
        
        for (int i = 0; i < Math.min(s.length, input.numRows); i++) {
            if (s[i] > threshold) {
                Sinv.set(i, i, 1.0 / s[i]);
            }
        }
        
        // A^+ = V * Sigma^-1 * U^T
        DMatrixRMaj temp = new DMatrixRMaj(input.numCols, input.numRows);
        CommonOps_DDRM.mult(V, Sinv, temp);
        
        DMatrixRMaj result = new DMatrixRMaj(input.numCols, input.numRows);
        DMatrixRMaj Ut = new DMatrixRMaj(U.numCols, U.numRows);
        CommonOps_DDRM.transpose(U, Ut);
        CommonOps_DDRM.mult(temp, Ut, result);
        
        return new NDArray(result, new int[]{input.numCols, input.numRows});
    }
    
    /**
     * Matrix power (A^n for integer n)
     */
    public static NDArray matrixPower(NDArray a, int n) {
        DMatrixRMaj input = a.getData();
        
        if (input.numRows != input.numCols) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        
        if (n == 0) {
            DMatrixRMaj identity = CommonOps_DDRM.identity(input.numRows);
            return new NDArray(identity, new int[]{identity.numRows, identity.numCols});
        }
        
        DMatrixRMaj result = CommonOps_DDRM.identity(input.numRows);
        DMatrixRMaj base = new DMatrixRMaj(input);
        
        for (int i = 0; i < Math.abs(n); i++) {
            DMatrixRMaj temp = new DMatrixRMaj(input.numRows, input.numCols);
            CommonOps_DDRM.mult(result, base, temp);
            result = temp;
        }
        
        if (n < 0) {
            DMatrixRMaj inverse = new DMatrixRMaj(result.numRows, result.numCols);
            if (!CommonOps_DDRM.invert(result, inverse)) {
                throw new RuntimeException("Matrix is singular and cannot be inverted");
            }
            result = inverse;
        }
        
        return new NDArray(result, new int[]{result.numRows, result.numCols});
    }
}

