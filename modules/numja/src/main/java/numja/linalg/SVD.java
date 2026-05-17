package numja.linalg;

import numja.core.NDArray;

/**
 * Result of Singular Value Decomposition (SVD)
 * Decomposes a matrix A into A = U * S * Vt
 * where U and Vt are orthogonal matrices and S contains singular values
 */
public class SVD {
    private final NDArray U;
    private final double[] singularValues;
    private final NDArray Vt;
    
    /**
     * Create SVD result
     * @param U left singular vectors matrix
     * @param singularValues array of singular values
     * @param Vt transpose of right singular vectors matrix
     */
    public SVD(NDArray U, double[] singularValues, NDArray Vt) {
        this.U = U;
        this.singularValues = singularValues;
        this.Vt = Vt;
    }
    
    /**
     * Get the U matrix (left singular vectors)
     * @return U matrix
     */
    public NDArray getU() {
        return U;
    }
    
    /**
     * Get the singular values
     * @return array of singular values in descending order
     */
    public double[] getSingularValues() {
        return singularValues;
    }
    
    /**
     * Get the transpose of V matrix (right singular vectors)
     * @return Vt matrix
     */
    public NDArray getV() {
        return Vt;
    }
}

