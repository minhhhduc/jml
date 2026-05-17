package numja.linalg;

import numja.core.NDArray;

/**
 * Result of QR decomposition
 * Decomposes a matrix A into A = Q * R
 * where Q is orthogonal and R is upper triangular
 */
public class QR {
    private final NDArray Q;
    private final NDArray R;
    
    /**
     * Create QR decomposition result
     * @param Q orthogonal matrix
     * @param R upper triangular matrix
     */
    public QR(NDArray Q, NDArray R) {
        this.Q = Q;
        this.R = R;
    }
    
    /**
     * Get the orthogonal matrix Q
     * @return Q matrix
     */
    public NDArray getQ() {
        return Q;
    }
    
    /**
     * Get the upper triangular matrix R
     * @return R matrix
     */
    public NDArray getR() {
        return R;
    }
}

