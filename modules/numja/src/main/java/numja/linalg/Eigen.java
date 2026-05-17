package numja.linalg;

import numja.core.NDArray;

/**
 * Result of eigenvalue decomposition
 * For a matrix A, computes eigenvalues λ and eigenvectors v such that A*v = λ*v
 */
public class Eigen {
    private final double[] eigenvalues;
    private final NDArray eigenvectors;
    
    /**
     * Create eigenvalue decomposition result
     * @param eigenvalues array of eigenvalues
     * @param eigenvectors matrix where columns are eigenvectors
     */
    public Eigen(double[] eigenvalues, NDArray eigenvectors) {
        this.eigenvalues = eigenvalues;
        this.eigenvectors = eigenvectors;
    }
    
    /**
     * Get the eigenvalues
     * @return array of eigenvalues
     */
    public double[] getEigenvalues() {
        return eigenvalues;
    }
    
    /**
     * Get the eigenvectors matrix
     * Each column is an eigenvector corresponding to an eigenvalue
     * @return matrix of eigenvectors
     */
    public NDArray getEigenvectors() {
        return eigenvectors;
    }
}

