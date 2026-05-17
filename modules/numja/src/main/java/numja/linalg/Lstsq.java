package numja.linalg;

import numja.core.NDArray;

/**
 * Result of least squares solution
 * Solves the overdetermined system Ax ≈ b when A is m×n (m > n)
 */
public class Lstsq {
    private final NDArray solution;
    private final NDArray residuals;
    private final int rank;
    private final double[] singularValues;

    /**
     * Create least squares result
     * @param solution the solution vector x that minimizes ||Ax - b||
     * @param residuals residuals of the fit
     * @param rank rank of the coefficient matrix
     * @param singularValues singular values from SVD decomposition
     */
    public Lstsq(NDArray solution, NDArray residuals, int rank, double[] singularValues) {
        this.solution = solution;
        this.residuals = residuals;
        this.rank = rank;
        this.singularValues = singularValues;
    }

    /**
     * Get the least squares solution vector
     * @return solution vector x
     */
    public NDArray getSolution() {
        return solution;
    }

    /**
     * Get the residuals (if system is not exactly determined)
     * @return residuals array
     */
    public NDArray getResiduals() {
        return residuals;
    }

    /**
     * Get the rank of the coefficient matrix
     * @return matrix rank
     */
    public int getRank() {
        return rank;
    }

    /**
     * Get the singular values from the decomposition
     * @return array of singular values
     */
    public double[] getSingularValues() {
        return singularValues;
    }
}

