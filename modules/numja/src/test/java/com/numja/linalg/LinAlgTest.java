package numja.linalg;

import numja.core.NDArray;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LinAlgTest {
    @Test
    public void detAndSolve() {
        NDArray A = new NDArray(new double[][]{{3.0,1.0},{1.0,2.0}});
        NDArray b = new NDArray(9.0, 8.0);

        double det = LinAlg.det(A);
        assertEquals(5.0, det, 1e-9);

        NDArray x = LinAlg.solve(A, b);
        // verify A @ x == b
        NDArray Ax = numja.NumJa.dot(A, x);
        assertEquals(b.getData().get(0,0), Ax.getData().get(0,0), 1e-8);
        assertEquals(b.getData().get(1,0), Ax.getData().get(1,0), 1e-8);
    }
}


