package numja.core;

import numja.core.NDArray;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NDArrayTest {
    @Test
    public void arithmeticAndReductions() {
        NDArray a = new NDArray(1.0, 2.0, 3.0);
        NDArray b = new NDArray(4.0, 5.0, 6.0);

        NDArray sum = a.add(b);
        assertEquals(5.0, sum.getData().get(0,0), 1e-9);
        assertEquals(7.0, sum.getData().get(1,0), 1e-9);

        NDArray prod = a.multiply(b);
        assertEquals(4.0, prod.getData().get(0,0), 1e-9);

        assertEquals(6.0, a.sum(), 1e-9);
        assertEquals(2.0, a.mean(), 1e-9);
    }
}

