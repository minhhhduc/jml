package pandas;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class DataFrameTest {
    @Test
    public void headAndShape() {
        double[][] data = new double[][]{{1,2,3},{4,5,6},{7,8,9}};
        DataFrame df = new DataFrame(data);
        assertArrayEquals(new int[]{3,3}, df.shape());

        DataFrame h = df.head(2);
        assertArrayEquals(new int[]{2,3}, h.shape());
    }
}


