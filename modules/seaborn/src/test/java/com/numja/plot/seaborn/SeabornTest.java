package seaborn;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertTrue;

public class SeabornTest {

    private static void assertPngCreated(File out) throws Exception {
        matplotlib.Matplotlib.savefig(out.getAbsolutePath());
        assertTrue("Output PNG should exist", out.exists());
        assertTrue("Output PNG should not be empty", out.length() > 0);
    }

    private static pandas.DataFrame toFrame(double[] x, double[] y) {
        // DataFrame rows are observations: each row is one (x, y) point
        double[][] rows = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            rows[i][0] = x[i];
            rows[i][1] = y[i];
        }
        return pandas.Pandas.DataFrame(rows, new String[]{"x", "y"});
    }

    @Test
    public void scatter_createsFile() throws Exception {
        double[] x = {0.0, 1.0, 2.0, 3.0};
        double[] y = {0.0, 1.0, 4.0, 9.0};
        File out = File.createTempFile("seaborn-test-", ".png");
        out.deleteOnExit();

        Seaborn.scatter(toFrame(x, y), "x", "y");
        assertPngCreated(out);
    }

    @Test
    public void plot_createsFile() throws Exception {
        double[] x = {0.0, 1.0, 2.0};
        double[] y = {1.0, 2.0, 3.0};
        File out = File.createTempFile("seaborn-line-", ".png");
        out.deleteOnExit();

        matplotlib.Matplotlib.clf();
        Seaborn.plot(toFrame(x, y), "x", "y");
        assertPngCreated(out);
    }
}
