package seaborn;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertTrue;

public class SeabornTest {

    @Test
    public void plotScatter_createsFile() throws Exception {
        double[] x = {0.0, 1.0, 2.0, 3.0};
        double[] y = {0.0, 1.0, 4.0, 9.0};
        File out = File.createTempFile("seaborn-test-", ".png");
        out.deleteOnExit();

        Seaborn.plotScatter(x, y, "scatter-test", out.getAbsolutePath());

        assertTrue("Output PNG should exist", out.exists());
        assertTrue("Output PNG should not be empty", out.length() > 0);
    }

    @Test
    public void plotLine_createsFile() throws Exception {
        double[] x = {0.0, 1.0, 2.0};
        double[] y = {1.0, 2.0, 3.0};
        File out = File.createTempFile("seaborn-line-", ".png");
        out.deleteOnExit();

        Seaborn.plotLine(x, y, "line-test", out.getAbsolutePath());

        assertTrue("Output PNG should exist", out.exists());
        assertTrue("Output PNG should not be empty", out.length() > 0);
    }
}


