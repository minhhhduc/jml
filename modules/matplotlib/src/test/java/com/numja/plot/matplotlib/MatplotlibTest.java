package matplotlib;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertTrue;

public class MatplotlibTest {

    @Test
    public void plotLine_createsFile() throws Exception {
        double[] x = {0.0, 1.0, 2.0, 3.0};
        double[] y = {0.0, 1.0, 4.0, 9.0};
        File out = File.createTempFile("matplotlib-test-", ".png");
        out.deleteOnExit();

        Matplotlib.plot(x, y, "test");
        Matplotlib.savefig(out.getAbsolutePath());

        assertTrue("Output PNG should exist", out.exists());
        assertTrue("Output PNG should not be empty", out.length() > 0);
    }
}


