package sklearn.pipeline;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import sklearn.naive_bayes.GaussianNB;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Integration tests for {@link PandasPipeline} — the 4-line caller
 * pattern (USE-02) plus IAE guards for missing load/partialFit calls.
 *
 * <p>Trust boundary: filesystem (caller-controlled Path) → in-process
 * GaussianNB. STRIDE-T401/T403 mitigated by try-with-resources +
 * Files.isReadable inside the reader</p>
 */
public class PandasPipelineTest {

    private Path tempCsv;
    private static final long SEED = 20260828L;

    @Before
    public void setUp() throws IOException {
        tempCsv = Files.createTempFile("phase4_pipe_", ".csv");
        Random rng = new Random(SEED);
        StringBuilder sb = new StringBuilder("f1,f2,label\n");
        for (int i = 0; i < 30; i++) {
            double f1 = rng.nextGaussian();
            double f2 = rng.nextGaussian();
            int label = (f1 + f2 > 0) ? 1 : 0;
            sb.append(f1).append(',').append(f2).append(',').append(label).append('\n');
        }
        Files.write(tempCsv, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    @After
    public void tearDown() throws IOException {
        if (tempCsv != null && Files.exists(tempCsv)) Files.delete(tempCsv);
    }

    @Test
    public void tenLineCallerPattern() throws IOException {
        GaussianNB model = new PandasPipeline()
            .load(Paths.get(tempCsv.toString()))
            .chunk(10)
            .partialFit(new GaussianNB())
            .labelColumn("label")
            .run();

        assertNotNull("Model must not be null", model);
        double[][] X = {{0.5, -0.3}, {-0.7, 0.4}, {0.1, 0.2}};
        int[] pred = model.predict(new numja.core.NDArray(X));
        assertEquals("3-row predict returns 3 predictions", 3, pred.length);
    }

    @Test
    public void missingPath_throwsIAE() throws IOException {
        try {
            new PandasPipeline()
                .partialFit(new GaussianNB())
                .labelColumn("label")
                .run();
            fail("run() without load() must throw IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(
                "Error message must mention load(path), got: " + e.getMessage(),
                e.getMessage().contains("load(path) required"));
        }
    }

    @Test
    public void missingEstimator_throwsIAE() throws IOException {
        try {
            new PandasPipeline()
                .load(Paths.get(tempCsv.toString()))
                .labelColumn("label")
                .run();
            fail("run() without partialFit() must throw IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(
                "Error message must mention partialFit, got: " + e.getMessage(),
                e.getMessage().contains("partialFit(estimator) required"));
        }
    }

    @Test
    public void modelPredictsAfterRun() throws IOException {
        Path bigCsv = Files.createTempFile("phase4_pipe_big_", ".csv");
        try {
            Random rng = new Random(SEED + 1);
            StringBuilder sb = new StringBuilder("f1,f2,label\n");
            for (int i = 0; i < 100; i++) {
                double f1 = rng.nextGaussian();
                double f2 = rng.nextGaussian();
                int label = (f1 + f2 > 0) ? 1 : 0;
                sb.append(f1).append(',').append(f2).append(',').append(label).append('\n');
            }
            Files.write(bigCsv, sb.toString().getBytes(StandardCharsets.UTF_8));

            GaussianNB model = new PandasPipeline()
                .load(Paths.get(bigCsv.toString()))
                .chunk(25)
                .partialFit(new GaussianNB())
                .labelColumn("label")
                .run();

            String text = new String(Files.readAllBytes(bigCsv), StandardCharsets.UTF_8);
            String[] lines = text.split("\n");
            double[][] X = new double[lines.length - 1][2];
            int[] y = new int[lines.length - 1];
            for (int i = 1; i < lines.length; i++) {
                String[] parts = lines[i].split(",");
                X[i - 1][0] = Double.parseDouble(parts[0]);
                X[i - 1][1] = Double.parseDouble(parts[1]);
                y[i - 1] = Integer.parseInt(parts[2]);
            }
            double acc = model.score(new numja.core.NDArray(X), y);
            assertTrue("Streaming-fit accuracy must beat 0.6 on linearly-separable data, got " + acc,
                acc > 0.6);
        } finally {
            Files.deleteIfExists(bigCsv);
        }
    }
}
