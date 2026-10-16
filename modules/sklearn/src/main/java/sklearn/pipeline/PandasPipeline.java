package sklearn.pipeline;

import numja.core.NDArray;
import pandas.DataFrame;
import pandas.Series;
import pandas.internal.ChunkedReadOptions;
import pandas.internal.CsvChunkReader;
import sklearn.naive_bayes.GaussianNB;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Fluent builder that streams a CSV through {@link GaussianNB#partial_fit}
 * chunk by chunk and calls {@link GaussianNB#finalize_fit} on completion.
 * Implements USE-02's 4-line caller pattern:
 * <pre>{@code
 * GaussianNB model = new PandasPipeline()
 *     .load(Paths.get("big.csv"))
 *     .chunk(10_000)
 *     .partialFit(new GaussianNB())
 *     .labelColumn("label")
 *     .run();
 * }</pre>
 *
 * <p>Trust boundary: filesystem (caller-controlled path) → in-process
 * GaussianNB. STRIDE threats:
 * <ul>
 *   <li>T401 path traversal — mitigated by {@code Files.isReadable()}
 *       check inside {@link CsvChunkReader</li>
 *   <li>T403 file handle leak — mitigated by try-with-resources on the
 *       reader; verified by repeated test runs (handle released</li>
 *</ul>
 *
 * @since ASVS-L1
 */
public final class PandasPipeline {
    private Path path;
    private final ChunkedReadOptions opts = new ChunkedReadOptions();
    private GaussianNB estimator;
    private String labelColumn = null;

    public PandasPipeline load(Path p) { this.path = p; return this; }
    public PandasPipeline chunk(int rows) { this.opts.chunkRows = rows; return this; }
    public PandasPipeline partialFit(GaussianNB e) { this.estimator = e; return this; }
    // ponytail: hardcoded GaussianNB typing — generalize to a PartialFitCapable
    // interface once a 2nd model gains partial_fit.
    public PandasPipeline labelColumn(String name) { this.labelColumn = name; return this; }

    public GaussianNB run() throws IOException {
        if (path == null) throw new IllegalArgumentException("load(path) required.");
        if (estimator == null) throw new IllegalArgumentException("partialFit(estimator) required.");
        if (labelColumn == null) throw new IllegalArgumentException("labelColumn(name) required.");

        try (CsvChunkReader reader = new CsvChunkReader(path, opts)) {
            while (reader.hasNext()) {
                DataFrame chunk = reader.next();
                String[] cols = chunk.columns();
                List<String> featureCols = new ArrayList<>(cols.length - 1);
                for (String c : cols) if (!c.equals(labelColumn)) featureCols.add(c);

                int nRows = chunk.shape()[0];
                double[][] Xarr = new double[nRows][featureCols.size()];
                int[] y = new int[nRows];
                Series labelSeries = chunk.getColumn(labelColumn);
                for (int i = 0; i < nRows; i++) y[i] = (int) Math.round(labelSeries.get(i));

                for (int fc = 0; fc < featureCols.size(); fc++) {
                    Series s = chunk.getColumn(featureCols.get(fc));
                    for (int i = 0; i < nRows; i++) Xarr[i][fc] = s.get(i);
                }

                estimator.partial_fit(new NDArray(Xarr), y);
            }
        }
        estimator.finalize_fit();
        return estimator;
    }
}
