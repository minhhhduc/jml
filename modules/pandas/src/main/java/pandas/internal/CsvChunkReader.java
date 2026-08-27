package pandas.internal;

import pandas.DataFrame;
import pandas.Series;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * Streaming CSV reader that yields one {@link pandas.DataFrame} per chunk via
 * {@link Iterator} and releases the underlying file handle on
 * {@link #close()}. Mirrors {@code Pandas.read_csv} parsing semantics
 * (NaN-on-bad-numeric, sep split, skiprows, index_col, header row).
 *
 * <p>Usage (always try-with-resources to release the file handle):
 * <pre>{@code
 * try (CsvChunkReader r = new CsvChunkReader(path, opts)) {
 *     while (r.hasNext()) {
 *         DataFrame chunk = r.next();
 *         // ...
 *     }
 * }
 * }</pre>
 *
 * <p>Trust boundary: filesystem (caller-controlled {@link Path}) → in-process
 * {@code DataFrame}. STRIDE threats:
 * <ul>
 *   <li>T401 path traversal — mitigated by caller-side {@code path.normalize()}
 *       (Wave-2 {@code Pandas.read_csv_streaming} responsibility</li>
 *   <li>T403 file handle leak — mitigated by {@link AutoCloseable} + the
 *       try-with-resources contract documented above; verified by
 *       {@code CsvChunkReaderTest.close_releasesFileHandle}.</li>
 *   <li>T405 OOM via pathologically large {@code chunkRows} — mitigated by
 *       {@link ChunkedReadOptions#validate()} ceiling</li>
 *   <li>T406 CSV formula injection — mitigated by leading-char NaN
 *       substitution (cells starting with {@code = + - @} become {@link Double#NaN}
 *       with the original string preserved on the {@link Series}).</li>
 *</ul>
 *
 * @since ASVS-L1
 */
public final class CsvChunkReader implements Iterator<DataFrame>, AutoCloseable {

    private final BufferedReader br;
    private final ChunkedReadOptions opts;
    private final String[] finalColumns;
    private final int targetColsCount;
    private final boolean hasIndexCol;
    private final int indexCol;

    // WR-05: explicit identity init (Java defaults are correct here, but
    // being explicit guards against future refactors that change field types).
    private DataFrame next = null;
    private boolean exhausted = false;
    private boolean closed = false;

    public CsvChunkReader(Path path, ChunkedReadOptions opts) throws IOException {
        if (path == null) throw new IllegalArgumentException("path must not be null");
        if (opts == null) throw new IllegalArgumentException("opts must not be null");

        opts.validate();

        this.opts = opts;
        this.hasIndexCol = opts.index_col != null;
        this.indexCol = hasIndexCol ? opts.index_col : -1;

        this.br = Files.newBufferedReader(path, StandardCharsets.UTF_8);

        // Skip skiprows lines.
        for (int i = 0; i < opts.skiprows; i++) {
            if (br.readLine() == null) break;
        }

        // Read header (first non-blank remaining line).
        String headerLine = null;
        while (headerLine == null) {
            String line = br.readLine();
            if (line == null) break;
            if (!line.trim().isEmpty()) {
                headerLine = line;
            }
        }

        if (headerLine == null) {
            throw new IllegalArgumentException("File CSV rong hoac bi skip het data.");
        }

        String[] columns = headerLine.split(opts.sep);
        for (int i = 0; i < columns.length; i++) columns[i] = columns[i].trim();

        if (hasIndexCol && indexCol >= columns.length) {
            throw new IllegalArgumentException(
                "index_col=" + indexCol + " out of range for " + columns.length + " columns at " + path);
        }

        if (hasIndexCol) {
            this.targetColsCount = columns.length - 1;
            this.finalColumns = new String[targetColsCount];
            int ci = 0;
            for (int c = 0; c < columns.length; c++) {
                if (c != indexCol) finalColumns[ci++] = columns[c];
            }
        } else {
            this.targetColsCount = columns.length;
            this.finalColumns = columns;
        }

        advance();
        if (exhausted) {
            throw new IllegalArgumentException("File CSV rong hoac bi skip het data.");
        }
    }

    private void advance() {
        if (exhausted) return;

        // ponytail: per-chunk Series rebuilt from scratch — recycling ArrayLists
        // across chunks adds complexity not worth saving one allocation.
        List<double[]> rows = new ArrayList<>(opts.chunkRows);
        List<String[]> strRows = new ArrayList<>(opts.chunkRows);
        List<String> idxList = new ArrayList<>(opts.chunkRows);
        boolean[] isStringCol = new boolean[targetColsCount];

        int localRowIndex = 0;
        try {
            while (rows.size() < opts.chunkRows) {
                String line = br.readLine();
                if (line == null) break;
                if (line.trim().isEmpty()) continue;

                String[] cells = line.split(opts.sep);
                double[] rowData = new double[targetColsCount];
                String[] rowStr = new String[targetColsCount];

                String idx = null;
                int targetColCount = 0;
                for (int c = 0; c < cells.length && targetColCount < targetColsCount; c++) {
                    if (hasIndexCol && c == indexCol) {
                        if (idx == null) idx = cells[c].trim();
                        continue;
                    }
                    String val = cells[c].trim();
                    // T406: leading-char formula substitution -> NaN.
                    boolean formula = !val.isEmpty()
                        && "+-@=".indexOf(val.charAt(0)) >= 0;
                    if (formula) {
                        rowData[targetColCount] = Double.NaN;
                        rowStr[targetColCount] = val;
                        isStringCol[targetColCount] = true;
                    } else {
                        try {
                            rowData[targetColCount] = Double.parseDouble(val);
                        } catch (NumberFormatException e) {
                            rowData[targetColCount] = Double.NaN;
                            rowStr[targetColCount] = val;
                            isStringCol[targetColCount] = true;
                        }
                    }
                    targetColCount++;
                }

                idxList.add(hasIndexCol && idx != null ? idx : String.valueOf(localRowIndex));
                rows.add(rowData);
                strRows.add(rowStr);
                localRowIndex++;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        if (rows.isEmpty()) {
            exhausted = true;
            next = null;
            return;
        }

        int dataRows = rows.size();
        String[][] stringData = new String[targetColsCount][dataRows];
        for (int r = 0; r < dataRows; r++) {
            String[] rowStr = strRows.get(r);
            for (int c = 0; c < targetColsCount; c++) {
                if (isStringCol[c]) stringData[c][r] = rowStr[c];
            }
        }

        String[] index = idxList.toArray(new String[0]);

        Map<String, Series> seriesMap = new LinkedHashMap<>();
        for (int c = 0; c < targetColsCount; c++) {
            double[] colData = new double[dataRows];
            for (int r = 0; r < dataRows; r++) colData[r] = rows.get(r)[c];
            Series s = isStringCol[c]
                ? new Series(finalColumns[c], colData, stringData[c], index)
                : new Series(finalColumns[c], colData, index);
            seriesMap.put(finalColumns[c], s);
        }

        next = new DataFrame(seriesMap);
    }

    @Override
    public boolean hasNext() {
        return !exhausted;
    }

    @Override
    public DataFrame next() {
        if (exhausted) throw new NoSuchElementException("No more chunks.");
        DataFrame current = next;
        advance();
        return current;
    }

    @Override
    public void close() throws IOException {
        if (!closed && br != null) {
            br.close();
            closed = true;
        }
    }
}
