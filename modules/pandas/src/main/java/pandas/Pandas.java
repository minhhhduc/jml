package pandas;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Pandas {

    // ===== FACTORY =====
    public static DataFrame DataFrame(double[][] data, String[] columns) {
        return new DataFrame(data, columns, defaultIndex(data.length));
    }

    // ===== READ CSV =====
    public static class ReadCsvOptions {
        public String sep = ",";
        public Integer header = 0;
        public String[] names = null;
        public int skiprows = 0;
        public Integer index_col = null;

        public ReadCsvOptions sep(String sep) { this.sep = sep; return this; }
        public ReadCsvOptions header(Integer header) { this.header = header; return this; }
        public ReadCsvOptions names(String... names) { this.names = names; return this; }
        public ReadCsvOptions skiprows(int skip) { this.skiprows = skip; return this; }
        public ReadCsvOptions index_col(Integer index_col) { this.index_col = index_col; return this; }
    }

    public static DataFrame read_csv(String filepath) throws IOException {
        return read_csv(filepath, new ReadCsvOptions());
    }

    public static DataFrame read_csv(String filepath, ReadCsvOptions options) throws IOException {
        List<String[]> lines = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            int currentRow = 0;
            while ((line = br.readLine()) != null) {
                if (currentRow < options.skiprows) { currentRow++; continue; }
                if (!line.trim().isEmpty()) lines.add(line.split(options.sep));
                currentRow++;
            }
        }

        if (lines.isEmpty()) throw new IllegalArgumentException("File CSV rong hoac bi skip het data.");

        int dataStartIndex = 0;
        int cols = lines.get(0).length;
        String[] columns;

        if (options.header != null) {
            columns = lines.get(options.header);
            dataStartIndex = options.header + 1;
        } else {
            columns = new String[cols];
            for (int i = 0; i < cols; i++) columns[i] = String.valueOf(i);
        }

        if (options.names != null) { columns = options.names; cols = options.names.length; }

        int dataRows = lines.size() - dataStartIndex;
        boolean hasIndexCol = options.index_col != null;
        int targetColsCount = hasIndexCol ? cols - 1 : cols;

        double[][] data = new double[dataRows][targetColsCount];
        String[][] stringData = new String[targetColsCount][dataRows];
        boolean[] isStringCol = new boolean[targetColsCount];
        String[] index = new String[dataRows];
        String[] finalColumns = hasIndexCol ? new String[targetColsCount] : columns;

        if (hasIndexCol) {
            int cIndex = 0;
            for (int c = 0; c < cols; c++) {
                if (c != options.index_col && cIndex < targetColsCount) finalColumns[cIndex++] = columns[c];
            }
        }

        for (int r = 0; r < dataRows; r++) {
            String[] rowVals = lines.get(dataStartIndex + r);
            if (hasIndexCol && options.index_col < rowVals.length) index[r] = rowVals[options.index_col].trim();
            else index[r] = String.valueOf(r);

            int targetColCount = 0;
            for (int c = 0; c < Math.min(rowVals.length, cols); c++) {
                if (hasIndexCol && c == options.index_col) continue;
                if (targetColCount < targetColsCount) {
                    String val = rowVals[c].trim();
                    try { data[r][targetColCount] = Double.parseDouble(val); }
                    catch (NumberFormatException e) {
                        data[r][targetColCount] = Double.NaN;
                        stringData[targetColCount][r] = val;
                        isStringCol[targetColCount] = true;
                    }
                    targetColCount++;
                }
            }
        }

        Map<String, Series> seriesMap = new LinkedHashMap<>();
        for (int c = 0; c < targetColsCount; c++) {
            double[] colDoubles = new double[dataRows];
            for (int r = 0; r < dataRows; r++) colDoubles[r] = data[r][c];
            Series s = isStringCol[c]
                ? new Series(finalColumns[c], colDoubles, stringData[c], index)
                : new Series(finalColumns[c], colDoubles, index);
            seriesMap.put(finalColumns[c], s);
        }

        return new DataFrame(seriesMap);
    }

    // ===== CUT - Bin continuous values into discrete intervals =====
    /**
     * Bin values into discrete intervals (equal-width bins)
     * @param data numeric values to bin
     * @param bins number of bins
     * @return Series with bin labels
     */
    public static Series cut(double[] data, int bins) {
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : data) {
            if (!Double.isNaN(v)) { if (v < min) min = v; if (v > max) max = v; }
        }
        double width = (max - min) / bins;

        double[] binEdges = new double[bins + 1];
        for (int i = 0; i <= bins; i++) binEdges[i] = min + i * width;
        return cutByEdges(data, binEdges);
    }

    /**
     * Bin values using custom bin edges
     * @param data numeric values to bin
     * @param binEdges array of bin edges (n+1 edges for n bins)
     * @return Series with bin labels
     */
    public static Series cut(double[] data, double[] binEdges) {
        return cutByEdges(data, binEdges);
    }

    private static Series cutByEdges(double[] data, double[] binEdges) {
        int bins = binEdges.length - 1;
        double[] binIndices = new double[data.length];
        String[] binLabels = new String[data.length];

        for (int i = 0; i < data.length; i++) {
            if (Double.isNaN(data[i])) {
                binIndices[i] = Double.NaN;
                binLabels[i] = "NaN";
                continue;
            }
            int binIdx = bins - 1; // default to last bin
            for (int b = 0; b < bins; b++) {
                if (data[i] <= binEdges[b + 1]) { binIdx = b; break; }
            }
            binIndices[i] = binIdx;
            binLabels[i] = String.format("(%.2f, %.2f]", binEdges[binIdx], binEdges[binIdx + 1]);
        }

        return new Series(null, binIndices, binLabels, defaultIndex(data.length));
    }

    // ===== QCUT - Quantile-based binning =====
    /**
     * Quantile-based binning (equal-frequency bins)
     * @param data numeric values
     * @param q number of quantiles
     * @return Series with quantile labels
     */
    public static Series qcut(double[] data, int q) {
        double[] sorted = data.clone();
        Arrays.sort(sorted);

        double[] edges = new double[q + 1];
        edges[0] = sorted[0];
        edges[q] = sorted[sorted.length - 1];
        for (int i = 1; i < q; i++) {
            double p = (double) i / q;
            int idx = (int) Math.round(p * (sorted.length - 1));
            edges[i] = sorted[idx];
        }

        return cutByEdges(data, edges);
    }

    // ===== GET_DUMMIES - One-hot encoding for a Series =====
    /**
     * Convert categorical variable into dummy/indicator variables
     * @param series the categorical series
     * @return DataFrame with one column per unique value
     */
    public static DataFrame get_dummies(Series series) {
        String[] uniqueVals = series.uniqueStrings();
        Arrays.sort(uniqueVals);

        int n = series.size();
        double[][] data = new double[n][uniqueVals.length];
        Map<String, Integer> valIdx = new LinkedHashMap<>();
        for (int i = 0; i < uniqueVals.length; i++) valIdx.put(uniqueVals[i], i);

        for (int i = 0; i < n; i++) {
            String val = series.getString(i);
            Integer idx = valIdx.get(val);
            if (idx != null) data[i][idx] = 1.0;
        }

        // Column names: prefix_value
        String prefix = series.getName() != null ? series.getName() + "_" : "";
        String[] colNames = new String[uniqueVals.length];
        for (int i = 0; i < uniqueVals.length; i++) colNames[i] = prefix + uniqueVals[i];

        return new DataFrame(data, colNames, series.getIndex());
    }

    // ===== CONCAT - Concatenate DataFrames =====
    /**
     * Concatenate DataFrames vertically (row-wise)
     */
    public static DataFrame concat(DataFrame... dfs) {
        if (dfs.length == 0) throw new IllegalArgumentException("No DataFrames to concatenate.");

        String[] columns = dfs[0].columns();
        int totalRows = 0;
        for (DataFrame df : dfs) totalRows += df.shape()[0];

        double[][] data = new double[totalRows][columns.length];
        String[] newIndex = new String[totalRows];
        int row = 0;

        for (DataFrame df : dfs) {
            int[] shape = df.shape();
            for (int r = 0; r < shape[0]; r++) {
                newIndex[row] = String.valueOf(row);
                for (int c = 0; c < columns.length; c++) {
                    try { data[row][c] = df.getColumn(columns[c]).get(r); }
                    catch (Exception e) { data[row][c] = Double.NaN; }
                }
                row++;
            }
        }

        return new DataFrame(data, columns, newIndex);
    }

    public static DataFrame read_excel(String filepath) {
        throw new UnsupportedOperationException("Chua ho tro. Can thu vien Apache POI.");
    }

    private static String[] defaultIndex(int length) {
        String[] idx = new String[length];
        for (int i = 0; i < length; i++) idx[i] = String.valueOf(i);
        return idx;
    }
}
