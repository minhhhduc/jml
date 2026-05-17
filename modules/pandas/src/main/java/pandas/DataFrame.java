package pandas;

import java.util.*;
import java.util.function.Function;
import java.io.PrintWriter;
import java.io.File;
import java.io.IOException;

/**
 * DataFrame - A 2D data structure for storing and manipulating tabular data
 * Similar to pandas.DataFrame in Python
 */
public class DataFrame {
    private final Map<String, Series> columnsData;
    private final String[] index;
    private final String[] columns;

    public DataFrame(double[][] data) {
        this(data, defaultCols(data.length > 0 ? data[0].length : 0), defaultIdx(data.length));
    }

    public DataFrame(double[][] data, String[] columns, String[] index) {
        if (data.length > 0 && data[0].length != columns.length)
            throw new IllegalArgumentException("Number of columns does not match data width.");
        if (data.length != index.length)
            throw new IllegalArgumentException("Number of rows does not match index length.");

        this.columns = Arrays.copyOf(columns, columns.length);
        this.index = Arrays.copyOf(index, index.length);
        this.columnsData = new LinkedHashMap<>();

        for (int c = 0; c < columns.length; c++) {
            double[] colData = new double[data.length];
            for (int r = 0; r < data.length; r++) colData[r] = data[r][c];
            this.columnsData.put(columns[c], new Series(columns[c], colData, this.index));
        }
    }

    public DataFrame(Map<String, Series> seriesMap) {
        this.columnsData = new LinkedHashMap<>();
        int length = -1;
        String[] firstIndex = null;
        List<String> colNames = new ArrayList<>();

        for (Map.Entry<String, Series> entry : seriesMap.entrySet()) {
            Series s = entry.getValue();
            if (length == -1) { length = s.size(); firstIndex = s.getIndex(); }
            else if (s.size() != length) throw new IllegalArgumentException("All Series must have the same length.");
            this.columnsData.put(entry.getKey(), s);
            colNames.add(entry.getKey());
        }
        this.columns = colNames.toArray(new String[0]);
        this.index = firstIndex != null ? firstIndex : new String[0];
    }

    private static String[] defaultCols(int n) { String[] c = new String[n]; for (int i = 0; i < n; i++) c[i] = String.valueOf(i); return c; }
    private static String[] defaultIdx(int n) { String[] c = new String[n]; for (int i = 0; i < n; i++) c[i] = String.valueOf(i); return c; }

    // ===== BASIC ACCESSORS =====
    public int[] shape() { return new int[]{index.length, columns.length}; }
    public String[] columns() { return columns; }
    public String[] index() { return index; }
    public Series getColumn(String name) {
        if (!columnsData.containsKey(name)) throw new IllegalArgumentException("Column not found: " + name);
        return columnsData.get(name);
    }
    public Series get(String name) { return getColumn(name); }
    public DataFrameILoc iloc() { return new DataFrameILoc(this); }
    public DataFrameLoc loc() { return new DataFrameLoc(this); }
    public DataFrame describe() { return DataFrameStats.describe(this); }

    // ===== HEAD / TAIL =====
    public DataFrame head(int n) {
        int rows = Math.min(n, index.length);
        int[] indices = new int[rows];
        for (int i = 0; i < rows; i++) indices[i] = i;
        
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) newMap.put(col, columnsData.get(col).subset(indices));
        return new DataFrame(newMap);
    }

    public DataFrame tail(int n) {
        int rows = Math.min(n, index.length);
        int start = index.length - rows;
        int[] indices = new int[rows];
        for (int i = 0; i < rows; i++) indices[i] = start + i;
        
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) newMap.put(col, columnsData.get(col).subset(indices));
        return new DataFrame(newMap);
    }

    // ===== INFO =====
    public void info() {
        System.out.println("<DataFrame>");
        System.out.println("RangeIndex: " + index.length + " entries");
        System.out.println("Data columns (total " + columns.length + " columns):");
        System.out.printf("%-5s %-20s %-10s %-10s\n", "#", "Column", "Non-Null", "Dtype");
        for (int c = 0; c < columns.length; c++) {
            Series s = columnsData.get(columns[c]);
            int nonNull = 0;
            for (int i = 0; i < s.size(); i++) if (!Double.isNaN(s.get(i))) nonNull++;
            String dtype = (s.getStringData() != null) ? "object" : "float64";
            System.out.printf("%-5d %-20s %-10d %-10s\n", c, columns[c], nonNull, dtype);
        }
    }

    // ===== DROP =====
    public DataFrame drop(String... colNames) {
        Set<String> toDrop = new HashSet<>(Arrays.asList(colNames));
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) {
            if (!toDrop.contains(col)) newMap.put(col, columnsData.get(col));
        }
        return new DataFrame(newMap);
    }

    // ===== RENAME =====
    public DataFrame rename(Map<String, String> mapping) {
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) {
            String newName = mapping.getOrDefault(col, col);
            Series s = columnsData.get(col);
            newMap.put(newName, new Series(newName, s.getData(), s.getStringData(), s.getIndex()));
        }
        return new DataFrame(newMap);
    }

    // ===== SORT =====
    public DataFrame sort_values(String column) { return sort_values(column, true); }

    public DataFrame sort_values(String column, boolean ascending) {
        Series s = getColumn(column);
        Integer[] sortIdx = new Integer[index.length];
        for (int i = 0; i < sortIdx.length; i++) sortIdx[i] = i;

        Arrays.sort(sortIdx, (a, b) -> {
            int cmp = Double.compare(s.get(a), s.get(b));
            return ascending ? cmp : -cmp;
        });

        int[] indices = new int[index.length];
        for (int r = 0; r < index.length; r++) indices[r] = sortIdx[r];
        
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) newMap.put(col, columnsData.get(col).subset(indices));
        return new DataFrame(newMap);
    }

    // ===== FILLNA / DROPNA =====
    public DataFrame fillna(double value) {
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) newMap.put(col, columnsData.get(col).fillna(value));
        return new DataFrame(newMap);
    }

    public DataFrame dropna() {
        // Drop rows where ANY column has NaN
        List<Integer> validRows = new ArrayList<>();
        for (int r = 0; r < index.length; r++) {
            boolean valid = true;
            for (String col : columns) {
                Series s = columnsData.get(col);
                if (s.getStringData() == null) {
                    if (Double.isNaN(s.get(r))) { valid = false; break; }
                } else {
                    String val = s.getString(r);
                    if (val == null || val.equalsIgnoreCase("NaN") || val.equalsIgnoreCase("null") || val.trim().isEmpty()) {
                        valid = false;
                        break;
                    }
                }
            }
            if (valid) validRows.add(r);
        }

        int[] indices = new int[validRows.size()];
        for (int i = 0; i < validRows.size(); i++) indices[i] = validRows.get(i);
        
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) newMap.put(col, columnsData.get(col).subset(indices));
        return new DataFrame(newMap);
    }

    // ===== APPLY =====
    public DataFrame apply(Function<Double, Double> func) {
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) newMap.put(col, columnsData.get(col).apply(func));
        return new DataFrame(newMap);
    }

    // ===== GROUPBY =====
    public GroupBy groupby(String column) { return new GroupBy(this, column); }

    // ===== MERGE =====
    public DataFrame merge(DataFrame other, String onColumn) {
        Series leftKey = getColumn(onColumn);
        Series rightKey = other.getColumn(onColumn);

        // Build right index map
        Map<String, List<Integer>> rightMap = new LinkedHashMap<>();
        for (int i = 0; i < rightKey.size(); i++) {
            String key = rightKey.getString(i);
            rightMap.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }

        // Collect all columns (avoid duplicate key column)
        List<String> allCols = new ArrayList<>(Arrays.asList(columns));
        for (String col : other.columns()) {
            if (!col.equals(onColumn)) allCols.add(col);
        }

        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < leftKey.size(); i++) {
            String key = leftKey.getString(i);
            List<Integer> matches = rightMap.get(key);
            if (matches == null) continue;

            for (int j : matches) {
                double[] row = new double[allCols.size()];
                int ci = 0;
                for (String col : columns) row[ci++] = columnsData.get(col).get(i);
                for (String col : other.columns()) {
                    if (!col.equals(onColumn)) row[ci++] = other.getColumn(col).get(j);
                }
                rows.add(row);
            }
        }

        double[][] data = rows.toArray(new double[0][]);
        return new DataFrame(data, allCols.toArray(new String[0]), defaultIdx(data.length));
    }

    // ===== TO ARRAY =====
    public double[][] toArray() {
        double[][] result = new double[index.length][columns.length];
        for (int c = 0; c < columns.length; c++) {
            Series s = columnsData.get(columns[c]);
            for (int r = 0; r < index.length; r++) result[r][c] = s.get(r);
        }
        return result;
    }

    // ===== TO_CSV =====
    public void to_csv(String filepath) {
        try (PrintWriter pw = new PrintWriter(new File(filepath))) {
            pw.print(String.join(",", columns) + "\n");
            for (int r = 0; r < index.length; r++) {
                List<String> rowStr = new ArrayList<>();
                for (String col : columns) {
                    rowStr.add(columnsData.get(col).getString(r));
                }
                pw.print(String.join(",", rowStr) + "\n");
            }
        } catch (IOException e) {
            throw new RuntimeException("Error writing to CSV: " + e.getMessage(), e);
        }
    }

    // ===== CORRELATION =====
    public DataFrame corr() {
        double[][] cMatrix = new double[columns.length][columns.length];
        for (int i = 0; i < columns.length; i++) {
            Series s1 = columnsData.get(columns[i]);
            for (int j = 0; j < columns.length; j++) {
                if (i == j) { cMatrix[i][j] = 1.0; continue; }
                Series s2 = columnsData.get(columns[j]);
                cMatrix[i][j] = pearsonCorr(s1, s2);
            }
        }
        return new DataFrame(cMatrix, columns, columns);
    }

    private double pearsonCorr(Series s1, Series s2) {
        double m1 = s1.mean(), m2 = s2.mean();
        if (Double.isNaN(m1) || Double.isNaN(m2)) return Double.NaN;
        double sumSq1 = 0, sumSq2 = 0, sumCop = 0;
        for (int i = 0; i < s1.size(); i++) {
            double v1 = s1.get(i), v2 = s2.get(i);
            if (Double.isNaN(v1) || Double.isNaN(v2)) continue;
            sumSq1 += (v1 - m1) * (v1 - m1);
            sumSq2 += (v2 - m2) * (v2 - m2);
            sumCop += (v1 - m1) * (v2 - m2);
        }
        return (sumSq1 == 0 || sumSq2 == 0) ? Double.NaN : sumCop / Math.sqrt(sumSq1 * sumSq2);
    }

    // ===== DROP_DUPLICATES =====
    public DataFrame drop_duplicates() {
        Set<String> seen = new HashSet<>();
        List<Integer> keepIndices = new ArrayList<>();
        for (int r = 0; r < index.length; r++) {
            StringBuilder sb = new StringBuilder();
            for (String col : columns) sb.append(columnsData.get(col).getString(r)).append("||");
            String rowStr = sb.toString();
            if (seen.add(rowStr)) {
                keepIndices.add(r);
            }
        }
        int[] indices = new int[keepIndices.size()];
        for (int i = 0; i < keepIndices.size(); i++) indices[i] = keepIndices.get(i);
        
        Map<String, Series> newMap = new LinkedHashMap<>();
        for (String col : columns) newMap.put(col, columnsData.get(col).subset(indices));
        return new DataFrame(newMap);
    }

    // ===== PIVOT_TABLE =====
    public DataFrame pivot_table(String indexCol, String columnsCol, String valuesCol, String aggfunc) {
        Series idxSeries = getColumn(indexCol);
        Series colSeries = getColumn(columnsCol);
        Series valSeries = getColumn(valuesCol);

        Set<String> rowKeys = new LinkedHashSet<>();
        Set<String> colKeys = new LinkedHashSet<>();
        for (int i = 0; i < idxSeries.size(); i++) {
            rowKeys.add(idxSeries.getString(i));
            colKeys.add(colSeries.getString(i));
        }

        List<String> rowKeysList = new ArrayList<>(rowKeys);
        List<String> colKeysList = new ArrayList<>(colKeys);

        Map<String, List<Double>>[][] aggMap = new Map[rowKeysList.size()][colKeysList.size()];
        for (int i = 0; i < idxSeries.size(); i++) {
            int r = rowKeysList.indexOf(idxSeries.getString(i));
            int c = colKeysList.indexOf(colSeries.getString(i));
            if (aggMap[r][c] == null) {
                aggMap[r][c] = new HashMap<>();
                aggMap[r][c].put("values", new ArrayList<>());
            }
            aggMap[r][c].get("values").add(valSeries.get(i));
        }

        double[][] out = new double[rowKeysList.size()][colKeysList.size()];
        for (int r = 0; r < rowKeysList.size(); r++) {
            for (int c = 0; c < colKeysList.size(); c++) {
                if (aggMap[r][c] == null || aggMap[r][c].get("values").isEmpty()) {
                    out[r][c] = Double.NaN;
                } else {
                    List<Double> vals = aggMap[r][c].get("values");
                    double sum = 0; for (double v : vals) sum += v;
                    if (aggfunc.equals("mean")) {
                        out[r][c] = sum / vals.size();
                    } else if (aggfunc.equals("sum")) {
                        out[r][c] = sum;
                    } else {
                        throw new IllegalArgumentException("Unsupported aggfunc: " + aggfunc);
                    }
                }
            }
        }
        return new DataFrame(out, colKeysList.toArray(new String[0]), rowKeysList.toArray(new String[0]));
    }

    // ===== SHOW / TOSTRING =====
    public void show() { System.out.println(this.toString()); }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-10s", ""));
        for (String col : columns) sb.append(String.format("%-15s", col));
        sb.append("\n");

        for (int r = 0; r < index.length; r++) {
            sb.append(String.format("%-10s", index[r]));
            for (String col : columns) sb.append(String.format("%-15s", columnsData.get(col).getString(r)));
            sb.append("\n");
        }
        return sb.toString();
    }
}
