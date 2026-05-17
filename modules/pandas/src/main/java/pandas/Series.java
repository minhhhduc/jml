package pandas;

import java.util.*;
import java.util.function.Function;

/**
 * Series - A 1D data structure with labeled indices
 * Similar to pandas.Series in Python
 */
public class Series {
    private String name;
    private double[] data;
    private String[] stringData;
    private String[] index;

    public Series(double[] data) { this(null, data, defaultIndex(data.length)); }
    public Series(String name, double[] data) { this(name, data, defaultIndex(data.length)); }
    public Series(String name, double[] data, String[] index) { this(name, data, null, index); }

    public Series(String name, double[] data, String[] stringData, String[] index) {
        if (data != null && index != null && data.length != index.length)
            throw new IllegalArgumentException("Data and index lengths must match.");
        this.name = name;
        this.data = (data != null) ? Arrays.copyOf(data, data.length) : null;
        this.stringData = (stringData != null) ? Arrays.copyOf(stringData, stringData.length) : null;
        this.index = Arrays.copyOf(index, index.length);
    }

    private static String[] defaultIndex(int length) {
        String[] idx = new String[length];
        for (int i = 0; i < length; i++) idx[i] = String.valueOf(i);
        return idx;
    }

    // ===== BASIC ACCESSORS =====
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public double[] getData() { return data; }
    public String[] getStringData() { return stringData; }
    public String[] getIndex() { return index; }
    public double get(int i) { return data[i]; }
    public String getString(int i) {
        if (stringData != null && i < stringData.length && stringData[i] != null) return stringData[i];
        return String.valueOf(data[i]);
    }
    public double get(String idxStr) {
        for (int i = 0; i < index.length; i++) if (index[i].equals(idxStr)) return data[i];
        throw new IllegalArgumentException("Index not found: " + idxStr);
    }
    public int size() { return index.length; }
    
    // ===== SUBSET =====
    public Series subset(int[] rowIndices) {
        double[] newData = new double[rowIndices.length];
        String[] newStringData = stringData != null ? new String[rowIndices.length] : null;
        String[] newIndex = new String[rowIndices.length];
        
        for (int i = 0; i < rowIndices.length; i++) {
            int r = rowIndices[i];
            newData[i] = data[r];
            if (stringData != null) newStringData[i] = stringData[r];
            newIndex[i] = index[r];
        }
        return new Series(name, newData, newStringData, newIndex);
    }

    // ===== STATISTICAL METHODS =====
    public double sum() {
        double s = 0; for (double v : data) if (!Double.isNaN(v)) s += v; return s;
    }
    public double mean() {
        int c = 0; double s = 0;
        for (double v : data) if (!Double.isNaN(v)) { s += v; c++; }
        return c > 0 ? s / c : Double.NaN;
    }
    public double std() {
        double m = mean(); int c = 0; double ss = 0;
        for (double v : data) if (!Double.isNaN(v)) { ss += (v - m) * (v - m); c++; }
        return c > 1 ? Math.sqrt(ss / (c - 1)) : 0;
    }
    public double min() {
        double m = Double.MAX_VALUE;
        for (double v : data) if (!Double.isNaN(v) && v < m) m = v;
        return m;
    }
    public double max() {
        double m = -Double.MAX_VALUE;
        for (double v : data) if (!Double.isNaN(v) && v > m) m = v;
        return m;
    }
    public double median() {
        double[] sorted = validValues();
        Arrays.sort(sorted);
        if (sorted.length == 0) return Double.NaN;
        int mid = sorted.length / 2;
        return sorted.length % 2 == 0 ? (sorted[mid-1] + sorted[mid]) / 2.0 : sorted[mid];
    }
    public double var() {
        double m = mean(); int c = 0; double ss = 0;
        for (double v : data) if (!Double.isNaN(v)) { ss += (v - m) * (v - m); c++; }
        return c > 1 ? ss / (c - 1) : 0;
    }

    // ===== UNIQUE / VALUE COUNTS =====
    public double[] unique() {
        Set<Double> set = new LinkedHashSet<>();
        for (double v : data) set.add(v);
        double[] result = new double[set.size()];
        int i = 0; for (double v : set) result[i++] = v;
        return result;
    }

    public String[] uniqueStrings() {
        if (stringData == null) {
            double[] u = unique();
            String[] result = new String[u.length];
            for (int i = 0; i < u.length; i++) result[i] = String.valueOf(u[i]);
            return result;
        }
        Set<String> set = new LinkedHashSet<>();
        for (String s : stringData) if (s != null) set.add(s);
        return set.toArray(new String[0]);
    }

    public int nunique() { return unique().length; }

    /** Value counts - returns a new Series with values as index and counts as data */
    public Series value_counts() {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int i = 0; i < size(); i++) {
            String key = getString(i);
            counts.put(key, counts.getOrDefault(key, 0) + 1);
        }
        // Sort by count descending
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(counts.entrySet());
        entries.sort((a, b) -> b.getValue() - a.getValue());

        String[] idx = new String[entries.size()];
        double[] vals = new double[entries.size()];
        for (int i = 0; i < entries.size(); i++) {
            idx[i] = entries.get(i).getKey();
            vals[i] = entries.get(i).getValue();
        }
        return new Series(name, vals, idx);
    }

    // ===== TRANSFORM METHODS =====
    /** Apply a function to each numeric value */
    public Series apply(Function<Double, Double> func) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) result[i] = func.apply(data[i]);
        return new Series(name, result, index);
    }

    /** Map values using a dictionary */
    public Series map(Map<String, String> mapping) {
        String[] newStringData = new String[size()];
        double[] newData = new double[size()];
        Arrays.fill(newData, Double.NaN);

        for (int i = 0; i < size(); i++) {
            String key = getString(i);
            String mapped = mapping.get(key);
            if (mapped != null) {
                newStringData[i] = mapped;
                try { newData[i] = Double.parseDouble(mapped); } catch (NumberFormatException e) { /* keep NaN */ }
            } else {
                newStringData[i] = key;
                newData[i] = data[i];
            }
        }
        return new Series(name, newData, newStringData, index);
    }

    /** Fill NaN values with a constant */
    public Series fillna(double value) {
        double[] result = data.clone();
        for (int i = 0; i < result.length; i++) if (Double.isNaN(result[i])) result[i] = value;
        return new Series(name, result, stringData, index);
    }

    /** Drop NaN values */
    public Series dropna() {
        List<Double> vals = new ArrayList<>();
        List<String> idx = new ArrayList<>();
        List<String> sData = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            if (!Double.isNaN(data[i])) {
                vals.add(data[i]); idx.add(index[i]);
                if (stringData != null) sData.add(stringData[i]);
            }
        }
        double[] d = new double[vals.size()];
        for (int i = 0; i < vals.size(); i++) d[i] = vals.get(i);
        String[] si = idx.toArray(new String[0]);
        String[] sd = sData.isEmpty() ? null : sData.toArray(new String[0]);
        return new Series(name, d, sd, si);
    }

    /** Convert to int array (for labels) */
    public int[] asIntArray() {
        int[] result = new int[data.length];
        for (int i = 0; i < data.length; i++) result[i] = (int) data[i];
        return result;
    }

    private double[] validValues() {
        List<Double> valid = new ArrayList<>();
        for (double v : data) if (!Double.isNaN(v)) valid.add(v);
        double[] result = new double[valid.size()];
        for (int i = 0; i < valid.size(); i++) result[i] = valid.get(i);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (name != null) sb.append("Series: ").append(name).append("\n");
        for (int i = 0; i < index.length; i++) {
            String val = (stringData != null && stringData[i] != null) ? stringData[i] : String.valueOf(data[i]);
            sb.append(index[i]).append("    ").append(val).append("\n");
        }
        return sb.toString();
    }
}
