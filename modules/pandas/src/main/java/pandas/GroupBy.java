package pandas;

import java.util.*;

/**
 * GroupBy - Group DataFrame by a column and apply aggregation functions
 * Similar to pandas.DataFrame.groupby()
 */
public class GroupBy {
    private final DataFrame df;
    private final String groupColumn;
    private final Map<String, List<Integer>> groups;

    public GroupBy(DataFrame df, String groupColumn) {
        this.df = df;
        this.groupColumn = groupColumn;
        this.groups = new LinkedHashMap<>();

        Series key = df.getColumn(groupColumn);
        for (int i = 0; i < key.size(); i++) {
            String k = key.getString(i);
            groups.computeIfAbsent(k, x -> new ArrayList<>()).add(i);
        }
    }

    /** Compute mean for each group */
    public DataFrame mean() { return aggregate("mean"); }

    /** Compute sum for each group */
    public DataFrame sum() { return aggregate("sum"); }

    /** Compute count for each group */
    public DataFrame count() { return aggregate("count"); }

    /** Compute min for each group */
    public DataFrame min() { return aggregate("min"); }

    /** Compute max for each group */
    public DataFrame max() { return aggregate("max"); }

    /** Compute std for each group */
    public DataFrame std() { return aggregate("std"); }

    /** General aggregation */
    public DataFrame aggregate(String func) {
        String[] allCols = df.columns();
        // Get numeric columns (exclude group column)
        List<String> numCols = new ArrayList<>();
        for (String col : allCols) {
            if (!col.equals(groupColumn)) numCols.add(col);
        }

        int nGroups = groups.size();
        int nCols = numCols.size();
        double[][] result = new double[nGroups][nCols];
        String[] groupKeys = groups.keySet().toArray(new String[0]);

        for (int g = 0; g < nGroups; g++) {
            List<Integer> indices = groups.get(groupKeys[g]);
            for (int c = 0; c < nCols; c++) {
                Series s = df.getColumn(numCols.get(c));
                double[] values = new double[indices.size()];
                for (int i = 0; i < indices.size(); i++) values[i] = s.get(indices.get(i));
                result[g][c] = computeAgg(values, func);
            }
        }

        return new DataFrame(result, numCols.toArray(new String[0]), groupKeys);
    }

    private double computeAgg(double[] values, String func) {
        switch (func) {
            case "sum": { double s = 0; for (double v : values) s += v; return s; }
            case "mean": { double s = 0; for (double v : values) s += v; return s / values.length; }
            case "count": return values.length;
            case "min": { double m = Double.MAX_VALUE; for (double v : values) if (v < m) m = v; return m; }
            case "max": { double m = -Double.MAX_VALUE; for (double v : values) if (v > m) m = v; return m; }
            case "std": {
                double mean = 0; for (double v : values) mean += v; mean /= values.length;
                double ss = 0; for (double v : values) ss += (v - mean) * (v - mean);
                return values.length > 1 ? Math.sqrt(ss / (values.length - 1)) : 0;
            }
            default: throw new IllegalArgumentException("Unknown aggregation: " + func);
        }
    }

    /** Get group keys */
    public Set<String> getGroups() { return groups.keySet(); }

    /** Get a specific group as DataFrame */
    public DataFrame getGroup(String key) {
        List<Integer> indices = groups.get(key);
        if (indices == null) throw new IllegalArgumentException("Group not found: " + key);

        String[] allCols = df.columns();
        double[][] data = new double[indices.size()][allCols.length];
        String[] newIndex = new String[indices.size()];

        for (int i = 0; i < indices.size(); i++) {
            int row = indices.get(i);
            newIndex[i] = df.index()[row];
            for (int c = 0; c < allCols.length; c++) {
                data[i][c] = df.getColumn(allCols[c]).get(row);
            }
        }
        return new DataFrame(data, allCols, newIndex);
    }
}
