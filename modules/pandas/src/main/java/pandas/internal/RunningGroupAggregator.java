package pandas.internal;

import pandas.DataFrame;
import pandas.Series;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Merge-aware per-group aggregator for streaming ingestion. Produces results
 * that match {@link pandas.GroupBy} (in-memory) within {@code 1e-9} for
 * {@code sum}, {@code mean}, {@code count}, {@code min}, {@code max}, and
 * {@code std}.
 *
 * <p>Usage: feed an {@link Iterator} of {@link DataFrame} chunks (typically
 * from {@link CsvChunkReader}); one row of the returned {@link DataFrame} per
 * unique group key, in first-seen order (LinkedHashMap).
 *
 * <p>Trust boundary: caller-supplied iterator + column names → in-process
 * DataFrame. STRIDE threats:
 * <ul>
 *   <li>T402 cross-path drift — mitigated by the {@code 1e-9} tolerance in
 *       {@code RunningGroupAggregatorTest.sum_matchesInMemory} et al</li>
 *   <li>T406 NaN cells — mitigated by {@code Double.isNaN(v)} skip in
 *       {@link RunningAgg#merge(double)} mirroring {@link pandas.Series#sum()}
 *       line 69</li>
 *</ul>
 *
 * @since ASVS-L1
 */
public final class RunningGroupAggregator {

    private RunningGroupAggregator() {}

    /**
     * Per-group accumulator. WR-05 defensive identity init: min/max start at
     * {@code +/-Double.MAX_VALUE} (NOT Java's {@code 0.0} default) so the
     * first {@code merge(v)} correctly tracks min/max.
     */
    private static final class RunningAgg {
        double sum;
        double sumSq;
        long count;
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        void merge(double v) {
            if (Double.isNaN(v)) return; // mirror Series.sum NaN-skip
            sum += v;
            sumSq += v * v;
            count++;
            if (v < min) min = v;
            if (v > max) max = v;
        }

        void merge(RunningAgg o) {
            sum += o.sum;
            sumSq += o.sumSq;
            count += o.count;
            if (o.min < min) min = o.min;
            if (o.max > max) max = o.max;
        }

        double value(String func) {
            switch (func) {
                case "sum":   return sum;
                case "mean":  return count > 0 ? sum / count : Double.NaN;
                case "count": return count;
                case "min":   return min;
                case "max":   return max;
                case "std": {
                    // sample std (n-1 denominator) matching GroupBy.computeAgg line 81
                    if (count < 2) return 0;
                    double mean = sum / count;
                    return Math.sqrt((sumSq - count * mean * mean) / (count - 1));
                }
                default: throw new IllegalArgumentException("Unknown aggregation: " + func);
            }
        }
    }

    private static DataFrame aggregate(Iterator<DataFrame> chunks,
                                       String groupColumn, String aggColumn,
                                       String func) {
        Map<String, RunningAgg> groups = new LinkedHashMap<>();
        while (chunks.hasNext()) {
            DataFrame chunk = chunks.next();
            Series gs = chunk.getColumn(groupColumn);
            Series vs = chunk.getColumn(aggColumn);
            int n = gs.size();
            for (int i = 0; i < n; i++) {
                String key = gs.getString(i);
                groups.computeIfAbsent(key, k -> new RunningAgg()).merge(vs.get(i));
            }
        }

        int nGroups = groups.size();
        double[][] result = new double[nGroups][1];
        String[] groupKeys = groups.keySet().toArray(new String[0]);
        for (int g = 0; g < nGroups; g++) {
            result[g][0] = groups.get(groupKeys[g]).value(func);
        }
        return new DataFrame(result, new String[]{aggColumn}, groupKeys);
    }

    public static DataFrame sum(Iterator<DataFrame> chunks, String groupColumn, String aggColumn) {
        return aggregate(chunks, groupColumn, aggColumn, "sum");
    }
    public static DataFrame mean(Iterator<DataFrame> chunks, String groupColumn, String aggColumn) {
        return aggregate(chunks, groupColumn, aggColumn, "mean");
    }
    public static DataFrame count(Iterator<DataFrame> chunks, String groupColumn, String aggColumn) {
        return aggregate(chunks, groupColumn, aggColumn, "count");
    }
    public static DataFrame min(Iterator<DataFrame> chunks, String groupColumn, String aggColumn) {
        return aggregate(chunks, groupColumn, aggColumn, "min");
    }
    public static DataFrame max(Iterator<DataFrame> chunks, String groupColumn, String aggColumn) {
        return aggregate(chunks, groupColumn, aggColumn, "max");
    }
    public static DataFrame std(Iterator<DataFrame> chunks, String groupColumn, String aggColumn) {
        return aggregate(chunks, groupColumn, aggColumn, "std");
    }
}
