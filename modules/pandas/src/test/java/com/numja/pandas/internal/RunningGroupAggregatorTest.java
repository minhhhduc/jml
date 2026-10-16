package com.numja.pandas.internal;

import org.junit.Test;
import pandas.DataFrame;
import pandas.GroupBy;
import pandas.Series;
import pandas.internal.RunningGroupAggregator;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

/**
 * Cross-path equivalence tests for {@link RunningGroupAggregator}.
 * Each `_matchesInMemory` test pins the streaming aggregator's output to
 * the in-memory {@link GroupBy} result within {@code 1e-9} tolerance
 * (std uses {@code 2e-9} to allow for two-pass FP rounding).
 *
 * <p>Strategy: build the SAME 6-row dataset, feed it through a 1-chunk
 * iterator (or 2-chunk for {@code twoChunks_mergeCorrectly}), compare the
 * aggregator result against {@code new GroupBy(df, "g").sum()} etc.
 */
public class RunningGroupAggregatorTest {

    /** Static helper: wrap varargs DataFrames as a one-shot Iterator<DataFrame>. */
    private static Iterator<DataFrame> chunksOf(DataFrame... chunks) {
        return new Iterator<DataFrame>() {
            int idx = 0;
            @Override public boolean hasNext() { return idx < chunks.length; }
            @Override public DataFrame next() {
                if (idx >= chunks.length) throw new NoSuchElementException();
                return chunks[idx++];
            }
        };
    }

    /** Build the standard 6-row test DataFrame: g=[A,A,B,B,A,B], v=[1,2,3,4,5,6]. */
    private static DataFrame fixture() {
        String[] index = {"0","1","2","3","4","5"};
        Series g = new Series("g", new double[]{0,0,0,0,0,0},
            new String[]{"A","A","B","B","A","B"}, index);
        Series v = new Series("v", new double[]{1,2,3,4,5,6}, index);
        java.util.Map<String, Series> m = new java.util.LinkedHashMap<>();
        m.put("g", g);
        m.put("v", v);
        return new DataFrame(m);
    }

    /** Look up the numeric value for a given group key in a single-column result DataFrame. */
    private static double valueFor(DataFrame result, String groupKey) {
        String[] idx = result.index();
        for (int i = 0; i < idx.length; i++) {
            if (idx[i].equals(groupKey)) {
                String[] cols = result.columns();
                return result.getColumn(cols[0]).get(i);
            }
        }
        fail("group not found: " + groupKey);
        return Double.NaN;
    }

    @Test
    public void sum_matchesInMemory() {
        DataFrame df = fixture();
        GroupBy mem = new GroupBy(df, "g");

        DataFrame memSum = mem.sum();
        double memA = valueFor(memSum, "A");
        double memB = valueFor(memSum, "B");

        DataFrame streamSum = RunningGroupAggregator.sum(
            chunksOf(df), "g", "v");
        double streamA = valueFor(streamSum, "A");
        double streamB = valueFor(streamSum, "B");

        assertEquals(memA, streamA, 1e-9);
        assertEquals(memB, streamB, 1e-9);
    }

    @Test
    public void mean_matchesInMemory() {
        DataFrame df = fixture();
        GroupBy mem = new GroupBy(df, "g");

        DataFrame memMean = mem.mean();
        DataFrame streamMean = RunningGroupAggregator.mean(
            chunksOf(df), "g", "v");

        assertEquals(valueFor(memMean, "A"), valueFor(streamMean, "A"), 1e-9);
        assertEquals(valueFor(memMean, "B"), valueFor(streamMean, "B"), 1e-9);
    }

    @Test
    public void count_matchesInMemory() {
        DataFrame df = fixture();
        GroupBy mem = new GroupBy(df, "g");

        DataFrame memCount = mem.count();
        DataFrame streamCount = RunningGroupAggregator.count(
            chunksOf(df), "g", "v");

        assertEquals(valueFor(memCount, "A"), valueFor(streamCount, "A"), 1e-9);
        assertEquals(valueFor(memCount, "B"), valueFor(streamCount, "B"), 1e-9);
    }

    @Test
    public void min_matchesInMemory() {
        DataFrame df = fixture();
        GroupBy mem = new GroupBy(df, "g");

        DataFrame memMin = mem.min();
        DataFrame streamMin = RunningGroupAggregator.min(
            chunksOf(df), "g", "v");

        assertEquals(valueFor(memMin, "A"), valueFor(streamMin, "A"), 1e-9);
        assertEquals(valueFor(memMin, "B"), valueFor(streamMin, "B"), 1e-9);
    }

    @Test
    public void max_matchesInMemory() {
        DataFrame df = fixture();
        GroupBy mem = new GroupBy(df, "g");

        DataFrame memMax = mem.max();
        DataFrame streamMax = RunningGroupAggregator.max(
            chunksOf(df), "g", "v");

        assertEquals(valueFor(memMax, "A"), valueFor(streamMax, "A"), 1e-9);
        assertEquals(valueFor(memMax, "B"), valueFor(streamMax, "B"), 1e-9);
    }

    @Test
    public void std_matchesInMemory() {
        DataFrame df = fixture();
        GroupBy mem = new GroupBy(df, "g");

        DataFrame memStd = mem.std();
        DataFrame streamStd = RunningGroupAggregator.std(
            chunksOf(df), "g", "v");

        // std uses 2e-9 tolerance to allow for two-pass FP rounding
        assertEquals(valueFor(memStd, "A"), valueFor(streamStd, "A"), 2e-9);
        assertEquals(valueFor(memStd, "B"), valueFor(streamStd, "B"), 2e-9);
    }

    @Test
    public void twoChunks_mergeCorrectly() {
        DataFrame df = fixture();
        // Split: rows 0-2 (A,A,B), rows 3-5 (B,A,B)
        DataFrame chunk1 = slice(df, 0, 3);
        DataFrame chunk2 = slice(df, 3, 6);

        DataFrame singleSum = RunningGroupAggregator.sum(
            chunksOf(df), "g", "v");
        DataFrame twoSum = RunningGroupAggregator.sum(
            chunksOf(chunk1, chunk2), "g", "v");

        assertEquals(valueFor(singleSum, "A"), valueFor(twoSum, "A"), 0.0);
        assertEquals(valueFor(singleSum, "B"), valueFor(twoSum, "B"), 0.0);

        DataFrame singleMean = RunningGroupAggregator.mean(
            chunksOf(df), "g", "v");
        DataFrame twoMean = RunningGroupAggregator.mean(
            chunksOf(chunk1, chunk2), "g", "v");
        assertEquals(valueFor(singleMean, "A"), valueFor(twoMean, "A"), 0.0);
        assertEquals(valueFor(singleMean, "B"), valueFor(twoMean, "B"), 0.0);

        DataFrame singleMin = RunningGroupAggregator.min(
            chunksOf(df), "g", "v");
        DataFrame twoMin = RunningGroupAggregator.min(
            chunksOf(chunk1, chunk2), "g", "v");
        assertEquals(valueFor(singleMin, "A"), valueFor(twoMin, "A"), 0.0);
        assertEquals(valueFor(singleMin, "B"), valueFor(twoMin, "B"), 0.0);

        DataFrame singleMax = RunningGroupAggregator.max(
            chunksOf(df), "g", "v");
        DataFrame twoMax = RunningGroupAggregator.max(
            chunksOf(chunk1, chunk2), "g", "v");
        assertEquals(valueFor(singleMax, "A"), valueFor(twoMax, "A"), 0.0);
        assertEquals(valueFor(singleMax, "B"), valueFor(twoMax, "B"), 0.0);
    }

    @Test
    public void nanValues_skipped() {
        // Build df with one NaN in v for group A: g=[A,A,A,B], v=[1, NaN, 3, 10]
        String[] index = {"0","1","2","3"};
        Series g = new Series("g", new double[4], new String[]{"A","A","A","B"}, index);
        Series v = new Series("v", new double[]{1.0, Double.NaN, 3.0, 10.0}, index);
        java.util.Map<String, Series> m = new java.util.LinkedHashMap<>();
        m.put("g", g); m.put("v", v);
        DataFrame df = new DataFrame(m);

        // NOTE: streaming aggregator NaN-skips (mirrors Series.sum line 69),
        // while in-memory GroupBy.sum() does NOT skip NaN. Test streaming behavior
        // in isolation, not equivalence.
        DataFrame streamSum = RunningGroupAggregator.sum(
            chunksOf(df), "g", "v");
        DataFrame streamCount = RunningGroupAggregator.count(
            chunksOf(df), "g", "v");
        DataFrame streamMean = RunningGroupAggregator.mean(
            chunksOf(df), "g", "v");

        // Group A: NaN skipped → sum=1+3=4, count=2, mean=4/2=2
        assertEquals(4.0, valueFor(streamSum, "A"), 1e-9);
        assertEquals(2.0, valueFor(streamCount, "A"), 1e-9);
        assertEquals(2.0, valueFor(streamMean, "A"), 1e-9);

        // Group B: no NaN → sum=10, count=1
        assertEquals(10.0, valueFor(streamSum, "B"), 1e-9);
        assertEquals(1.0, valueFor(streamCount, "B"), 1e-9);
    }

    /** Slice a DataFrame to rows [from, to). Preserves columns + index labels. */
    private static DataFrame slice(DataFrame src, int from, int to) {
        String[] cols = src.columns();
        java.util.Map<String, Series> out = new java.util.LinkedHashMap<>();
        int len = to - from;
        String[] newIdx = new String[len];
        for (int i = 0; i < len; i++) newIdx[i] = String.valueOf(i);
        for (String col : cols) {
            Series s = src.getColumn(col);
            double[] data = new double[len];
            String[] strings = null;
            if (s.getStringData() != null) {
                strings = new String[len];
                for (int i = 0; i < len; i++) {
                    data[i] = s.get(from + i);
                    strings[i] = s.getString(from + i);
                }
                out.put(col, new Series(col, data, strings, newIdx));
            } else {
                for (int i = 0; i < len; i++) data[i] = s.get(from + i);
                out.put(col, new Series(col, data, newIdx));
            }
        }
        return new DataFrame(out);
    }
}
