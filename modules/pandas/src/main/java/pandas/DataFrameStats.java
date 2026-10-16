package pandas;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Arrays;

public class DataFrameStats {

    public static DataFrame describe(DataFrame df) {
        String[] originalCols = df.columns();
        String[] statNames = {"count", "mean", "std", "min", "25%", "50%", "75%", "max"};
        
        Map<String, Series> statSeriesMap = new LinkedHashMap<>();
        
        for (String col : originalCols) {
            Series s = df.getColumn(col);
            double[] data = s.getData();
            int count = data.length;
            
            if (count == 0) {
                statSeriesMap.put(col, new Series(col, new double[8], statNames));
                continue;
            }

            double sum = 0;
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            
            for (double v : data) {
                sum += v;
                if(v < min) min = v;
                if(v > max) max = v;
            }
            double mean = sum / count;
            
            double sumSq = 0;
            for (double v : data) {
                sumSq += (v - mean) * (v - mean);
            }
            double std = count > 1 ? Math.sqrt(sumSq / (count - 1)) : 0.0;
            
            double[] sorted = data.clone();
            Arrays.sort(sorted);
            double p25 = percentile(sorted, 0.25);
            double p50 = percentile(sorted, 0.50);
            double p75 = percentile(sorted, 0.75);
            
            double[] statsData = {count, mean, std, min, p25, p50, p75, max};
            statSeriesMap.put(col, new Series(col, statsData, statNames));
        }
        
        return new DataFrame(statSeriesMap);
    }

    private static double percentile(double[] sorted, double percent) {
        int index = (int) Math.round(percent * (sorted.length - 1));
        return sorted[index];
    }
}


