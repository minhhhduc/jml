package pandas;

import java.util.Map;
import java.util.LinkedHashMap;

public class DataFrameILoc {
    private final DataFrame df;

    public DataFrameILoc(DataFrame df) {
        this.df = df;
    }

    /**
     * Lay 1 dong duy nhat dua tren index (so nguyen) -> tra ve 1 Series (ten cot -> gia tri)
     */
    public Series get(int rowIndex) {
        if (rowIndex < 0 || rowIndex >= df.shape()[0]) {
            throw new IndexOutOfBoundsException("Row index out of bounds: " + rowIndex);
        }
        
        String[] originalCols = df.columns();
        double[] rowData = new double[originalCols.length];
        
        for (int c = 0; c < originalCols.length; c++) {
            rowData[c] = df.getColumn(originalCols[c]).get(rowIndex);
        }
        
        // Series bieu dien 1 row se co index la ten cac cot
        return new Series(df.index()[rowIndex], rowData, originalCols);
    }

    /**
     * Lay 1 bang con (sub-dataframe) tu danh sach index dong (rows)
     */
    public DataFrame get(int[] rowIndices) {
        String[] originalCols = df.columns();
        Map<String, Series> newCols = new LinkedHashMap<>();
        
        String[] newIndex = new String[rowIndices.length];
        for (int i = 0; i < rowIndices.length; i++) {
            newIndex[i] = df.index()[rowIndices[i]];
        }

        for (String colName : originalCols) {
            Series oldSeries = df.getColumn(colName);
            double[] newData = new double[rowIndices.length];
            for (int i = 0; i < rowIndices.length; i++) {
                newData[i] = oldSeries.get(rowIndices[i]);
            }
            newCols.put(colName, new Series(colName, newData, newIndex));
        }

        return new DataFrame(newCols);
    }
}


