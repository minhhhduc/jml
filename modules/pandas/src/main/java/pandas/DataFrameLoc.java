package pandas;

import java.util.LinkedHashMap;
import java.util.Map;

public class DataFrameLoc {
    private final DataFrame df;

    public DataFrameLoc(DataFrame df) {
        this.df = df;
    }

    /**
     * Lay 1 dong duy nhat dua tren ten index (String)
     */
    public Series get(String indexLabel) {
        String[] indices = df.index();
        int targetRow = -1;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i].equals(indexLabel)) {
                targetRow = i;
                break;
            }
        }
        
        if (targetRow == -1) {
            throw new IllegalArgumentException("Index label not found: " + indexLabel);
        }
        
        return df.iloc().get(targetRow);
    }
    
    /**
     * Lay nhieu dong dua tren danh sach ten index (String)
     */
    public DataFrame get(String[] indexLabels) {
        String[] indices = df.index();
        int[] targetRows = new int[indexLabels.length];
        
        for (int i = 0; i < indexLabels.length; i++) {
            targetRows[i] = -1;
            for (int j = 0; j < indices.length; j++) {
                if (indices[j].equals(indexLabels[i])) {
                    targetRows[i] = j;
                    break;
                }
            }
            if (targetRows[i] == -1) {
                throw new IllegalArgumentException("Index label not found: " + indexLabels[i]);
            }
        }
        
        return df.iloc().get(targetRows);
    }
}

