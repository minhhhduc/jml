package sklearn.preprocessing;

import java.util.*;

/**
 * LabelEncoder - Encode target labels with value between 0 and n_classes-1
 * Converts String labels to integer indices and back
 */
public class LabelEncoder {
    private Map<String, Integer> labelToIndex;
    private String[] indexToLabel;
    private boolean fitted = false;

    public LabelEncoder() {}

    public LabelEncoder fit(String[] labels) {
        Set<String> uniqueSet = new LinkedHashSet<>();
        for (String l : labels) uniqueSet.add(l);
        indexToLabel = uniqueSet.toArray(new String[0]);
        Arrays.sort(indexToLabel);
        labelToIndex = new HashMap<>();
        for (int i = 0; i < indexToLabel.length; i++) labelToIndex.put(indexToLabel[i], i);
        fitted = true;
        return this;
    }

    public int[] transform(String[] labels) {
        check();
        int[] result = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            Integer idx = labelToIndex.get(labels[i]);
            if (idx == null) throw new IllegalArgumentException("Unknown label: " + labels[i]);
            result[i] = idx;
        }
        return result;
    }

    public int[] fitTransform(String[] labels) { return fit(labels).transform(labels); }

    public String[] inverseTransform(int[] encoded) {
        check();
        String[] result = new String[encoded.length];
        for (int i = 0; i < encoded.length; i++) {
            if (encoded[i] < 0 || encoded[i] >= indexToLabel.length)
                throw new IllegalArgumentException("Invalid index: " + encoded[i]);
            result[i] = indexToLabel[encoded[i]];
        }
        return result;
    }

    public String[] getClasses() { return indexToLabel != null ? indexToLabel.clone() : null; }
    public int getNumClasses() { return indexToLabel != null ? indexToLabel.length : 0; }

    private void check() { if (!fitted) throw new IllegalStateException("Not fitted."); }
}
