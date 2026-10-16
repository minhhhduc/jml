package seaborn;

import matplotlib.Matplotlib;
import pandas.DataFrame;
import pandas.Series;

/**
 * Seaborn - Statistical data visualization library
 * Built on top of Matplotlib with high-level interface
 */
public final class Seaborn {
    private Seaborn() {}

    /**
     * Set the visual theme for plots
     */
    public static void set_theme() {}

    /**
     * Load a standard dataset from the local datasets/ folder
     * Available datasets: "iris", etc.
     * @param name name of the dataset to load
     * @return DataFrame containing the dataset
     * @throws RuntimeException if dataset file not found
     */
    public static DataFrame load_dataset(String name) {
        try {
            // Tim file trong thu muc datasets/ o local
            String path = "datasets/" + name + ".csv";
            return pandas.Pandas.read_csv(path);
        } catch (Exception e) {
            throw new RuntimeException("Khong the tai dataset '" + name + "'. Hay dam bao file ton tai tai 'datasets/" + name + ".csv'");
        }
    }

    /**
     * Plot a line graph from DataFrame columns
     * @param data DataFrame containing the data
     * @param x column name for X-axis
     * @param y column name for Y-axis
     */
    public static void plot(DataFrame data, String x, String y) {
        Series seriesX = data.get(x);
        Series seriesY = data.get(y);
        Matplotlib.plot(seriesX.getData(), seriesY.getData(), y);
        Matplotlib.xlabel(x);
        Matplotlib.ylabel(y);
    }

    /**
     * Plot a scatter graph from DataFrame columns
     * @param data DataFrame containing the data
     * @param x column name for X-axis
     * @param y column name for Y-axis
     */
    public static void scatter(DataFrame data, String x, String y) {
        Series seriesX = data.get(x);
        Series seriesY = data.get(y);
        Matplotlib.scatter(seriesX.getData(), seriesY.getData(), y);
        Matplotlib.xlabel(x);
        Matplotlib.ylabel(y);
    }

    /**
     * Plot a correlation or confusion matrix heatmap
     * @param matrix 2D double array containing data values
     * @param labels string labels for X/Y axes ticks
     */
    public static void heatmap(double[][] matrix, String[] labels) {
        Matplotlib.heatmap(matrix, labels);
    }

    /**
     * Plot a confusion matrix heatmap
     * @param matrix 2D integer array containing counts (confusion matrix values)
     * @param labels string labels for classes
     */
    public static void heatmap(int[][] matrix, String[] labels) {
        double[][] dMatrix = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                dMatrix[i][j] = matrix[i][j];
            }
        }
        Matplotlib.heatmap(dMatrix, labels);
    }

    /**
     * Plot a correlation or confusion matrix heatmap from an NDArray
     * @param matrix NDArray containing data values
     * @param labels string labels for X/Y axes ticks
     */
    public static void heatmap(numja.core.NDArray matrix, String[] labels) {
        Matplotlib.heatmap(matrix.toArray(), labels);
    }
}


