package plot;

import matplotlib.Matplotlib;
import seaborn.Seaborn;
import sklearn.metrics.Metrics;

public class TestConfusionMatrix {
    public static void main(String[] args) {
        System.out.println("=========================================");
        System.out.println("  CONFUSION MATRIX PLOTTER EXAMPLE       ");
        System.out.println("=========================================");

        // Simulating some real and predicted values for a 3-class classification task
        // (e.g. Iris)
        // 0: Setosa, 1: Versicolor, 2: Virginica
        int[] y_true = {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        };

        // Add some classification errors
        int[] y_pred = {
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, // One Setosa predicted as Versicolor
                1, 1, 1, 1, 1, 2, 1, 1, 1, 1, // One Versicolor predicted as Virginica
                2, 2, 2, 1, 2, 2, 2, 2, 2, 2 // One Virginica predicted as Versicolor
        };

        // Calculate Confusion Matrix using sklearn metrics
        int[][] cm = Metrics.confusionMatrix(y_true, y_pred);

        System.out.println("Generated Confusion Matrix:");
        Metrics.printConfusionMatrix(cm);

        // Class labels
        String[] classes = { "Setosa", "Versicolor", "Virginica" };

        boolean headless = args.length > 0 && args[0].equals("--headless");

        try {
            // Clear previous figures
            Matplotlib.clf();

            // Set a beautiful figure title
            Matplotlib.title("Iris Species Confusion Matrix");

            System.out.println("\nGenerating Confusion Matrix Heatmap...");

            // Draw heatmap using Seaborn + Matplotlib
            Seaborn.heatmap(cm, classes);

            if (!headless) {
                System.out.println("Close the Swing window to save the plot image and complete the test.");
                // Display interactive window (blocking)
                Matplotlib.show();
            }

            // Save the plot as a PNG image in the examples output folder
            String outImagePath = "confusion_matrix.png";
            System.out.println("Saving plot to " + outImagePath + "...");
            Matplotlib.savefig(outImagePath);

            System.out.println("=========================================");
            System.out.println("  PLOT GENERATED AND SAVED SUCCESSFULLY! ");
            System.out.println("=========================================");
        } catch (Exception e) {
            System.err.println("Error generating plot: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
