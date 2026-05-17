import static numja.NumJa.*;
import numja.core.NDArray;
import matplotlib.Matplotlib;

public class NumJaClientExample {
    public static void main(String[] args) throws Exception {
        System.out.println("=== NumJa Client Example ===\n");

        // Create arrays
        NDArray x = linspace(0, 10, 100);
        NDArray y = sin(x);

        System.out.println("=== MATH & STATS ===");
        System.out.println("y mean: " + mean(y));
        System.out.println("y min: " + min(y));
        System.out.println("y max: " + max(y));

        System.out.println("\n=== LINEAR ALGEBRA ===");
        NDArray A = array(new double[][] { { 1, 2 }, { 3, 4 } });
        NDArray b = array(5.0, 6.0);
        System.out.println("Matrix A:\n" + A);
        System.out.println("Vector b: " + b);
        System.out.println("det(A) = " + det(A));
        System.out.println("trace(A) = " + trace(A));
        System.out.println("norm(A) = " + norm(A));

        System.out.println("\nSolving Ax = b:");
        NDArray x_sol = solve(A, b);
        System.out.println("Solution x to Ax=b: " + x_sol);

        // Plotting (if graphics available)
        try {
            double[] xdata = x.toDoubleArray();
            double[] ydata = y.toDoubleArray();
            Matplotlib.clf();
            Matplotlib.plot(xdata, ydata, "sin(x)");
            Matplotlib.savefig("sine_plot.png");
            System.out.println("\nPlot saved as sine_plot.png");
        } catch (Exception e) {
            System.out.println("Plotting skipped: " + e.getMessage());
        }

        System.out.println("\n=== Success ===");
    }
}
