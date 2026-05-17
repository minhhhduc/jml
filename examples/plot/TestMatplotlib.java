package plot;

import static numja.NumJa.*;
import numja.core.NDArray;
import matplotlib.Matplotlib;

public class TestMatplotlib {
    public static void main(String[] args) {
        System.out.println("=== TEST MODULE: MATPLOTLIB ===");

        try {
            // Tao du lieu mau
            NDArray x = linspace(0, 10, 100);
            NDArray y = sin(x);
            NDArray y2 = cos(x);

            double[] x_arr = x.toDoubleArray();
            double[] y_arr = y.toDoubleArray();
            double[] y2_arr = y2.toDoubleArray();

            System.out.println("Dang tao do thi phuc tap...");
            Matplotlib.clf();
            Matplotlib.title("Trigonometric Functions");
            Matplotlib.xlabel("X-axis");
            Matplotlib.ylabel("Y-axis");

            Matplotlib.plot(x_arr, y_arr, "Sine Wave");
            Matplotlib.scatter(x_arr, y2_arr, "Cosine Wave");
            Matplotlib.legend();

            System.out.println("Hien thi giao dien (Dong cua so de tiep tuc)...");
            Matplotlib.show();

            System.out.println("=== MATPLOTLIB TEST SUCCESS ===");

        } catch (Exception e) {
            System.out.println("Loi khi ve bieu do: " + e.getMessage());
        }
    }
}
