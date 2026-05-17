package plot;

import pandas.DataFrame;
import pandas.Pandas;
import matplotlib.Matplotlib;
import seaborn.Seaborn;

public class TestSeaborn {
    public static void main(String[] args) {
        System.out.println("=== TEST MODULE: SEABORN ===");

        try {
            // Tao du thu nghiem Dataframe
            double[][] data = {
                    { 1.1, 2.2 },
                    { 4.4, 5.5 },
                    { 7.7, 8.8 },
                    { 10.1, 11.2 }
            };
            String[] columns = { "X_Axis", "Y_Axis" };
            String[] index = { "0", "1", "2", "3" };

            DataFrame df = new DataFrame(data, columns, index);

            System.out.println("Dang ve bieu do Scatter bang DataFrame (Seaborn)...");
            Matplotlib.clf();
            Seaborn.scatter(df, "X_Axis", "Y_Axis");

            System.out.println("Hien thi giao dien (Dong cua so de tiep tuc)...");
            Matplotlib.show();

            System.out.println("=== SEABORN TEST SUCCESS ===");

        } catch (Exception e) {
            System.out.println("Loi khi ve bieu do: " + e.getMessage());
        }
    }
}
