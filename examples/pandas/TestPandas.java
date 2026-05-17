package pandas;

import pandas.DataFrame;
import pandas.Pandas;
import pandas.Series;

public class TestPandas {
    public static void main(String[] args) {
        System.out.println("=== TEST MODULE: PANDAS (DATAFRAME & SERIES) ===");
        
        double[][] data = {
            {1.1, 2.2, 3.3},
            {4.4, 5.5, 6.6},
            {7.7, 8.8, 9.9},
            {10.1, 11.1, 12.1}
        };
        
        String[] columns = {"A", "B", "C"};
        String[] index = {"row1", "row2", "row3", "row4"};

        DataFrame df = new DataFrame(data, columns, index);
        System.out.println("Shape cua DataFrame: [" + df.shape()[0] + " rows, " + df.shape()[1] + " cols]");
        
        System.out.println("\nHien thi 2 dong dau tien (head):");
        System.out.println(df.head(2));

        System.out.println("\nTrich xuat Series C (column 'C'):");
        Series seriesC = df.get("C");
        System.out.println(seriesC);
        
        System.out.println("\n=== THU NGHIEM DOC FILE CSV NANG CAO ===");
        try {
            // Tao 1 file csv tam o local de thu nghiem voi format bi loch
            java.io.PrintWriter pw = new java.io.PrintWriter("sample_advanced.csv");
            pw.println("Day la dong bi skip 1");
            pw.println("Day la dong bi skip 2");
            pw.println("Id;Age;Salary");
            pw.println("USR1; 25; 1000.5");
            pw.println("USR2; 30; 2000.0");
            pw.println("USR3; 22; 1500.2");
            pw.close();

            // Setup giong y het pandas python kwargs:
            // df = pd.read_csv('sample_advanced.csv', sep=';', skiprows=2, index_col=0, names=['Tuoi', 'Luong'])
            DataFrame csvDf = Pandas.read_csv("sample_advanced.csv", new Pandas.ReadCsvOptions()
                    .sep(";")
                    .skiprows(2)
                    .index_col(0)
                    .names("ID_Khach", "Tuoi", "Luong"));
            
            System.out.println(csvDf.toString());
            
            System.out.println("Thong ke mo ta (describe):");
            System.out.println(csvDf.describe().toString());

            System.out.println("\nLoc dong thu 2 bang iloc(1):");
            System.out.println(csvDf.iloc().get(1).toString());
            
            System.out.println("\nLoc dong co ma 'USR3' bang loc('USR3'):");
            System.out.println(csvDf.loc().get("USR3").toString());
            
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("=== PANDAS TEST SUCCESS ===");
    }
}


