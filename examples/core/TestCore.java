package core;

import static numja.NumJa.*;
import numja.core.NDArray;

public class TestCore {
    public static void main(String[] args) {
        System.out.println("=== TEST MODULE: NUMJA CORE & LINALG ===");
        
        // Tạo ma trận
        NDArray A = array(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });
        System.out.println("Ma tran A:\n" + A);
        
        // Đại số tuyến tính
        System.out.println("Dinh thuc (Det): " + det(A));
        System.out.println("Vet (Trace): " + trace(A));
        System.out.println("Ma tran nghich dao (Inv):\n" + inv(A));
        
        // Thống kê
        NDArray v = array(10, 20, 30, 40, 50);
        System.out.println("Vector v: " + v);
        System.out.println("Trung binh: " + mean(v));
        System.out.println("Do lech chuan: " + std(v));
        
        System.out.println("=== CORE TEST SUCCESS ===");
    }
}


