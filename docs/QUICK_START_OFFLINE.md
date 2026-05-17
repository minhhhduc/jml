# NumJa - Hướng dẫn chạy offline (không cần mạng)

## Bước 1: Chuẩn bị (trên máy có mạng/libraries)

Trên máy có kết nối internet hoặc có các dependencies, chạy:

```powershell
cd d:\unknown\projects\java_ml
mvn -f pom.xml -DskipTests package
.\dist\make_dist.ps1
```

Kết quả: `dist\numja-0.1.0.zip` chứa tất cả JAR cần thiết.

## Bước 2: Phân phối zip cho client

Gửi file `dist\numja-0.1.0.zip` bằng USB, email, share, v.v. (không cần mạng khi cài).

## Bước 3: Client cài đặt (trên máy không cần kết nối mạng)

### Cách 1: Cài vào thư mục người dùng (đơn giản nhất)

```powershell
# 1. Giải nén zip vào thư mục tạm, ví dụ:
cd C:\tmp
Expand-Archive -Path numja-0.1.0.zip -DestinationPath numja

# 2. Chạy installer
cd numja
powershell -ExecutionPolicy Bypass -File .\install_from_zip.ps1
```

Kết quả: Tất cả JAR được copy vào `%USERPROFILE%\libs\numja\0.1.0`.

**Sử dụng trong IDE hoặc dự án:**
- IntelliJ/Eclipse: Thêm các JAR từ `%USERPROFILE%\libs\numja\0.1.0` vào Global Libraries hoặc Project Libraries.
- Gradle flatDir:
```groovy
repositories {
  flatDir {
    dirs "${System.properties['user.home']}/libs/numja/0.1.0"
  }
}
dependencies {
  implementation name: 'numja-0.1.0'
}
```

### Cách 2: Cài vào local Maven repo (nếu máy client có Maven)

```powershell
# 1. Giải nén zip
cd C:\tmp
Expand-Archive -Path numja-0.1.0.zip -DestinationPath numja
cd numja

# 2. Cài vào ~/.m2
powershell -ExecutionPolicy Bypass -File .\install_into_m2.ps1
```

Kết quả: Các JAR được cài vào local Maven repo. Sau đó, các dự án Maven có thể dùng:

```xml
<dependency>
  <groupId>com.numja</groupId>
  <artifactId>numja</artifactId>
  <version>0.1.0</version>
</dependency>
<dependency>
  <groupId>org.ejml</groupId>
  <artifactId>ejml-core</artifactId>
  <version>0.43.1</version>
</dependency>
<!-- + các dependencies khác -->
```

## Bước 4: Chạy ví dụ

Xem file `examples/NumJaClientExample.java` để học cách sử dụng NumJa.

Ví dụ cơ bản:

```java
import com.numja.NumJa;
import com.numja.core.NDArray;

public class MyApp {
    public static void main(String[] args) {
        // Tạo mảng
        NDArray a = NumJa.array(1, 2, 3, 4, 5);
        
        // Phép toán
        System.out.println("Sum: " + NumJa.sum(a));
        System.out.println("Mean: " + NumJa.mean(a));
        
        // Ma trận
        NDArray A = NumJa.array(new double[][]{{1,2},{3,4}});
        double det = NumJa.det(A);
        System.out.println("det(A) = " + det);
    }
}
```

## Bước 5: Biên dịch dự án client

**Nếu dùng IDEs (IntelliJ/Eclipse):**
- Sau khi thêm JAR vào libraries, IDEs sẽ tự nhận diện.
- Compile và chạy như bình thường.

**Nếu dùng Maven + cài vào ~/.m2:**
```powershell
mvn clean package
```

**Nếu dùng Gradle + flatDir:**
```powershell
gradle build
```

## Ghi chú

- Cách 1 (thư mục người dùng) là **đơn giản nhất** và không cần Maven trên máy client.
- Cách 2 (cài vào ~/.m2) cần Maven nhưng cho phép tham chiếu qua `groupId:artifactId:version`.
- Không cần internet hoặc mạng khi cài — chỉ cần file zip.
- Tất cả dependencies (EJML, JFreeChart, v.v.) đều có sẵn trong zip.
- Nếu máy client không có JDK 25, cần cài trước.

## Các hàm NumJa phổ biến

```java
// Tạo mảng
NumJa.array(1, 2, 3)           // từ values
NumJa.zeros(3, 4)              // ma trận 0
NumJa.ones(2, 3)               // ma trận 1
NumJa.eye(3)                   // ma trận đơn vị
NumJa.linspace(0, 10, 100)     // 100 điểm từ 0 đến 10
NumJa.arange(0, 10, 0.5)       // mảng từ 0 đến 10, bước 0.5

// Phép toán
NumJa.add(a, b)                // a + b
NumJa.multiply(a, b)           // a * b (element-wise)
NumJa.dot(A, B)                // A @ B (ma trận nhân)

// Rút gọn
NumJa.sum(a)                   // tổng tất cả
NumJa.mean(a)                  // trung bình
NumJa.std(a)                   // độ lệch chuẩn
NumJa.min(a)                   // giá trị nhỏ nhất
NumJa.max(a)                   // giá trị lớn nhất

// Đại số tuyến tính
NumJa.det(A)                   // định thức
NumJa.inv(A)                   // ma trận nghịch đảo
NumJa.solve(A, b)              // giải hệ Ax = b
NumJa.trace(A)                 // vết (tổng đường chéo)
NumJa.norm(A)                  // chuẩn Frobenius
NumJa.matrixRank(A)            // hạng ma trận

// Hàm toán học
NumJa.sqrt(a)                  // căn bậc 2
NumJa.exp(a)                   // e^a
NumJa.log(a)                   // ln(a)
NumJa.sin(a), cos(a), tan(a)   // lượng giác
NumJa.abs(a)                   // giá trị tuyệt đối
NumJa.power(a, n)              // a^n

// Vẽ biểu đồ
import com.numja.plot.matplotlib.Matplotlib;
Matplotlib.plotLine(x, y, "title", "output.png");

import com.numja.plot.seaborn.Seaborn;
Seaborn.plotScatter(x, y, "title", "scatter.png");
```

Chúc mừng! Bạn đã sẵn sàng sử dụng NumJa mà không cần mạng. 🎉
