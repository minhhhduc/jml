# 📊 NumJa - Thư Viện Array/Ma Trận Cho Java

NumJa là thư viện NumPy-like chạy trên **Java**. Điểm đặc biệt của phiên bản này là **hoạt động hoàn toàn độc lập**, không cần mạng, không cần cài Maven, và **CHỈ CẦN CÀI ĐẶT 1 LẦN** là có thể gọi thư viện ở bất kỳ đâu trên máy tính bằng `javac` và `java` nguyên bản!

---

## ⚡ CÁCH CÀI ĐẶT CHO MÁY KHÁC (Chỉ làm 1 lần duy nhất)

Nếu bạn gửi dự án này sang một máy tính khác, hãy làm theo các bước sau để máy đó có thể dùng thư viện mà không cần gõ lệnh cấu hình phức tạp:

### Bước 1: Sinh ra file thư viện (.jar)
Vào thư mục gốc và chạy tập lệnh PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_core.ps1
```

### Bước 2: Cài đặt vĩnh viễn vào máy
Để không bao giờ phải gõ đường dẫn thư viện (`-cp`) mỗi khi lập trình, hãy chạy tập lệnh:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_core.ps1
```

Script này sẽ:
1. Copy file thư viện đóng gói bảo mật (ProGuard obfuscated) `.jar` vào thư mục `dist/` và cài đặt vào môi trường của máy tính.
2. Cập nhật biến môi trường **`CLASSPATH`** của hệ thống.

### Bước 3: Khởi động lại Terminal
**Cực kỳ quan trọng:** Sau khi cài xong, bạn **BẮT BUỘC** phải tắt cái Terminal hiện tại đi và mở một cái Terminal mới tinh để hệ điều hành nhận diện biến môi trường vừa cài.

---

## 💥 CÁCH SỬ DỤNG (Tuyệt đối không cần option)

Từ bây giờ trở đi, bất kể bạn hay người khác tạo file Java ở thư mục nào trên máy tính, chỉ cần viết code và chạy bằng 2 lệnh cơ bản nhất.

**Ví dụ, tạo file `NumJaDemo.java`:**
```java
import numja.core.NDArray;
import static numja.NumJa.*;

public class NumJaDemo {
    public static void main(String[] args) {
        NDArray x = array(new double[]{10.0, 20.0, 30.0});
        System.out.println("Trung binh: " + mean(x));
    }
}
```

**Biên dịch và Chạy thẳng trên Terminal:**
```powershell
javac NumJaDemo.java
java NumJaDemo
```
*(Hoàn toàn không cần dùng tham số `-cp` hay khai báo thư viện phức tạp!)*

---

## 🔧 NẾU DÙNG VS CODE (Chỉ việc bấm nút Run)

Mọi thứ đã được cấu hình sẵn trong thư mục `.vscode/`:
1. Mở thư mục này bằng VS Code.
2. Mở file `.java` bất kỳ nằm trong thư mục gốc hoặc `examples/`
3. Bấm **Nút Run (Play)** ở góc trên bên phải màn hình.
✅ VS Code sẽ tự lo liệu mọi thứ mà không văng lỗi `ClassNotFoundException`.

---

## 📚 TÍNH NĂNG CHÍNH CỦA THƯ VIỆN

### 🧠 NumJa Core (Lớp `numja.NumJa` & `numja.core.NDArray`)
- **Toán Học & Tạo Mảng:** `zeros`, `ones`, `linspace`, `sin`, `cos`, `exp`, `array`...
- **Thống Kê:** `mean`, `std`, `var`, `min`, `max`, `argmin`...
- **Đại Số Tuyến Tính:** `det`, `trace`, `inv`, `matmul`, `solve`, `svd`, `qr`, `eig`...

### 🐼 Pandas (Lớp `pandas.DataFrame`)
- **Đọc/Ghi Dữ Liệu:** `read_csv`, `read_json`
- **Thao Tác Dữ Liệu:** `loc[]`, `iloc[]`, `describe()`, `head()`, `show()`, `corr()`...

### 📈 Trực Quan Hóa (Matplotlib & Seaborn)
- **Matplotlib Style (`matplotlib.Matplotlib`):** 
  - `plot(x, y)`: Vẽ đồ thị đường thẳng (Hỗ trợ trực tiếp `NDArray` và `double[]`).
  - `scatter(x, y)`: Vẽ đồ thị phân tán (Hỗ trợ trực tiếp `NDArray` và `double[]`).
  - `imshow(data)`: **[MỚI]** Vẽ lưới điểm ảnh pixel mịn thực tế (Hỗ trợ `NDArray` 2D và ma trận `double[][]`), tự động giãn tỷ lệ ảnh.
  - `title()`, `xlabel()`, `ylabel()`, `clf()`, `savefig()`, `show()`.
- **Seaborn Style (`seaborn.Seaborn`):** 
  - `heatmap(matrix, labels)`: Bản đồ nhiệt confusion/correlation matrix (Hỗ trợ `NDArray` 2D, `double[][]`, `int[][]`).
  - `set_theme()`, `load_dataset()`.

### 🤖 Học Máy & Tập Dữ Liệu (SKLearn)
- **SKLearn Datasets (`sklearn.datasets.Datasets`):** **[MỚI]**
  - `loadIris()`: Nạp tập dữ liệu hoa diên vĩ chuẩn về đối tượng `Bunch` (chứa `.getData()`, `.getTarget()`, `.getFeatureNames()`, `.getTargetNames()`).
  - `makeBlobs()`: Bộ sinh các cụm Gaussian ngẫu nhiên phục vụ bài toán phân cụm.
  - `makeRegression()`: Bộ sinh dữ liệu hồi quy ngẫu nhiên có độ nhiễu Gauss.
- **Model Selection & Preprocessing:** `train_test_split`, `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `SimpleImputer`.
- **Phân Lớp & Hồi Quy:** 9 mô hình phân lớp (`DecisionTree`, `RandomForest`, `KNeighbors`, `GaussianNB`, `LogisticRegression`, `SVC`, `GradientBoosting`, `AdaBoost`, `MLPClassifier`) và 8 mô hình hồi quy.
- **Phân Cụm & Phân Rã:** `KMeans`, `DBSCAN`, `PCA`.
- **Đánh Giá & Pipeline:** `GridSearchCV`, `Pipeline`, `accuracyScore`, `classificationReport`, `confusionMatrix`.
