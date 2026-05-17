# 🏆 Thư Viện Học Máy Java Độc Lập - NumJa User Guide
> **Phát hành phiên bản 0.2.0 - Closed-Source Release Model**
> Một giải pháp học máy độc lập, siêu gọn nhẹ, hiệu năng cao phỏng theo hệ sinh thái Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn) được tối ưu hóa cho môi trường Java offline (không Maven, không cài đặt phức tạp).

---

## 📦 Kiến Trúc Phân Phối Độc Lập (Multi-Module JARs)

Dự án được phân phối dưới dạng các mô-đun JAR độc lập, được biên dịch ở Java 11 (tương thích tối đa với Java 11 đến Java 25+) và đã qua bảo mật tối đa bằng công cụ **ProGuard Obfuscator** để chống dịch ngược mã nguồn:

```text
dist/
├── numja.jar          ✅ Mảng đa chiều NDArray & Đại số tuyến tính chuyên sâu (LinAlg)
├── pandas.jar         ✅ Bảng dữ liệu DataFrame, Series và I/O CSV cực mạnh
├── matplotlib.jar     ✅ Đồ thị 2D, Scatter Plot & Trình vẽ ảnh imshow() mịn điểm ảnh
├── seaborn.jar        ✅ Bản đồ nhiệtstatistical statistical statistical statistical Seaborn Heatmap
├── sklearn.jar        ✅ Công cụ học máy tối tân (50+ Classes Classifier, Regressor, Pipeline...)
└── libs/              ✅ Thư viện phụ thuộc bên thứ ba (Commons Math, EJML, JFreeChart...)
```

> [!IMPORTANT]  
> **Mô Hình Phân Phối Closed-Source:**  
> Toàn bộ mã nguồn cốt lõi trong `modules/` được cấu hình ẩn khỏi Git qua `.gitignore` để bảo vệ bản quyền nhà phát triển. Người dùng cuối và cộng tác viên chỉ cần kéo thư mục `dist/` về là có thể phát triển phần mềm ngay lập tức trên các IDE phổ biến (VS Code, IntelliJ) mà không cần mạng Internet.

---

## 🛠️ Hướng Dẫn Sử Dụng & Phát Triển Siêu Tốc

### 1. Cách Chạy Các Ví Dụ Mẫu Offline
Chúng tôi đã cung cấp sẵn tập lệnh PowerShell tiện ích để bạn chạy thử nhanh bất kỳ ví dụ nào trong thư mục `examples/`:

```powershell
# Chạy ví dụ đồ thị Matplotlib
./scripts/run_example.ps1 examples/plot/TestMatplotlib.java

# Chạy ví dụ tổng hợp đầy đủ kiểm định học máy & xử lý dữ liệu
./scripts/run_example.ps1 examples/TestComprehensive.java
```

### 2. Cách Biên Dịch & Chạy Chương Trình Của Riêng Bạn (Bằng Command Line)
Để viết ứng dụng client sử dụng thư viện NumJa của riêng bạn (ví dụ `YourApp.java`), hãy sử dụng lệnh sau:

**Biên dịch chương trình:**
```bash
javac -cp "dist/*;dist/libs/*" YourApp.java
```

**Thực thi chương trình:**
```bash
java -cp "dist/*;dist/libs/*;." YourApp
```

### 3. Thiết Lập Trên VS Code Độc Lập (Không Cần Maven)
Dự án đã được cấu hình sẵn môi trường phát triển tối ưu trên VS Code:
1. Mở thư mục gốc của dự án bằng **VS Code**.
2. Khi mở các tệp `.java` trong `examples/`, VS Code sẽ tự động nhận diện tất cả thư viện đóng gói trong `dist/` và `dist/libs/` thông qua cấu hình `.vscode/settings.json`.
3. Chỉ cần bấm nút **Run** / **Debug** (hoặc nhấn `F5`) ở góc trên bên phải để chạy trực tiếp (Chương trình sẽ tự động thực thi trong RAM thông qua tính năng JEP 330 của Java hiện đại mà không sinh tệp tin `.class` thừa).

---

## 🚀 Tính Năng Vượt Trội & Danh Sách Các Lớp (API Inventory)

### 📊 1. NumJa Core (Mảng & Đại Số Tuyến Tính)
Thiết kế mô phỏng cực sát cú pháp mảng đa chiều của **NumPy** trong Python.
* **`NumJa`**: Lớp factory tiện ích (`array()`, `zeros()`, `ones()`, `linspace()`, `mean()`, `std()`).
* **`NDArray`**: Bộ chứa mảng đa chiều hỗ trợ đầy đủ toán tử vector hóa, cộng trừ nhân chia vô hướng và **tự động Broadcasting** như Python.
* **`LinAlg`**: Giải các bài toán ma trận phức tạp:
  * `inv(NDArray)`: Nghịch đảo ma trận.
  * `det(NDArray)`: Định thức.
  * `solve(NDArray A, NDArray b)`: Giải hệ phương trình tuyến tính cực nhanh.
  * `svd()`, `qr()`, `eig()`, `lstsq()`: Phân tích trị riêng và bình phương tối thiểu.

### 🐼 2. Pandas (Xử Lý Dữ Liệu Lớn)
Cung cấp giải pháp phân tích dữ liệu dạng bảng tương đương **Pandas**.
* **`Pandas` (I/O)**: `read_csv()`, `read_json()` hỗ trợ các tham số chuyên sâu (`sep`, `skiprows`, `header`, `index_col`).
* **`DataFrame`**: Cấu trúc dữ liệu 2D gắn nhãn. Hỗ trợ đầy đủ:
  * `describe()`: Bảng tóm tắt thống kê mô tả toàn diện (mean, std, min, các percentile 25/50/75, max).
  * `groupby(col)`: Nhóm dữ liệu nâng cao.
  * `dropna()`, `fillna()`, `drop_duplicates()`: Làm sạch dữ liệu.
* **`DataFrameLoc` / `DataFrameILoc`**: Truy xuất dữ liệu nâng cao theo nhãn dòng (`loc`) hoặc theo tọa độ nguyên (`iloc`) cực kỳ linh hoạt (`df.iloc(1, 2)`).
* **`Series`**: Mảng 1D gắn nhãn để xử lý vector dữ liệu riêng lẻ.

### 📈 3. Matplotlib & Seaborn (Trực Quan Hóa Dữ Liệu)
Công cụ đồ họa mạnh mẽ kế thừa sức mạnh hiển thị điểm ảnh thực tế.
* **`Matplotlib` (State-Machine)**:
  * `plot(x, y)`: Vẽ đồ thị đường thẳng.
  * `scatter(x, y)`: Vẽ cụm điểm phân phối.
  * `title()`, `xlabel()`, `ylabel()`: Đặt tên tiêu đề và các trục đồ thị.
  * `clf()`: Xóa màn hình vẽ.
  * `show()`: Bật cửa sổ GUI hiển thị biểu đồ đồ họa thời gian thực.
  * `imshow(double[][])` / `imshow(NDArray)`: Vẽ điểm ảnh pixel siêu mịn, tự động kéo dãn phân giải và hiển thị thanh dải màu color-bar cao cấp.
* **`Seaborn`**:
  * `heatmap(int[][], String[] classes)`: Trực quan hóa ma trận nhầm lẫn (Confusion Matrix) dạng bản đồ nhiệt statistical statistical statistical statistical Seaborn Heatmap.

### 🤖 4. SKLearn (Hệ Thống Machine Learning Toàn Diện - 50+ Classes)
Bộ thư viện học máy đồ sộ bậc nhất trong Java, kế thừa triết lý lập trình của Python **Scikit-Learn**:

| Nhóm Tính Năng | Lớp / Lớp Phân Phối (Classes) | Chi Tiết Sử Dụng |
| :--- | :--- | :--- |
| **Supervised Classifiers** | `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`, `LogisticRegression`, `SVC` (SVM), `GaussianNB`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `MLPClassifier` | Hệ thống phân lớp cực kỳ đa dạng. Hỗ trợ đa luồng tự động (`n_jobs=-1`) cho Random Forest và KNN để tối ưu tối đa hiệu năng CPU. |
| **Supervised Regressors** | `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `RandomForestRegressor`, `KNeighborsRegressor`, `SVR`, `MLPRegressor` | Hồi quy tuyến tính, L1/L2 regularized, hồi quy phi tuyến đa lớp và hồi quy qua mạng Nơ-ron nhân tạo MLP. |
| **Data Preprocessing** | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `OneHotEncoder`, `PolynomialFeatures`, `SimpleImputer` | Chuẩn hóa chuẩn tắc Z-score, chuẩn hóa khoảng [0,1], xử lý dữ liệu khuyết thiếu (NaN), mã hóa nhãn targets và sinh các đặc trưng đa thức bậc cao. |
| **Model Selection** | `trainTestSplit()`, `trainTestSplitRegression()`, `CrossValidation` (K-Fold), `GridSearchCV` | Phân tách dữ liệu kiểm thử, kiểm định chéo và tự động tối ưu hóa siêu tham số (Hyperparameter Tuning) đa luồng. |
| **Pipelines** | `Pipeline` | Gom toàn bộ quy trình biến đổi dữ liệu (`SimpleImputer` ➔ `StandardScaler` ➔ `Model`) thành một chuỗi duy nhất tự động. |
| **Datasets** | `loadIris()`, `makeBlobs()`, `makeRegression()` | Trình nạp tập dữ liệu thực tế mẫu và trình sinh dữ liệu kiểm định phân cụm/hồi quy ngẫu nhiên. |

---

## 🌟 Ví Dụ Thực Tế: Xây Dựng Pipeline Học Máy Toàn Diện
Dưới đây là đoạn mã thực tế minh họa cách nạp dữ liệu bằng **Pandas**, chia dữ liệu và đưa vào **Pipeline** chuẩn hóa + dự đoán học máy cực kỳ ngắn gọn bằng thư viện **NumJa** của bạn:

```java
import pandas.DataFrame;
import pandas.Pandas;
import numja.core.NDArray;
import sklearn.model_selection.ModelSelection;
import sklearn.pipeline.Pipeline;
import sklearn.preprocessing.StandardScaler;
import sklearn.ensemble.RandomForest;
import sklearn.metrics.Metrics;

import java.util.ArrayList;
import java.util.List;

public class MyMLPipeline {
    public static void main(String[] args) throws Exception {
        // 1. Nạp dữ liệu bằng Pandas
        DataFrame df = Pandas.read_csv("dist/datasets/iris.csv");
        System.out.println("Shape dữ liệu: " + df.shape()[0] + " dòng x " + df.shape()[1] + " cột");

        // 2. Tách đặc trưng (Features) và nhãn (Target) dạng NDArray
        NDArray X = df.drop("species").toNDArray();
        int[] y = df.getColumn("species").toIntArray(); // giả sử nhãn đã được mã hóa

        // 3. Chia tập dữ liệu Train:Test tỉ lệ 70:30
        Object[] split = ModelSelection.trainTestSplit(X, y, 0.3, 42);
        NDArray X_train = (NDArray) split[0];
        NDArray X_test = (NDArray) split[1];
        int[] y_train = (int[]) split[2];
        int[] y_test = (int[]) split[3];

        // 4. Xây dựng Pipeline tự động (Chuẩn hóa StandardScaler -> Rừng Ngẫu Nhiên RandomForest)
        List<Object> steps = new ArrayList<>();
        steps.add(new StandardScaler());
        steps.add(new RandomForest(15, 5, 2, "classifier", -1)); // n_jobs=-1 dùng full luồng CPU
        Pipeline pipeline = new Pipeline(steps);

        // Fit mô hình trên tập huấn luyện
        pipeline.fit(X_train, y_train);

        // Dự đoán và tính toán độ chính xác
        int[] predictions = (int[]) pipeline.predict(X_test);
        double accuracy = Metrics.accuracyScore(y_test, predictions);
        System.out.printf("🎯 Độ chính xác kiểm thử của Pipeline: %.2f%%\n", accuracy * 100);
    }
}
```

---
**NumJa - Mang sức mạnh khoa học dữ liệu Python vào thế giới Java nguyên bản một cách đơn giản nhất!**
