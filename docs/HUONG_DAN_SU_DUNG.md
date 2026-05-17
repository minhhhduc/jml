# 🏆 Tổng Quan Tính Năng & Danh Sách Mô-đun (NumJa API Overview)
> **Phát hành phiên bản 0.2.0 - Closed-Source Release Model**
> Tài liệu này mô tả chi tiết toàn bộ các lớp (Classes), phương thức (APIs) và cấu trúc tính năng được triển khai trong hệ sinh thái **NumJa** (mô phỏng theo hệ sinh thái Khoa học Dữ liệu & Học máy của Python bao gồm NumPy, Pandas, Matplotlib, Seaborn, và Scikit-Learn).

---

## 📊 1. Mô-đun NumJa Core (Mô Phỏng NumPy)
Mô-đun cốt lõi xử lý mảng đa chiều hiệu năng cao, đại số tuyến tính chuyên sâu và các toán tử toán học vector hóa.

### Lớp chính: `numja.core.NDArray`
Bộ chứa mảng đa chiều trung tâm phỏng theo `numpy.ndarray`.
* **Khởi tạo & Định dạng:**
  * `shape()` / `getShape()`: Trả về hình dạng ma trận (số chiều, số cột, số dòng).
  * `reshape(int...)`: Tái cấu trúc hình dạng mảng mà không thay đổi dữ liệu.
  * `T()`: Phép chuyển vị ma trận nhanh (Transpose).
  * `toArray()` / `toArray2D()`: Xuất dữ liệu mảng ra mảng Java nguyên bản (`double[]` hoặc `double[][]`).
* **Toán tử Vector hóa & Tự động Broadcasting:**
  * `add(NDArray)` / `add(double)`: Cộng hai mảng hoặc cộng một hằng số với cơ chế tự động căn chỉnh kích thước chiều tự động (Broadcasting).
  * `subtract(NDArray)` / `subtract(double)`: Phép trừ vector hóa tự động phát sóng.
  * `multiply(NDArray)` / `multiply(double)`: Nhân từng phần tử (Element-wise multiplication).
  * `divide(NDArray)` / `divide(double)`: Chia từng phần tử (Element-wise division).
  * `dot(NDArray)` / `matmul(NDArray)`: Nhân ma trận (Matrix multiplication) tối ưu hóa đa luồng.

### Lớp Tiện ích: `numja.NumJa`
Lớp factory chứa các phương thức khởi tạo mảng tĩnh và các hàm thống kê tiện ích tương đương `numpy`:
* `array(double[][])` / `array(double...)`: Tạo `NDArray` nhanh từ dữ liệu Java nguyên bản.
* `zeros(int...)` / `ones(int...)`: Khởi tạo nhanh mảng toàn số 0 hoặc số 1.
* `linspace(double start, double stop, int num)`: Tạo dãy số phân bố đều tuyến tính.
* `mean(NDArray)` / `std(NDArray)`: Tính trung bình cộng và độ lệch chuẩn của mảng.

### Thư viện Đại số tuyến tính chuyên sâu: `numja.linalg.LinAlg`
Cung cấp các phép phân tích và biến đổi ma trận chuyên nghiệp:
* `inv(NDArray)`: Nghịch đảo ma trận (Matrix Inverse).
* `det(NDArray)`: Tính định thức của ma trận vuông (Determinant).
* `solve(NDArray A, NDArray b)`: Giải hệ phương trình tuyến tính $Ax = b$ tốc độ cao.
* `trace(NDArray)`: Tính vết của ma trận (Tổng đường chéo chính).
* `svd(NDArray)`: Phân tích suy hao kỳ dị (Singular Value Decomposition), trả về kết quả lớp chứa `SVD`.
* `qr(NDArray)`: Phân tích QR (QR Decomposition), trả về kết quả lớp chứa `QR`.
* `eig(NDArray)`: Tính toán trị riêng và vector riêng (Eigenvalues & Eigenvectors), trả về đối tượng `Eigen`.
* `lstsq(NDArray A, NDArray b)`: Giải bài toán bình phương tối thiểu (Ordinary Least Squares), trả về kết quả `Lstsq`.

---

## 🐼 2. Mô-đun Pandas (Xử Lý Dữ Liệu Lớn)
Hệ thống xử lý, phân tích và làm sạch dữ liệu có cấu trúc phỏng theo thư viện **Pandas** trong Python.

### Lớp Loader: `pandas.Pandas`
Trình đọc và ghi tệp tin nâng cao:
* `read_csv(String path)` / `read_csv(String path, ReadCsvOptions options)`: Bộ nạp CSV mạnh mẽ, hỗ trợ các tùy chọn chuyên sâu như dấu phân cách (`sep`), bỏ qua dòng (`skiprows`), tiêu đề dòng (`header`), và cột chỉ mục (`index_col`).
* `read_json(String path)`: Nạp dữ liệu cấu trúc JSON trực tiếp vào DataFrame.

### Lớp Trung tâm: `pandas.DataFrame`
Cấu trúc dữ liệu 2D dạng bảng với nhãn dòng (Index) và nhãn cột (Columns).
* **Kiểm tra thông số & Xem nhanh:**
  * `shape()`: Lấy kích thước bảng (Dòng x Cột).
  * `head(n)` / `tail(n)`: Xem $n$ bản ghi đầu tiên hoặc cuối cùng.
  * `columns()`: Trích xuất danh sách tên cột dữ liệu.
  * `describe()`: Tính toán nhanh bảng thống kê mô tả bao gồm: số lượng mẫu (`count`), giá trị trung bình (`mean`), độ lệch chuẩn (`std`), giá trị nhỏ nhất (`min`), các phân vị (`25%`, `50%`, `75%`), và giá trị lớn nhất (`max`).
* **Lọc & Biến đổi dữ liệu:**
  * `loc` (qua lớp `DataFrameLoc`): Lọc trích xuất dữ liệu dựa trên nhãn tên (Label-based indexer).
  * `iloc` (qua lớp `DataFrameILoc`): Lọc trích xuất dữ liệu dựa trên tọa độ vị trí nguyên (Integer-based indexer).
  * `drop(String...)` / `drop_duplicates()`: Loại bỏ các cột không dùng hoặc loại bỏ dòng trùng lặp.
  * `dropna()` / `fillna(double)`: Xử lý làm sạch giá trị bị khuyết thiếu (NaN).
  * `sort_values(String col)`: Sắp xếp bảng theo giá trị của cột xác định.
  * `groupby(String col)`: Gom nhóm dữ liệu nâng cao phỏng theo `pandas.DataFrame.groupby`.
  * `toNDArray()`: Chuyển đổi bảng số liệu trực tiếp thành mảng `NDArray`.

### Lớp Vector 1D: `pandas.Series`
Biểu diễn một cột dữ liệu 1D gắn nhãn đi kèm. Hỗ trợ đầy đủ các phép toán thống kê mô tả riêng lẻ và chuyển đổi ngược về mảng Java.

---

## 📈 3. Mô-đun Matplotlib (Trực Quan Hóa 2D)
Mô phỏng cơ chế máy trạng thái (State-machine) vẽ đồ thị của **matplotlib.pyplot**.

### Lớp chính: `matplotlib.Matplotlib`
* `plot(x, y)` / `plot(NDArray x, NDArray y)`: Vẽ đồ thị dạng đường (Line plot). Hỗ trợ nạp mảng `NDArray` trực tiếp.
* `scatter(x, y)` / `scatter(NDArray x, NDArray y)`: Vẽ biểu đồ cụm điểm phân phối (Scatter plot).
* `title(String)`: Thiết lập tiêu đề phía trên đồ thị.
* `xlabel(String)` / `ylabel(String)`: Gán nhãn cho trục hoành và trục tung.
* `clf()`: Xóa sạch khung hình hiện tại để chuẩn bị vẽ biểu đồ mới.
* `show()`: Khởi chạy cửa sổ đồ họa GUI hiển thị hình vẽ trực quan thời gian thực.
* `savefig(String path)`: Xuất đồ thị đang vẽ ra file ảnh tĩnh (PNG).
* `imshow(double[][])` / `imshow(NDArray)`: Trình trực quan hóa hình ảnh dạng lưới pixel mịn cao cấp, hỗ trợ dải màu gradient color-bar chuyên nghiệp mà không có các số phân mảnh ô đè lên ảnh.

---

## 📊 4. Mô-đun Seaborn (Thống Kê Trực Quan)
Cung cấp các biểu đồ thống kê dạng nâng cao phỏng theo **Seaborn** của Python, hoạt động liên kết chặt chẽ với đối tượng `DataFrame`.

### Lớp chính: `seaborn.Seaborn`
* `heatmap(int[][] confusionMatrix, String[] classes)` / `heatmap(NDArray matrix, String[] classes)`: Trực quan hóa ma trận nhầm lẫn (Confusion Matrix) dạng bản đồ nhiệt sang trọng, tự động tích hợp thanh dải màu gradient hiển thị mật độ số liệu phân bổ.
* `load_dataset(String name)`: Nạp tự động các tập dữ liệu thống kê kiểm định thực tế có sẵn.

---

## 🤖 5. Mô-đun SKLearn (Hệ Thống Học Máy Scikit-Learn - 50+ Classes)
Bộ thư viện học máy đồ sộ được module hóa chặt chẽ, tối ưu hóa xử lý đa luồng phỏng theo **scikit-learn** của Python.

### 5.1 Các Bộ Phân Lớp Giám Sát (Supervised Classifiers)
* `sklearn.tree.DecisionTreeClassifier`: Bộ phân lớp Cây quyết định dựa trên tiêu chí phân chia tối ưu Entropy/Gini.
* `sklearn.ensemble.RandomForestClassifier`: Mô hình Rừng ngẫu nhiên song song đa luồng. Hỗ trợ tham số `n_jobs=-1` để kích hoạt toàn bộ các nhân xử lý CPU tăng tốc độ huấn luyện.
* `sklearn.neighbors.KNeighborsClassifier`: Bộ phân lớp K-Láng giềng gần nhất hỗ trợ thuật toán cây tìm kiếm K-D Tree nâng cao.
* `sklearn.linear_model.LogisticRegression`: Phân lớp tuyến tính Logistic Gradient Descent.
* `sklearn.svm.SVC`: Bộ phân lớp Vector hỗ trợ (Support Vector Machine) hỗ trợ các không gian hạt nhân (Kernel functions) đa dạng.
* `sklearn.naive_bayes.GaussianNB`: Phân lớp xác suất Gaussian Naive Bayes.
* `sklearn.ensemble.GradientBoostingClassifier`: Bộ tăng cường độ dốc Gradient Boosting Classifier tuần tự.
* `sklearn.ensemble.AdaBoostClassifier`: Bộ tăng cường AdaBoost Classifier tuần tự.
* `sklearn.neural_network.MLPClassifier`: Mạng Nơ-ron nhân tạo đa tầng (Multi-Layer Perceptron Classifier) hỗ trợ tùy biến các hàm kích hoạt (Sigmoid, ReLU, Tanh, Softmax) và tối ưu hóa lan truyền ngược.

### 5.2 Các Bộ Hồi Quy Giám Sát (Supervised Regressors)
* `sklearn.linear_model.LinearRegression`: Hồi quy tuyến tính bình phương tối thiểu.
* `sklearn.linear_model.Ridge`: Hồi quy Ridge tích hợp ràng buộc L2 regularization chống quá khớp (overfitting).
* `sklearn.linear_model.Lasso`: Hồi quy Lasso tích hợp ràng buộc L1 regularization tối ưu hóa lựa chọn đặc trưng thưa.
* `sklearn.tree.DecisionTreeRegressor` / `sklearn.ensemble.RandomForestRegressor`: Cây quyết định và rừng ngẫu nhiên áp dụng cho bài toán dự báo liên tục.
* `sklearn.neighbors.KNeighborsRegressor` / `sklearn.svm.SVR` / `sklearn.neural_network.MLPRegressor`: Các biến thể hồi quy K-Láng giềng gần nhất, SVM Regressor, và Mạng nơ-ron hồi quy MLP.

### 5.3 Tiền Xử Lý Dữ Liệu & Điền Khuyết (Data Preprocessing)
* `sklearn.preprocessing.StandardScaler`: Chuẩn hóa dữ liệu theo phân phối chuẩn tắc (Z-score normalization: mean = 0, std = 1).
* `sklearn.preprocessing.MinMaxScaler`: Chuẩn hóa mảng dữ liệu về phạm vi xác định (mặc định $[0, 1]$).
* `sklearn.preprocessing.RobustScaler`: Chuẩn hóa kháng nhiễu dựa trên trung vị (Median) và khoảng tứ phân vị (IQR), tối ưu cho dữ liệu chứa nhiều ngoại lai (Outliers).
* `sklearn.preprocessing.LabelEncoder`: Mã hóa nhãn văn bản danh mục thành các chỉ số nguyên.
* `sklearn.preprocessing.OneHotEncoder`: Mã hóa One-Hot chuyển đổi biến danh mục thành ma trận nhị phân thưa.
* `sklearn.preprocessing.PolynomialFeatures`: Tạo thêm các biến đặc trưng đa thức bậc cao (Polynomial degree expansion).
* `sklearn.impute.SimpleImputer`: Tự động điền các giá trị trống (NaN) bằng phương pháp trung bình (Mean), trung vị (Median), hoặc giá trị hằng số xác định.

### 5.4 Lựa Chọn Mô Hình & Xích Quy Trình (Model Selection & Pipelines)
* `sklearn.model_selection.ModelSelection`: 
  * `trainTestSplit()`: Phân tách dữ liệu Train/Test theo tỷ lệ mong muốn đi kèm thiết lập seed cố định ngẫu nhiên (`random_state`).
  * `crossValScore()`: Đánh giá chéo K-Fold Cross Validation.
* `sklearn.model_selection.GridSearchCV`: Tìm kiếm lưới tự động để tối ưu hóa siêu tham số (Hyperparameter tuning) hỗ trợ thực thi song song đa luồng (`n_jobs=-1`).
* `sklearn.pipeline.Pipeline`: Cho phép gom toàn bộ quy trình tiền xử lý và mô hình học máy thành một chuỗi duy nhất nối tiếp (`Imputer` ➔ `Scaler` ➔ `Estimator`), tự động kích hoạt tuần tự qua lệnh gọi `.fit()` và `.predict()`.

### 5.5 Bộ dữ liệu mẫu (Datasets)
* `sklearn.datasets.Datasets`: Cung cấp các phương thức nạp dữ liệu mẫu nhanh như `loadIris()`, sinh dữ liệu hồi quy ngẫu nhiên `makeRegression()`, sinh dữ liệu phân cụm `makeBlobs()`.
* `sklearn.datasets.Bunch`: Bộ chứa kiểu Dictionary tương tự Python, cung cấp các phương thức `.getData()` và `.getTarget()` lấy nhanh số liệu đặc trưng.
