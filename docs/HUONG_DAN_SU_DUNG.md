# Hướng Dẫn Sử Dụng Nhanh (Offline & Độc Lập)

Dự án đã được thiết lập để có thể chạy **hoàn toàn độc lập** mà không cần cài đặt Maven, không cần kết nối mạng tải thư viện, hay các cấu hình phức tạp nào khác. Bạn chỉ cần gửi cho người khác thư mục này (hoặc chỉ file `.jar`).

## 1. Cách xây dựng (Build) và Chạy (Run) thủ công bằng PowerShell (Không dùng Maven)
Trong trường hợp bạn đã thay đổi code và muốn biên dịch toàn bộ các file `.java` (bao gồm code library và code ví dụ) mà không phụ thuộc vào `build.bat` cũ hay Maven, và tránh xung đột với các gói JAR bản cũ đang để sẵn ở thư mục `libs`, bạn dán lệnh sau vào PowerShell:

**Biên dịch:**
```powershell
$jars = Get-ChildItem -Path libs -Filter *.jar | Where-Object { $_.Name -notmatch "numja-core" -and $_.Name -notmatch "plot" } | Select-Object -ExpandProperty FullName
$cp = "out;" + ($jars -join ";")
javac -cp $cp -d out (@(Get-ChildItem -Path modules, examples -Recurse -Filter *.java | Where-Object { $_.FullName -notmatch "src\\test" -and $_.FullName -notmatch "target" } | Select-Object -ExpandProperty FullName))
```

**Chạy ví dụ đồ thị (Matplotlib / Seaborn):**
```powershell
# Chạy TestMatplotlib
java -cp $cp examples.plot.TestMatplotlib

# Chạy TestSeaborn
java -cp $cp examples.plot.TestSeaborn
```

Sau khi chạy xong, kết quả đồ thị sẽ được pop-up hiển thị bình thường.

## 2. Cách xây dựng (Build) một lần
Để biên dịch toàn bộ dự án và gom tất cả các thư viện phụ thuộc vào một file `.jar` duy nhất (`numja-all.jar`), bạn chỉ cần chạy 1 trong 2 file sau (tuỳ thuộc vào việc bạn dùng Command Prompt hay PowerShell):
- Chạy file `build.bat` (Nếu dùng Command Prompt)
- Chạy file `build.ps1` (Nếu dùng PowerShell)

Sau khi chạy xong, file `numja-all.jar` sẽ được tạo ra trong thư mục `build/`.

## 2. Cách chạy ứng dụng / thư viện
Khi đã có file `build/numja-all.jar`, bạn và bất kỳ ai khác có thể chạy nó ở bất kỳ đâu chỉ với Java:

**Chạy test nội bộ của thư viện:**
```bat
java -jar build\numja-all.jar
```

**Chạy file ví dụ `NumJaClientExample`:**
```bat
java -cp build\numja-all.jar NumJaClientExample
```

## 3. Cách sử dụng (Run / Debug) trực tiếp trên VS Code
Mọi cấu hình (`.vscode/settings.json` và `.vscode/launch.json`) đã được thiết lập sẵn:
1. Mở thư mục dự án bằng **VS Code**.
2. Mở file bạn muốn chạy (ví dụ `examples/NumJaClientExample.java` hoặc `src/test/java/com/numja/tests/NumJaTest.java`).
3. Nhấn nút **Run** (hoặc **Debug**) ở góc trên bên phải, hoặc chọn cấu hình trong tab **Run and Debug** bên trái. Code sẽ tự động nhận diện tất cả thư viện trong thư mục `libs/`.

## 4. Cách sử dụng thư viện cho dự án khác của bạn
Bạn chỉ cần gửi file `build/numja-all.jar` cho người khác. Họ có thể viết code Java và gọi thư viện này bằng `javac` và `java` bình thường:

Biên dịch file của họ (Ví dụ: `YourApp.java`):
```bat
javac -cp build\numja-all.jar YourApp.java
```

Chạy file của họ:
```bat
java -cp "build\numja-all.jar;." YourApp
```

## 5. Danh sách các tính năng / API đã xây dựng

Thư mục dự án cung cấp các API được thiết kế mô phỏng rất sát với Cú pháp của Python (NumPy, Pandas, Matplotlib, Seaborn).

### 5.1 NumJa (Mô phỏng NumPy) - `NDArray`
- `T()`: Ma trận chuyển vị (Transpose).
- `matmul(NDArray)`, `dot(NDArray)`: Nhân hai ma trận.
- `reshape(int...)`: Đổi hình dạng (kích thước) mảng.
- `add()`, `subtract()`, `multiply()`: Hỗ trợ cộng/trừ/nhân vô hướng và tự động broadcasting.

### 5.2 Pandas (Mô phỏng Pandas) - `DataFrame` & `Series`
- **`Pandas` (I/O):**
  - `read_csv(String path, ReadCsvOptions options)`: Đọc tập tin CSV hỗ trợ tùy chỉnh `sep`, `skiprows`, `header`, `index_col`.
- **`Series`:** Mô phỏng kiểu dữ liệu mảng 1D đi kèm nhãn (index).
- **`DataFrame`:** 
  - `head(n)`: Xem nhanh dữ liệu đầu.
  - `shape()`: Lấy kích thước (số dòng, số cột).
  - `columns()`, `index()`: Lấy danh sách tên cột và các nhãn (labels).
  - Khả năng lọc và truy xuất thông qua các thành phần chuyên biệt: `loc` (lọc theo label), `iloc` (lọc theo tọa độ index nguyên).
  - `describe()`: Tính toán các chỉ số thống kê tổng hợp (`count`, `mean`, `std`, `min`, `max`, các `percentile` 25%, 50%, 75%).

### 5.3 Plot - Matplotlib (Mô phỏng `matplotlib.pyplot`)
Hoạt động dạng State-machine (duy trì trạng thái khung hình).
- `plot(x, y)`: Vẽ đồ thị dạng đường.
- `scatter(x, y)`: Vẽ cụm điểm phân tán.
- `title(String)`, `xlabel(String)`, `ylabel(String)`: Đặt tiêu đề và tên trục.
- `clf()`: Xóa sạch đồ thị hiện tại để bắt đầu hình mới.
- `show()`: Bật cửa sổ đồ họa GUI (chặn chương trình chờ đến khi đóng cửa sổ).

### 5.4 Plot - Seaborn (Mô phỏng `seaborn`)
Tương tác trực tiếp sử dụng đối tượng `DataFrame`.
- `plot(DataFrame df, String x_col, String y_col)`: Gắn liền tên dữ liệu trong DataFrame thành biểu đồ.
- `scatter(DataFrame df, String x_col, String y_col)`: Trực quan dạng phân tán bằng số liệu lấy từ DataFrame.
