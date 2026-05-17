# 📝 Nhật Ký Thay Đổi (Changelog)

Tất cả các thay đổi đáng chú ý đối với dự án này sẽ được ghi tài liệu trong tệp này.
Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/vi/1.0.0/).

## [0.2.0] - 2026-05-17
### Thêm mới (Added)
- **Học máy & Tập dữ liệu (`sklearn.datasets`)**: Tích hợp lớp nạp dữ liệu cao cấp `Datasets` cùng đối tượng `Bunch`, phương thức nạp dữ liệu di động tương đối `loadIris()`, bộ sinh dữ liệu phân cụm `makeBlobs()` và bộ sinh dữ liệu hồi quy `makeRegression()`.
- **Trình vẽ pixel-level `imshow()`**: Xây dựng lại hoàn toàn hàm `Matplotlib.imshow()` vẽ điểm ảnh pixel mịn thực tế thông qua `ImshowPanel` không có lưới số đè lên, tự động kéo dãn độ phân giải ảnh, hỗ trợ nạp và trực quan hóa ảnh xám thời gian thực.
- **Hỗ trợ `NDArray` trực tiếp cho Đồ thị**: Tích hợp nạp `NDArray` trực tiếp không cần `.toArray()` cho tất cả các phương thức vẽ `plot()`, `scatter()`, `heatmap()`, `imshow()` của `Matplotlib` và `Seaborn`.

### Cập nhật (Changed)
- **Closed-Source Release Git Model**: Tái cấu trúc cấu hình `.gitignore` để chuyển dự án sang mô hình phân phối mã nguồn đóng. Toàn bộ mã nguồn cốt lõi trong `modules/` được bảo vệ ẩn hoàn toàn khỏi Git, trong khi các JAR thành phẩm đã được ProGuard mã hóa bảo mật tối đa trong `dist/` được công khai để người dùng khác tải và lập trình trực tiếp.

## [0.1.0] - 2026-05-17
### Thêm mới (Added)
- Core: Các hàm toán học, `NDArray` và thao tác đại số tuyến tính (`solve`, `det`, `eig`).
- Pandas: Cấu trúc `DataFrame`, `Series`, khả năng đọc/ghi file CSV (`read_csv` tái lập kwargs của python pandas).
- Plot: Tích hợp thư viện đồ thị JFreeChart mô phỏng `matplotlib` và `seaborn`.
- Build & Run Tool: Tập lệnh build siêu tốc cho nền tảng Windows (`build_core.ps1`, `build_check.ps1`).
- Tính năng chạy mọi nơi: Phát hành `dist/` cùng `nj.ps1` để thực thi file dễ dàng.
- Tương thích IDE hoàn hảo: Tích hợp thiết lập thư mục trực tiếp cho VS Code (Không cần Maven).

### Cập nhật (Changed)
- Hạ tương thích biên dịch ngôn ngữ (Target Language Level) từ JDK tĩnh 25 xuống **JDK 17**.
- Cho phép chạy tệp tin duy nhất (JEP 330) trên các phiên bản Java hiện đại không cần tạo mã file class rác.

### Đã Fix (Fixed)
- Sửa lỗi khai báo package không hợp lệ ở các thư mục `examples/*`.
- Sửa lỗi VS Code nhận nhầm và đính class paths không cần thiết vào preLaunchTasks.