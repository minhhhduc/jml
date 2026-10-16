# 🤝 Đóng Góp (Contributing)

Cảm ơn bạn đã quan tâm và muốn đóng góp cho dự án **NumJa**! Dưới đây là quy trình đơn giản để góp sức:

## 1. Báo lỗi & Góp ý (Issues)
- Nếu bạn gặp lỗi khi chạy `nj` hoặc compile, hãy tạo một Issue bằng ngôn ngữ tiếng Việt (hoặc tiếng Anh).
- Kèm theo log hoặc ảnh chụp màn hình bị lỗi.

## 2. Quy trình Gửi Pull Request (PR)
1. **Fork** dự án này về tài khoản GitHub của bạn.
2. **Clone** repo vừa fork về máy:
   ```bash
   git clone https://github.com/your-username/java_ml.git
   ```
3. Tạo nhánh mới (`git checkout -b feature/them-tinh-nang-moi`).
4. Viết code (nhớ chạy thử bằng cách vào thư mục `examples` và gõ `java -cp...`).
5. Build test để chắc chắn không lỗi:
   ```powershell
   cd scripts
   .\build_core.ps1 -Module all
   ```
6. Commit và đẩy nhánh của bạn lên (`git push origin feature/them-tinh-nang-moi`).
7. Tạo **Pull Request** vào nhánh `main` của dự án gốc.

## 3. Quy Tắc Viết Code (Coding Conventions)
- Mọi hàm trong `NumJa`, `Pandas`, `Matplotlib` nên bám sát hệ sinh thái Python (VD: `arange`, `linspace`, `mean`).
- Hãy viết tài liệu JavaDoc (`/** ... */`) cho các phương thức public mới.
- Hạn chế tối đa việc sử dụng thêm các thư viện third-party bên ngoài `pom.xml` định sẵn để giữ project nhẹ nhất có thể.

Một lần nữa, xin cảm ơn!