# Research Summary: NumJa Performance & Scalability

> Generated 2026-08-25. **Note:** web search và subagent research bị lỗi hạ tầng trong phiên này — nội dung tổng hợp từ kiến thức nền của model, KHÔNG phải tra cứu live. Các version cần verify lại khi thực thi (đã ghi vào phase tương ứng).

## Stack (khuyến nghị kỹ thuật)

| Công nghệ | Vai trò | Khuyến nghị | Confidence |
|---|---|---|---|
| **JMH** (OpenJDK) | Benchmark harness | Chuẩn de-facto đo microbenchmark JVM. Tránh System.nanoTime() thủ công (JIT warmup, dead-code elimination) | High |
| **ParallelUtils mở rộng** (có sẵn) | CPU đa luồng | ForkJoinPool đã tồn tại trong codebase — mở rộng sang elementwise/reduce trước khi thêm lib mới | High |
| **Java Vector API** (`jdk.incubator.vector`) | SIMD | JDK 21+ có API ổn định hơn nhiều nhưng **vẫn incubator/preview** ở phần lớn JDK 21–25. Phải optional: runtime-check `Vector.getPlatformVectorShape()` style fallback thuần Java. Compile `--release 11` hiện tại không chứa Vector API → cần multi-release jar hoặc nâng profile compile cho module perf | Medium |
| **TornadoVM** | GPU offload thuần Java | Ứng viên GPU số 1 (bytecode→OpenCL/CUDA/SPIR-V). Rủi ro: setup phức tạp (JDK riêng, driver), chỉ hỗ trợ subset op, version thay đổi nhanh. Chỉ POC 1 op qua abstraction layer, không commit production ở milestone này | Medium |
| **Aparapi / JCuda / JavaCPP** | GPU alternatives | Aparapi ít bảo trì; JCuda = JNI binding CUDA (mất zero-install); JavaCPP nặng. Không khuyến nghị trừ khi TornadoVM thất bại | Low-Medium |
| **Kahan summation** | Accuracy | Thuật toán ~2 dòng, chi phí ~2x trên reduce nhưng chỉ áp cho reduce lớn (>10⁶ elements) hoặc song song theo tree-reduce | High |
| **Out-of-core chunking** | Large data | Tự viết chunk iterator trên read_csv/streaming — không có stdlib; thư viện JVM out-of-core DataFrame không có lựa chọn trưởng thành | Medium |

## Features (table stakes cho "ML library xử lý dữ liệu lớn")

- **Table stakes**: benchmark harness; multithreaded elementwise/reduce; stable numerics (softmax max-shift, Kahan); shape-mismatch error messages rõ ràng
- **Differentiators**: adaptive in-memory/out-of-core switch minh bạch với caller; ComputeBackend abstraction sẵn sàng GPU/TPU
- **Anti-features**: không bắt user chọn backend thủ công cho từng op; không breaking change

## Architecture hướng đi

1. **Backend-first layering**: định nghĩa `ComputeBackend` interface (op-level: map, reduce, gemm) → `CpuThreadBackend` cài đầu tiên → GPU backend sau cùng interface đó. Dispatch tự động theo size threshold.
2. **Chunk pipeline**: `Pandas.read_csv` streaming mode trả về iterator of DataFrames; aggregation ops khai báo associative để gộp cross-chunk.
3. **Benchmark làm phase đầu**: mọi tối ưu sau đó so baseline, tránh premature optimization.

## Pitfalls (cảnh báo chính)

1. **Microbenchmark sai**: đo bằng nanoTime không warmup → số liệu vô nghĩa. Dùng JMH, fork JVM, average time.
2. **Multithread overhead trên mảng nhỏ**: parallel split cost > gain khi n < ~100k. Cần threshold.
3. **SIMD dead-code elimination**: JIT xóa loop nếu kết quả không dùng — JMH blackhole bắt buộc.
4. **GPU offload ảo tưởng tốc độ**: transfer CPU↔GPU thường đắt hơn tính toán với op nhỏ; chỉ win với batch lớn, op lặp lại.
5. **Numerics khi song song**: sum tuần tự ≠ sum tree/kahan — golden-value tests phải chạy cả hai path.
6. **API freeze vs internals**: đổi signature NDArray (thêm backend field) vẫn là breaking change nếu serialization/reflection phụ thuộc — kiểm tra trước.

## Gaps cần verify khi thực thi (do research offline)

- Version TornadoVM mới nhất + JDK tương thích
- Trạng thái Vector API final/incubator trên JDK target
- EJML có bản multithreaded mt-0.43.x hay không (tồn tại artifact `ejml-*` mt variants)
