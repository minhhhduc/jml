# Requirements: NumJa Performance & Scalability

> Generated 2026-08-25. v1 scope for this milestone.

## v1 Requirements

### Benchmark & Measurement
- [ ] **BENCH-01**: Developer có thể chạy benchmark harness (một lệnh) đo throughput/latency cho core ops: NDArray matmul, elementwise ops, DataFrame groupby, sklearn fit/predict
- [ ] **BENCH-02**: Mỗi op có baseline số liệu được ghi lại để so sánh trước/sau tối ưu
- [ ] **BENCH-03**: Benchmark tự động phát hiện suy giảm hiệu năng > ngưỡng (regression gate)

### Adaptive Memory
- [ ] **MEM-01**: Dataset nhỏ hơn memory threshold xử lý in-memory như API hiện tại, không đổi hành vi
- [ ] **MEM-02**: Dataset vượt threshold tự động xử lý theo chunk (out-of-core) mà caller không cần sửa code
- [ ] **MEM-03**: Ops hỗ trợ chunking: read_csv → column stats, groupby aggregations, fit cho các model hỗ trợ partial fit / streaming

### CPU Throughput
- [ ] **CPU-01**: Elementwise ops của NDArray (add, mul, exp...) chạy đa luồng qua ParallelUtils với speedup đo được trên multi-core
- [ ] **CPU-02**: Reduce operations (sum, mean, min/max) đa luồng với numerical stability giữ nguyên
- [ ] **CPU-03**: SIMD (Java Vector API) áp dụng cho hot loops nếu benchmark chứng minh lợi ích; có fallback thuần Java khi JDK không hỗ trợ

### Hardware Abstraction
- [ ] **HW-01**: Interface ComputeBackend cho phép route một op tới backend khác nhau (CPU threads mặc định)
- [ ] **HW-02**: Op dispatch tự động chọn backend theo khả năng máy + kích thước công việc
- [ ] **HW-03**: Research document kết luận GPU/TPU backend (TornadoVM vs alternatives) với POC benchmark ít nhất 1 op

### Numerical Accuracy
- [ ] **ACC-01**: Reduce lớn dùng Kahan/compensated summation — sai số không tăng theo kích thước dữ liệu
- [ ] **ACC-02**: Softmax/log/exp stable (max-shift) — không overflow/underflow ở giá trị biên
- [ ] **ACC-03**: Kết quả benchmark ops khớp golden values tham chiếu NumPy trong tolerance ghi rõ

### Usability
- [ ] **USE-01**: Lỗi shape/dtype mismatch có message nêu rõ shape hai phía và op đang chạy
- [ ] **USE-02**: Có helper factory/builder dựng pipeline dữ liệu lớn (load → chunk → transform → fit) dưới 10 dòng code

## v2 Requirements (deferred)

- Distributed/cluster execution (Spark-like)
- Lazy evaluation + fluent query API cho DataFrame
- GPU backend production-ready cho toàn bộ op set
- Thêm model ML mới

## Out of Scope

- Breaking change API công khai — Core Value cấm
- Native JNI tự viết — mất tính zero-install, chỉ dùng lib tăng tốc có sẵn nếu research chỉ ra cần thiết
- UI/dashboard — library only
- Đa máy phân tán — quy mô 1 máy

## Traceability

<!-- Roadmap fills this in: requirement → phase mapping -->

| Requirement | Phase | Status |
|-------------|-------|--------|
| (pending roadmap) | | |
