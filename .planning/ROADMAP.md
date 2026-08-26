# ROADMAP: NumJa Performance & Scalability

> Generated 2026-08-25. Work happens on branch `dev` (from `revent-backup` — contains full source, pom.xml, JUnit tests).

## Phases

### Phase 1: Restore Dev Environment & Baseline Benchmark
**Goal:** Source từ nhánh dev chạy được, có benchmark harness đo baseline mọi core op — con số trước khi tối ưu.
**Mode:** mvp
**Success Criteria**:
1. `mvn test` pass trên nhánh dev với toàn bộ test suite hiện có
2. JMH harness chạy được bằng một lệnh, đo: NDArray matmul/elementwise/reduce, DataFrame groupby, sklearn fit/predict
3. File baseline kết quả benchmark được ghi lại và commit
4. TornadoVM/Vector API version check hoàn tất (verify các gap trong research/SUMMARY.md)

Requirements: BENCH-01, BENCH-02 (+ verify research gaps)

### Phase 2: CPU Parallel Core (Threads)
**Goal:** Elementwise ops và reduce của NDArray chạy đa luồng qua ParallelUtils mở rộng, speedup đo được trên multi-core, không regress ở mảng nhỏ.
**Mode:** mvp
**Success Criteria**:
1. Elementwise ops (add/mul/exp...) đạt speedup ≥2x trên mảng lớn (≥10⁶ elements) máy multi-core so baseline Phase 1
2. Mảng nhỏ (<100k) không chậm hơn baseline quá 10% (size threshold dispatch)
3. Reduce ops đa luồng cho kết quả khớp tuần tự trong tolerance
4. Toàn bộ test cũ vẫn pass

Requirements: CPU-01, CPU-02

### Phase 3: Numerical Accuracy Hardening
**Goal:** Sai số không tăng theo kích thước dữ liệu; kết quả khớp NumPy golden values.
**Mode:** mvp
**Success Criteria**:
1. Kahan/tree summation áp dụng cho reduce lớn — sai số tuyệt đối giảm đo được so naive song song
2. Softmax/log/exp stable (max-shift) không overflow tại giá trị biên (test case ±1e300)
3. Golden-value test suite so với tham chiếu NumPy pass trong tolerance ghi rõ cho mọi op benchmarked
4. Regression gate (BENCH-03) chạy được: cảnh báo khi perf tụt > ngưỡng hoặc sai số vượt tolerance

Requirements: ACC-01, ACC-02, ACC-03, BENCH-03

### Phase 4: Adaptive Memory Model
**Goal:** Dataset vượt RAM xử lý tự động theo chunk/out-of-core — caller không sửa code.
**Mode:** mvp
**Success Criteria**:
1. read_csv streaming mode trả chunk iterator; column stats/groupby aggregation chạy đúng trên file > RAM available (test với giới hạn heap nhỏ -Xmx giả lập)
2. Kết quả out-of-core khớp in-memory trong tolerance trên cùng dataset nhỏ
3. API in-memory hiện tại hành vi không đổi với dataset nhỏ
4. USE-02: pipeline helper dựng load→chunk→transform→fit dưới 10 dòng

Requirements: MEM-01, MEM-02, MEM-03, USE-02

### Phase 5: Hardware Abstraction Layer & GPU POC
**Goal:** ComputeBackend interface với CPU backend mặc định; POC GPU qua 1 op chứng minh kiến trúc mở rộng được sang GPU/TPU.
**Mode:** mvp
**Success Criteria**
1. ComputeBackend interface cài CpuThreadBackend, dispatch tự động theo size/khả năng máy, default path = behavior cũ
2. POC document + benchmark: ít nhất 1 op (gemm) chạy qua TornadoVM (hoặc alternative đã research) với số đo so CPU
3. Kết luận go/no-go cho GPU production backend ở milestone sau, dựa trên số liệu
4. Public API không thay đổi (backend selection nội bộ)

Requirements: HW-01, HW-02, HW-03

### Phase 6: Usability Polish & Release
**Goal:** Error messages rõ ràng, docs cập nhật, obfuscated release jars mới.
**Mode:** mvp
**Success Criteria**
1. Shape/dtype mismatch error nêu shape hai phía + op name (test case xác nhận message)
2. Docs (README, QUICK_REFERENCE) phản ánh tính năng mới: streaming mode, accuracy guarantees, benchmark tool
3. ProGuard build pipeline chạy ra dist/*.jar v0.3.0, examples chạy thành công với jars mới
4. CHANGELOG ghi đầy đủ thay đổi

Requirements: USE-01 (+ release hygiene)

## Requirement Coverage

| Requirement | Phase |
|---|---|
| BENCH-01 | 1 |
| BENCH-02 | 1 |
| BENCH-03 | 3 |
| CPU-01 | 2 |
| CPU-02 | 2 |
| CPU-03 | 2 (evaluate; ship only if proven) |
| ACC-01 | 3 |
| ACC-02 | 3 |
| ACC-03 | 3 |
| MEM-01 | 4 |
| MEM-02 | 4 |
| MEM-03 | 4 |
| HW-01 | 5 |
| HW-02 | 5 |
| HW-03 | 5 |
| USE-01 | 6 |
| USE-02 | 4 |

Coverage: 17/17 requirements mapped ✓
