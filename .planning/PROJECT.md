# NumJa Performance & Scalability

## What This Is

Nâng cấp NumJa (thư viện Java ML mô phỏng NumPy/Pandas/Sklearn, đóng gói 5 JAR obfuscated) để xử lý dữ liệu lớn trên cả bốn trục: **tốc độ**, **tăng tốc phần cứng** (thread/GPU/TPU), **độ chính xác**, và **dễ thao tác**. Người dùng cuối là lập trình viên Java cần phân tích dữ liệu offline, không phụ thuộc Python.

## Core Value

Xử lý tập dữ liệu lớn nhanh hơn và chính xác hơn mà **API hiện tại không thay đổi** — code người dùng cũ chạy lại không sửa một dòng.

## Requirements

### Validated

<!-- Sẵn có trong codebase, được giữ nguyên làm nền -->

- ✓ NDArray container + linear algebra (EJML backend): matmul, SVD, QR, Eigen, Lstsq — existing
- ✓ Pandas layer: read_csv/read_json, DataFrame/Series, loc/iloc, stats, groupby — existing
- ✓ Sklearn suite ~50 classes: 9 classifiers, 8 regressors, preprocessing, model selection, clustering, PCA, MLP — existing
- ✓ Matplotlib/Seaborn: line/scatter plots, imshow, heatmap — existing
- ✓ Multi-threading nền: ThreadPoolConfig + ParallelUtils (~60% cores) — existing
- ✓ Maven build + JUnit tests trên nhánh dev (NDArrayTest, LinAlgTest, NumJaTest, TestSklearn) — existing on `dev`
- ✓ ProGuard obfuscation pipeline → dist/*.jar — existing

### Active

<!-- Milestone này: hypotheses đến khi shipped -->

- [ ] PERF-01: Benchmark harness đo baseline throughput/latency cho core ops (matmul, elementwise, groupby, fit/predict) — không đo thì không biết tối ưu cái gì
- [ ] PERF-02: Adaptive memory model: dataset nhỏ (< RAM threshold) in-memory như cũ; dataset lớn tự động chunk/out-of-core processing, API gọi bên ngoài không đổi
- [ ] PERF-03: Tối ưu CPU đa luồng: mở rộng ParallelUtils cho elementwise ops, parallel sort/reduce; đánh giá Vector API (SIMD) cho hot loops
- [ ] PERF-04: Hardware abstraction layer (ComputeBackend): interface backend để op chạy trên CPU thread / GPU / TPU tùy khả năng máy — GPU/TPU backend cài sau qua TornadoVM hoặc tương đương
- [ ] PERF-05: Độ chính xác: kiểm tra numerical stability (Kahan summation cho reduce lớn, stable softmax/log), so sánh kết quả với NumPy/scipy trên benchmark datasets
- [ ] PERF-06: Usability: cải thiện error messages (shape mismatch, dtype), thêm builder/factory helpers cho pipeline dữ liệu lớn — không breaking change
- [ ] PERF-07: Regression gate: benchmark CI-style check chống suy giảm hiệu năng + test accuracy vs golden values

### Out of Scope

- Distributed computing đa máy (Spark-like cluster) — quy mô hiện tại là 1 máy; đánh giá lại nếu dữ liệu vượt 50GB+
- Đổi API công khai / breaking change — Core Value cấm; fluent/lazy style chỉ cân nhắc milestone sau
- Viết lại bằng native code (C++/JNI thuần) — mất tính "zero-install thuần JVM"; JNI chỉ qua thư viện tăng tốc có sẵn nếu thực sự cần
- UI/dashboard — đây là library
- Thêm model ML mới — milestone này chỉ tối ưu những gì đã có

## Context

- **Source nằm ở nhánh `dev`** (tạo từ `revent-backup`): đủ `modules/*/src`, pom.xml, proguard.pro, JUnit tests. Nhánh main thiếu source vì `.gitignore` chặn `/modules/`. Mọi work xảy ra trên `dev`.
- Codebase map đầy đủ tại `.planning/codebase/` (STACK, ARCHITECTURE, CONCERNS...).
- Hiện trạng hiệu năng chưa được đo bao giờ — không có benchmark. ParallelUtils mới dùng cho một số model (RandomForest, GridSearchCV); NDArray elementwise ops có thể vẫn single-thread.
- EJML 0.43.1 là backend dense matrix — đã tối ưu sẵn ở tầng thấp; dư địa lớn nhất nằm ở tầng pandas/sklearn và ops tự viết.
- Docs claim "Java 11–25+"; build pin `--release 11`. Vector API (JDK16+, incubator→final JDK21+) cần quyết định target JDK.
- Chưa xác minh GPU/TPU backend nào phù hợp nhất (TornadoVM là ứng viên mặc định vì thuần JVM) — research trước khi code.

## Constraints

- **Compatibility**: Giữ nguyên public API v0.2.0 — người dùng đang deploy offline với jars trong dist/
- **Tech stack**: Thuần JVM, không bắt buộc native lib mới; mọi dependency mới phải vendored vào dist/libs/
- **Runtime**: Target Java 11 trở lên; tính năng JDK cao hơn (Vector API) phải optional-fallback
- **Closed-source**: modules/ vẫn git-ignored theo mô hình phát hành; build vẫn qua PowerShell scripts + ProGuard
- **Chính xác**: Kết quả số phải khớp golden values (NumPy reference) trong tolerance — tốc độ không được đánh đổi bằng sai số

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Làm việc trên nhánh `dev` từ revent-backup | Main thiếu source (gitignore /modules/) | — Pending |
| Giữ nguyên API, tối ưu ngầm | Người dùng đang dùng jars offline, không muốn breaking change | — Pending |
| Adaptive memory (chunking) thay vì yêu cầu user quản lý | "Dễ dàng thao tác" = library tự thích nghi kích thước dữ liệu | — Pending |
| ComputeBackend abstraction trước, GPU cụ thể sau | Tránh khóa sớm vào TornadoVM/CUDA; CPU threads là backend đầu tiên | — Pending |
| Benchmark trước, optimize sau (PERF-01 đứng đầu) | Không có số liệu baseline thì mọi tối ưu là phỏng đoán | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-25 after initialization*
