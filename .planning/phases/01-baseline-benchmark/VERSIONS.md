# VERSIONS — Dependency & Runtime Research

> Verified 2026-08-26. Bổ sung cho `.planning/research/SUMMARY.md` (đã ghi "offline research"). Mỗi mục có nguồn URL, version cụ thể, và nhãn confidence (High/Medium/Low).

## Tóm tắt nhanh

| Công nghệ | Version hiện tại | Trạng thái | Confidence |
|---|---|---|---|
| TornadoVM | 5.2.0 (dual-track `-jdk21` / `-jdk25`) | Stable release Jul 2026 | High |
| Java Vector API | JDK 24 second preview (JEP 460); JDK 25 vẫn preview; JDK 26 mục tiêu GA | Preview/Incubator | High |
| EJML | 0.45.0 (May 2026) — `ejml-core` + `ejml-all` | Stable | High |

---

## 1. TornadoVM

**Version đã verify:** **5.2.0** (released 23 Jul 2026) — phát hành song song 2 biến thể:
- `v5.2.0-jdk21` (build cho OpenJDK 21)
- `v5.2.0-jdk25` (build cho OpenJDK 25)

**JDK tương thích:** JDK 21, JDK 25. Các bản dual-track cho phép target đúng JDK đang build mà không cần fork.

**GPU backend hỗ trợ:**
- NVIDIA CUDA (Linux + Windows, amd64): FP8 + FP8/BF16 tensor-core MMA, cp.async copies, cuBLAS/CUTLASS bf16 GEMM, fused epilogues (SiLU/Sigmoid/Tanh/HardSwish)
- AMD OpenCL (Linux amd64, macOS aarch64, Windows amd64)
- Apple Silicon Metal (macOS aarch64)
- Intel Level Zero / SPIR-V

**Ảnh hưởng tới NumJa:**
- Phase 5 (GPU POC) — chọn `-jdk21` để khớp target compile hiện tại. Trên máy dev Windows + CUDA: cần `tornadovm-5.2.0-jdk21-cuda-windows-amd64.tar.gz`.
- Compile `modules/numja` không cần TornadoVM runtime — chỉ Phase 5 thêm dependency.
- **Rủi ro thực tế:** setup yêu cầu driver NVIDIA/AMD riêng, không phải `mvn install` đơn thuần. POC đầu tiên nên dùng CUDA (Windows có sẵn dev kit hơn Metal/Level Zero).

**Nguồn:**
- https://github.com/beehive-lab/TornadoVM/releases
- https://github.com/beehive-lab/TornadoVM

**Confidence:** **High** — confirmed live từ GitHub releases page 2026-08-26.

---

## 2. Java Vector API (`jdk.incubator.vector` → `java.base`)

**Trạng thái trên các JDK gần đây:**
- JDK 16 (Mar 2021): incubator module đầu tiên
- JDK 17–20: tiếp tục incubator, refine API
- JDK 21 (Sep 2023): fifth incubator
- JDK 22 (Mar 2024): sixth incubator
- JDK 23 (Sep 2024): seventh incubator
- **JDK 24 (Mar 2025): second preview trong `java.base` — JEP 460** (vẫn cần `--enable-preview`)
- **JDK 25 (Sep 2025, hiện tại): vẫn preview** (target cuối cùng là final ~JDK 26)
- JDK 26 (Mar 2026): mục tiêu GA nếu feedback preview đủ tốt

**Ảnh hưởng tới NumJa:**
- Hiện `modules/*/pom.xml` compile `--release 11` — Vector API không có sẵn ở JDK 11. Để dùng Phase 2 (CPU SIMD), phải:
  1. **Nâng `--release` lên 21** cho module `numja` (SIMD chỉ dùng trong core) — hoặc
  2. Dùng multi-release JAR (mr-JAR với entries cho JDK 21+).
- Lựa chọn đơn giản hơn: nâng `--release` lên 21 cho module `numja` (JDK 21 là JDK LTS phổ biến). Phải giữ `--release 11` cho các module public-facing nếu downstream cần JDK 11 compat — cần check trước Phase 2.
- Runtime detect: `Runtime.version().feature() >= 21` → dùng Vector API, fallback thuần Java nếu không. JIT sẽ tối ưu fallback gần bằng scalar SIMD code đã viết tay trong nhiều trường hợp.

**Rủi ro thực tế:** API surface vẫn thay đổi giữa các preview — nếu build trên JDK 25, code có thể không compile trên JDK 21. **Giải pháp:** pin target compile = JDK 21 (LTS, ổn định đến 2026+), tránh dùng API preview mới nhất của JDK 25.

**Nguồn:**
- JEP 460 (Vector API, JDK 24): https://openjdk.org/jeps/460
- Project Panama / Vector API: https://wiki.openjdk.org/display/Panama/Vector+API+Home

**Confidence:** **High** — JEP timeline là public record trên openjdk.org.

---

## 3. EJML (Efficient Java Matrix Library)

**Version đã verify:** **0.45.0** (released 15 May 2026).

**Artifact Maven:**
- `org.ejml:ejml-core` — core dense matrix operations (nhẹ, chỉ dense)
- `org.ejml:ejml-all` — bundle all-in-one (core + sparse + extra + simple interface)
- `org.ejml:ejml-simple` — SimpleMatrix wrapper (optional)
- `org.ejml:ejml-fdense` — fixed-size dense matrix utilities (optional)

**Multithreaded variant:**
- Research gốc ghi "EJML có bản multithreaded mt-0.43.x" — **không confirm được** artifact `ejml-mt-*` từ official docs/Maven Central searches trong phiên này.
- EJML core API hiện đã tích hợp concurrency nội bộ cho một số op (theo news 2026 "More concurrency") — không cần artifact phụ.
- **Khuyến nghị Phase 2/3:** import `ejml-all` 0.45.0, dùng `DMatrixRMaj` + `CommonOps` + `MatrixMult`. Nếu cần threading chi tiết, dùng `ConcurrencyUtils` / ForkJoinPool thủ công (giống pattern đã có trong `numja.linalg.ParallelUtils`).

**Ảnh hưởng tới NumJa:**
- `numja.linalg.LinAlg` hiện viết tay matmul/lu/inverse. Phase 3 (math hardening) có thể thay một số op bằng EJML để đo sai số và cross-check.
- `ejml-all` 0.45.0 tương thích JDK 11+ — không cần nâng `--release`.

**Rủi ro thực tế:** Dependency size — `ejml-all` ~700 KB JAR. Chấp nhận được; module `numja` không nằm trong ProGuard distribution chính (chỉ `sklearn`/`pandas` jar cuối cùng mới obfuscate, theo PROJECT.md).

**Nguồn:**
- http://ejml.org/wiki/index.php/Main_Page (v0.45.0 listed 2026-05-15)
- Maven Central: `org.ejml:ejml-all` (search "g:org.ejml")

**Confidence:** **High** cho version 0.45.0 + artifact `ejml-all`. **Medium** cho "không có artifact `ejml-mt`" — không tìm được official doc khẳng định explicit, chỉ có absence of evidence từ wiki + central search.

---

## Gaps còn lại

- [ ] Xác nhận CUDA toolkit version mà TornadoVM 5.2.0 yêu cầu (thường CUDA 12.x) — Phase 5 mới cần.
- [ ] Đo thực tế Vector API overhead vs scalar loop trên mảng nhỏ trên máy dev (Phase 2).
- [ ] EJML thread-safety trên `DMatrixRMaj` — Phase 3 verify.
