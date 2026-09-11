# Phase 7 Discussion Log

**Date:** 2026-09-12
**Command:** /gsd-discuss-phase 6 (redirected per user choice)

## Background

User invoked `/gsd-discuss-phase 6`. Phase 6 in ROADMAP was "Usability Polish & Release", but earlier conversation established that if the Phase 5 TornadoVM POC fails on Colab T4, the next work item should be a JCuda/cuBLAS GPU fallback. User was asked which to discuss and chose inserting the new GPU fallback phase instead of discussing Phase 6.

## Decision 1 — Phase selection

- **Question:** Phase 6 trong ROADMAP hiện là "Usability Polish & Release". Muốn discuss phase nào?
- **Options:** (a) Usability Polish & Release [ROADMAP] (b) Chèn phase GPU fallback JCuda
- **Selection:** Chèn phase GPU fallback JCuda
- **Action:** Added Phase 7 "GPU Fallback POC via JCuda/cuBLAS" via `gsd-sdk query phase.add`, filled its goal/success criteria, added requirement GPU-04 (conditional) to REQUIREMENTS.md, coverage row, STATE.md note, committed `9ec47cd`.

## Decision 2 — Scope

- **Question:** "Chuyển hết phần tính toán bằng GPU sang JCuda" — scope Phase 7 tới đâu?
- **Options:** (a) POC trước, tích hợp sau (b) POC + tích hợp HAL luôn 1 phase
- **Selection:** POC trước, tích hợp sau
- **Note:** User's freeform answer "chuyen het phan tinh toan bang GPU sang JCuda" expresses the end goal; the POC-first staging keeps production modules dependency-free until gates pass.

## Decision 3 — Dependency acquisition

- **Question:** Cách lấy dependency JCuda cho POC?
- **Options:** (a) Notebook tải + install-file (b) Vendor vào dist/libs
- **Selection:** Notebook tải + install-file
- **Rationale:** Matches the TornadoVM SDK jar pattern already proven in the Colab notebook; no repo bloat, no Windows build pipeline changes.

## Decision 4 — Bench module + notebook shape

- **Question:** Hình dạng bench module + notebook cho POC JCuda?
- **Options:** (a) Clone tornado-poc (b) Bench mới hoàn toàn
- **Selection:** Clone tornado-poc
- **Rationale:** Same key=value format, seeds, N, CPU baseline → direct cross-comparison against Phase 5 TornadoVM numbers.

## Decision 5 — Failure policy

- **Question:** Nếu JCuda/cuBLAS cũng fail trên Colab T4 thì sao?
- **Options:** (a) Defer GPU, đóng phase (b) Thử backend thứ 3 luôn
- **Selection:** Defer GPU, đóng phase
- **Rationale:** Avoids an endless POC chain; documented NO-GO defers GPU to a future milestone.

## Claude's Discretion Items

- Exact JCuda artifact set (jcuda + jcublas + native jars) for the 12.6.0 release
- Warmup/timing methodology inside GemmBench (mirror tornado-poc)
- Notebook cell layout and env persistence mechanics

## Deferred Ideas

- `JcudaBackend implements ComputeBackend` + allowlist entry — future phase, gated on POC gates
- More GPU ops beyond matmul — after integration proves value
- Vendoring JCuda into dist/libs for offline users — only if integration ships
