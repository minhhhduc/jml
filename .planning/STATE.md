# STATE: NumJa Performance & Scalability

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-25)

**Core value:** Xử lý dữ liệu lớn nhanh hơn và chính xác hơn mà API không thay đổi
**Current focus:** Phase 1 — Restore Dev Environment & Baseline Benchmark

## Current Position

- **Milestone:** v0.3.0 Performance & Scalability (first)
- **Phase:** 1 of 6 — **tất cả 3 plans executed** (01-01: `59e10f4`+`d5e1280`, 01-02: `96616a5`, 01-03: `c040ae7`) — chờ verify + close phase
- **Branch:** `dev`
- **Next command:** `/gsd-verify-work` (hoặc `/gsd-validate-phase`) để close Phase 1

## Key Context for Future Sessions

- Source KHÔNG có trên nhánh main (gitignore /modules/) — luôn làm việc trên `dev`
- Codebase map: `.planning/codebase/` (7 docs)
- Research summary: `.planning/research/SUMMARY.md` + **VERIFIED VERSIONS tại `.planning/phases/01-baseline-benchmark/VERSIONS.md`**
- Baseline numbers: `.planning/phases/01-baseline-benchmark/BASELINE.md` — matmul 1024²~1s/op, elementwise 10⁷ ~40ms, reduce ~11ms
- Accuracy gaps: sum/mean lệch NumPy 1e-14 (sequential vs pairwise) — fix bằng tree/Kahan ở Phase 3; matmul + softmax + linreg trong tolerance
- Build: Maven 3.9.15 tại `C:\Users\Admin\.maven\maven-3.9.15\bin` (**không trên PATH** — prefix `$env:Path = '...\bin;' + $env:Path`)
- Benchmarks: `mvn -pl bench -am package -DskipTests` rồi `java -jar bench/target/benchmarks.jar` từ repo root
- Golden tests: `mvn -pl modules/sklearn -am test "-Dtest=GoldenReferenceTest" "-Dsurefire.failIfNoSpecifiedTests=false"`
- Tests: full suite xanh (8+5 tests). GoldenReferenceTest dùng JUnit4 @Category (KHÔNG phải @Tag JUnit5)
- API công khai bị đóng băng (v0.2.0) — mọi tối ưu là internal

## Decisions Log

See PROJECT.md ## Key Decisions.

## Blockers

- None known. Phase 1 bắt đầu bằng việc switch sang nhánh `dev` nếu chưa.
