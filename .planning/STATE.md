# STATE: NumJa Performance & Scalability

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-25)

**Core value:** Xử lý dữ liệu lớn nhanh hơn và chính xác hơn mà API không thay đổi
**Current focus:** Phase 1 — Restore Dev Environment & Baseline Benchmark

## Current Position

- **Milestone:** v0.3.0 Performance & Scalability (first)
- **Phase:** 1 of 6 — **plan 01-01 ✅ done (commit 59e10f4)** — Wave 1 complete
- **Branch:** `dev` (tạo từ `revent-backup` — chứa đầy đủ source, pom.xml, JUnit tests)
- **Next command:** `/gsd-execute-phase` continues with Wave 2 (plans 01-02 + 01-03 in parallel)

## Key Context for Future Sessions

- Source KHÔNG có trên nhánh main (gitignore /modules/) — luôn làm việc trên `dev`
- Codebase map: `.planning/codebase/` (7 docs)
- Research summary: `.planning/research/SUMMARY.md` + **VERIFIED VERSIONS at `.planning/phases/01-baseline-benchmark/VERSIONS.md`** (committed 59e10f4)
- Build: Maven (pom.xml trên dev branch) + PowerShell scripts cho release; ProGuard obfuscation
- Tests tồn tại trên dev: 8 tests across NDArrayTest, LinAlgTest, NumJaJUnitTest, MatplotlibTest, DataFrameTest, SeabornTest, TestSklearnJUnit — all green as of 01-01
- API công khai bị đóng băng (v0.2.0) — mọi tối ưu là internal
- **Maven on PATH quirk:** Maven installed at `C:\Users\Admin\.maven\maven-3.9.15\bin` — NOT on default PATH. Prefix every `mvn` invocation with `$env:Path = 'C:\Users\Admin\.maven\maven-3.9.15\bin;' + $env:Path`
- **JDK on PATH:** Eclipse Adoptium 17.0.18 (default) + 25.0.3.9 (alternate). Vector API activation needs 21+ — Phase 2 work will need JDK bump decision

## Decisions Log

See PROJECT.md ## Key Decisions.

## Blockers

- None known. Phase 1 bắt đầu bằng việc switch sang nhánh `dev` nếu chưa.
