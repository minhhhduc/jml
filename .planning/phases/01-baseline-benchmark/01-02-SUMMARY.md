# 01-02 SUMMARY: JMH Benchmark Harness + Baseline

**Plan:** 01-02 (Wave 2)
**Type:** execute
**Branch:** `dev`
**Completed:** 2026-08-26
**Commits:** `96616a5`

## What shipped

### Task 1 — Maven module `bench/` với JMH

- `bench/pom.xml`: parent = root pom, deps = `jmh-core` 1.37 + `jmh-generator-annprocess` (provided) + `numja-core` / `pandas-wrapper` / `sklearn`
- Shade plugin → standalone `bench/target/benchmarks.jar` (mainClass `org.openjdk.jmh.Main`)
- Root pom: thêm `<module>bench</module>`
- Verify: `mvn -pl bench -am compile` exit 0 ✅

### Task 2 — Benchmark classes

- **CoreBench**: matmul N∈{256,1024}; elementwise add/mul trên double[10⁶] và [10⁷]; reduce sum/mean trên double[10⁷]. Mọi method consume qua Blackhole.
- **PandasBench**: `read_csv california_housing.csv`; `groupby("sex").mean()` trên titanic.
- **SklearnBench**: `LinearRegression.fit` california_housing; `KMeans(3).fit` iris.
- Setup pre-allocate data với seed=42, chỉ đo op.
- Verify: package exit 0, smoke test `matmul256 -f 0 -wi 1 -i 1` in ra score ✅

### Task 3 — Full baseline

- Chạy full suite `-f 1 -wi 3 -i 3`. Kết quả trong [BASELINE.md](BASELINE.md) (12 benchmarks).
- Machine metadata: i7-1255U, 15.7GB RAM, Windows 11, Temurin JDK 25.0.3.

## Deviations

| Plan said | Did | Why |
|---|---|---|
| groupby cột "category" titanic | dùng cột `sex` | titanic.csv thực tế không có cột category; `sex` là cột string tự nhiên. Fix đầu tiên (`"Sex"`) fail vì header là lowercase — đã sửa và re-run riêng benchmark đó. |
| Main class runner per-suite regex | dùng JMH CLI regex args trực tiếp (`java -jar benchmarks.jar "regex"`) | JMH Main đã có sẵn cơ chế này — viết wrapper là trùng lặp. |

## Key baseline findings (chi tiết trong BASELINE.md)

1. matmul 1024×1024 ~0.9–1.6s/op — EJML single-thread, target lớn nhất Phase 2/3
2. elementwise add 10⁷ ~40ms (~250 MB/s) — song song hoá sẽ win lớn
3. reduce ~11ms/80MB (~7 GB/s) — gần memory bandwidth ceiling, gain ít hơn
4. read_csv ~43ms/20k rows — chậm hơn pandas Python, candidate streaming parser
5. Error bars lớn do hybrid CPU (P/E-core) — khi so sánh cần ≥5 forks

## Files modified

- NEW: `bench/pom.xml`, `bench/src/main/java/bench/{CoreBench,PandasBench,SklearnBench}.java`, `.planning/phases/01-baseline-benchmark/BASELINE.md`
- MOD: `pom.xml` (thêm module bench)
