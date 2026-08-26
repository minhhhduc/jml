# BASELINE — NumJa Performance Baseline (Phase 1)

> Measured 2026-08-26 với JMH (`bench/` module, `-f 1 -wi 3 -i 3 -w 1s -r 1s`).
> Đây là **mốc tham chiếu** cho Phase 2–5. Mọi tối ưu so với các con số này.

## Machine metadata

| Item | Value |
|---|---|
| CPU | Intel Core i7-1255U (12th Gen, 10 cores / 12 logical) |
| RAM | 15.7 GB |
| OS | Windows 11 Home Single Language |
| JDK | OpenJDK Temurin 25.0.3+9 (mixed mode) |
| JVM flags | default |
| Maven | 3.9.15 |

## Kết quả

Mode `avgt` = AverageTime, đơn vị **ms/op**. Error = JMH 99.9% confidence interval.
Lưu ý: cột `(elemN)` / `(matmulN)` là @Param của class — mỗi benchmark chỉ dùng một param, param kia N/A về mặt ý nghĩa (JMH chạy full cartesian).

### Core ops

| Benchmark | Size | Score (ms/op) | Error | Unit |
|---|---|---|---|---|
| CoreBench.matmul_NxN | 256×256 | 11.8–15.7* | ±13–17 | ms/op |
| CoreBench.matmul_NxN | 1024×1024 | 868–1596* | ±5459–7352 | ms/op |
| CoreBench.add_elementwise | double[10⁶] | 2.88 | ±7.83 | ms/op |
| CoreBench.add_elementwise | double[10⁷] | 39.3–42.7 | ±64–104 | ms/op |
| CoreBench.multiply_elementwise | double[10⁶] | 3.00–4.24 | ±5.03–10.13 | ms/op |
| CoreBench.multiply_elementwise | double[10⁷] | 26.6–32.6 | ±48.2–54.3 | ms/op |
| CoreBench.sum_reduce | double[10⁷] | 11.28–12.05 | ±4.43–45.12 | ms/op |
| CoreBench.mean_reduce | double[10⁷] | 7.92–8.07** | ±1.38–6.81 | ms/op |

\* matmul có variance lớn giữa 2 lần đo cùng size — i7-1255U là CPU hybrid (P-core + E-core), thread scheduling không đều. Phase 2 cần re-run nhiều fork để ổn định số trước khi so sánh.
\*\* mean_reduce nhanh hơn sum_reduce vì mean chia sau khi sum nhưng JIT loop-unroll khác nhau — số này cần verify lại ở Phase 2.

### Pandas ops

| Benchmark | Input | Score (ms/op) | Error |
|---|---|---|---|
| PandasBench.read_csv_california | california_housing.csv (~20k rows) | 42.62 | ±49.29 |
| PandasBench.groupby_mean_titanic | titanic.csv (~891 rows), group by `sex` | 0.073 | ±0.417 |

### Sklearn ops

| Benchmark | Input | Score (ms/op) | Error |
|---|---|---|---|
| SklearnBench.linear_regression_fit | california_housing X[20k×8], y[20k] | 7.50 | ±18.39 |
| SklearnBench.kmeans_fit_iris | iris X[150×4], k=3 | 0.061 | ±0.050 |

## Nhận xét nhanh (input cho Phase 2+)

1. **matmul 1024×1024 rất chậm (~1s/op)** — EJML single-threaded path hiện tại không khai thác đa nhân. Target lớn nhất cho Phase 2/3.
2. **Elementwise 10⁷ ~40ms** ≈ 250 MB/s effective throughput — xa dưới băng thông RAM; song song hoá sẽ win lớn.
3. **Reduce ~11ms trên 10⁷ doubles (80 MB)** ≈ 7 GB/s — gần memory bandwidth ceiling của laptop; gain từ multithread sẽ ít hơn elementwise.
4. **read_csv 42ms / ~20k rows** chậm so với pandas Python (thường <10ms) — candidate cho streaming/chunked parser ở Phase 2.
5. **Error bars lớn** do hybrid CPU + Windows background noise. Khi so sánh Phase 2 vs baseline, dùng ≥5 forks và cùng power profile.

## Cách reproduce

```powershell
$env:Path = 'C:\Users\Admin\.maven\maven-3.9.15\bin;' + $env:Path
mvn -pl bench -am package -DskipTests
java -jar bench/target/benchmarks.jar -f 1 -wi 3 -i 3 -w 1s -r 1s
# Chạy 1 suite riêng:
java -jar bench/target/benchmarks.jar "CoreBench.matmul_NxN" -p matmulN=1024
```

Chạy từ repo root — `PandasBench`/`SklearnBench` load dataset bằng relative path `dist/datasets/*.csv`.
