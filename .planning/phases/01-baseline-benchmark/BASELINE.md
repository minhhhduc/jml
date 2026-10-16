# BASELINE — NumJa Performance Baseline (Phase 1)

> Measured 2026-08-26 (baseline v2) với JMH (`bench/` module, `-f 1 -wi 3 -i 3 -w 1s -r 1s`).
> Đây là **mốc tham chiếu** cho Phase 2–5. Mọi tối ưu so với các con số này.
>
> v2 thay baseline đầu tiên: CoreBench được sửa sau code review (MJ-02 — tách
> state classes, hết cartesian @Param; mỗi benchmark chỉ chạy đúng param của nó),
> và `groupby_mean_titanic` chạy được (cột `sex` lowercase).

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
Cột `(n)` là @Param của state class tương ứng.

### Core ops

| Benchmark | Size | Score (ms/op) | Error |
|---|---|---|---|
| CoreBench.matmul_NxN | 256×256 | 29.27 | ±56.01 |
| CoreBench.matmul_NxN | 1024×1024 | 6846.8 | ±27430.7 |
| CoreBench.add_elementwise | double[10⁶] | 4.93 | ±4.91 |
| CoreBench.add_elementwise | double[10⁷] | 89.38 | ±507.8 |
| CoreBench.multiply_elementwise | double[10⁶] | 7.22 | ±5.87 |
| CoreBench.multiply_elementwise | double[10⁷] | 98.68 | ±497.0 |
| CoreBench.sum_reduce | double[10⁷] | 27.21 | ±176.7 |
| CoreBench.mean_reduce | double[10⁷] | 42.73 | ±102.8 |

### Pandas ops

| Benchmark | Input | Score (ms/op) | Error |
|---|---|---|---|
| PandasBench.read_csv_california | california_housing.csv (~20k rows) | 105.5 | ±164.8 |
| PandasBench.groupby_mean_titanic | titanic.csv (~891 rows), group by `sex` | 0.226 | ±0.681 |

### Sklearn ops

| Benchmark | Input | Score (ms/op) | Error |
|---|---|---|---|
| SklearnBench.linear_regression_fit | california_housing X[20k×8], y[20k] | 20.67 | ±63.55 |
| SklearnBench.kmeans_fit_iris | iris X[150×4], k=3 | 0.283 | ±1.88 |

## Nhận xét nhanh (input cho Phase 2+)

1. **matmul 1024×1024 rất chậm (~7s/op trong run này)** — EJML single-threaded path hiện tại không khai thác đa nhân. Target lớn nhất cho Phase 2/3.
2. **Elementwise 10⁷ ~90–100ms** ≈ ~100 MB/s effective throughput — xa dưới băng thông RAM; song song hoá sẽ win lớn.
3. **Reduce ~30–40ms trên 10⁷ doubles (80 MB)** ≈ 2–3 GB/s — dưới memory bandwidth ceiling; multithread reduce có dư địa.
4. **read_csv ~105ms / ~20k rows** chậm so với pandas Python (<10ms) — candidate cho streaming/chunked parser ở Phase 2.
5. **Error bars rất lớn** (đôi khi > score) do hybrid CPU (P/E-core) + Windows background noise + chỉ 3 iterations rút gọn. Baseline v1 cùng hiện tượng. Khi so sánh Phase 2 vs baseline: dùng ≥5 forks, cùng power profile, và re-measure cả hai phía cùng lúc nếu có thể.

## Cách reproduce

```powershell
$env:Path = 'C:\Users\Admin\.maven\maven-3.9.15\bin;' + $env:Path
mvn -pl bench -am package -DskipTests
java -jar bench/target/benchmarks.jar -f 1 -wi 3 -i 3 -w 1s -r 1s
# Chạy 1 suite riêng:
java -jar bench/target/benchmarks.jar "CoreBench.matmul_NxN" -p n=1024
```

Chạy từ repo root — `PandasBench`/`SklearnBench` load dataset bằng relative path `dist/datasets/*.csv`.
