# Golden Values — Regeneration Guide

Golden reference values cho accuracy tests được sinh từ NumPy (source of truth) và commit vào `bench/src/test/resources/golden/`.

## Khi nào cần regenerate

- Upgrade NumPy/Python trong `.venv` (kết quả float có thể đổi ở ulp cuối)
- Thêm op mới vào test coverage
- Không cần regenerate khi chỉ sửa Java code

## Cách regenerate

```powershell
.venv\Scripts\python.exe scripts\golden\generate_golden.py
```

Script dùng seed cố định (`SEED=42`, các dataset dùng 42+n) nên output deterministic trên cùng một NumPy version.

## Format JSON

```json
{
  "op": "tên op (matmul | sum_mean | softmax | linear_regression)",
  "seed": 42,
  "inputs": { ... input arrays ... },
  "expected": [ ... ] hoặc { "sum": ..., "mean": ... },
  "tolerance_rel": 1e-12
}
```

`tolerance_rel` = sai số relative tối đa chấp nhận được khi so kết quả NumJa với `expected`.

## Tolerance per op và lý do

| Op | tolerance_rel | Lý do |
|---|---|---|
| matmul 256×256 | 1e-12 | EJML blocked matmul vs NumPy BLAS — khác thứ tự cộng tích lũy |
| sum/mean double[10⁶] | 1e-15 | Sum tuần tự vs pairwise summation của NumPy — drift nhỏ |
| softmax vector[1000] ±700 | 1e-12 | Cả hai phải max-shift trước exp; sai số tích lũy exp |
| LinearRegression iris | 1e-9 | Normal equation + Gaussian elimination vs sklearn lstsq solver |

## Test tiêu thụ

`modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java` — JUnit, tag `@Tag("golden")`, soft-fail mode: assertion fail không làm crash suite, chỉ ghi nhận gap vào report. Chạy riêng bằng:

```powershell
mvn test -Dgroups=golden
```
