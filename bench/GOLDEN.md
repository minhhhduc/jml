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
  "input_gen": "mô tả cách sinh inputs từ seed bằng java.util.Random",
  "expected": [ ... ] hoặc { "sum": ..., "mean": ... },
  "tolerance_rel": 1e-12
}
```

**Inputs KHÔNG được lưu trong JSON** — chúng được sinh deterministic từ `seed`
qua `java.util.Random` ở cả hai phía: Python generator reimplement LCG của Java
(bit-identical), và Java test dùng `new Random(seed)` thật. Điều này giữ file
golden nhỏ (~1MB thay vì 33MB raw doubles). **Lưu ý draw order**: thứ tự gọi
`nextDouble()` phải khớp chính xác giữa generator và test (vd matmul draw toàn
bộ ma trận `a` trước, rồi mới đến `b`).

`tolerance_rel` = sai số relative tối đa chấp nhận được khi so kết quả NumJa với `expected`.

## Tolerance per op và lý do

| Op | tolerance_rel | Lý do |
|---|---|---|
| matmul 256×256 | 1e-12 | EJML blocked matmul vs NumPy BLAS — khác thứ tự cộng tích lũy (đo được: maxErr ~2e-15) |
| sum/mean double[10⁶] | 1e-13 | Sum tuần tự vs pairwise summation của NumPy — drift thực tế ~1e-14 tại n=10⁶, tolerance phải nằm trên drift này |
| softmax vector[1000] ±700 | 1e-12 | Cả hai phải max-shift trước exp; sai số tích lũy exp (đo được: ~4e-16) |
| LinearRegression iris | 1e-9 | Normal equation + Gaussian elimination vs sklearn lstsq solver |

## Test tiêu thụ

`modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java` —
JUnit 4 với `@Category(Golden.class)` marker interface (JUnit 4 không dùng
`@Tag` của JUnit 5). Soft-fail mode: assertion fail không làm crash suite —
mỗi check ghi PASS/FAIL vào stdout (`[GOLDEN] ...`) và flush ra
`target/golden-report.txt`. Chạy bằng:

```powershell
mvn -pl modules/sklearn -am test "-Dtest=GoldenReferenceTest" "-Dsurefire.failIfNoSpecifiedTests=false"
```

(Lưu ý: `-Dgroups=golden` KHÔNG có tác dụng với JUnit 4 categories ở surefire
config hiện tại — dùng `-Dtest=` như trên.)
