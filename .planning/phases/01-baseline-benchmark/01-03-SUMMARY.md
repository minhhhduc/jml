# 01-03 SUMMARY: Golden-Value Accuracy Harness

**Plan:** 01-03 (Wave 2, TDD)
**Type:** tdd
**Branch:** `dev`
**Completed:** 2026-08-26
**Commits:** `c040ae7`

## What shipped

### Task 1 — Golden values từ NumPy reference

- `scripts/golden/generate_golden.py` — dùng `.venv` (numpy 2.4.5, sklearn 1.8.0), seed=42 (+n per dataset)
- 4 JSON files trong `bench/src/test/resources/golden/`, mỗi file có `op`, `seed`, `inputs`, `expected`, `tolerance_rel`:
  - `matmul_256.json` (tol 1e-12)
  - `sum_mean_1e6.json` (tol 1e-15)
  - `softmax_1000_extreme.json` — vector[1000] giá trị biên ±700 (tol 1e-12)
  - `linear_regression_iris.json` (tol 1e-9)
- `bench/GOLDEN.md` — quy trình regenerate + rationale từng tolerance
- Verify: script chạy ra 4 file, đều chứa `tolerance_rel` ✅

### Task 2 — GoldenReferenceTest

- `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java`
- **Soft-fail mode đúng spec**: assertion gaps ghi vào report + stderr, suite vẫn exit 0. Phase 3 mới fix thuật toán.
- Verify: exit 0, report liệt kê pass/fail per op ✅; full suite `mvn clean test` vẫn xanh ✅

## Baseline sai số đo được (mục tiêu chính của plan)

| Op | NumJa err vs NumPy | Tol | Verdict |
|---|---|---|---|
| matmul 256×256 | 1.866e-15 | 1e-12 | ✅ PASS |
| softmax[1000] ±700 | 2.211e-16 | 1e-12 | ✅ PASS |
| sum double[10⁶] | 1.088e-14 | 1e-15 | ❌ GAP — sequential summation drift |
| mean double[10⁶] | 1.083e-14 | 1e-15 | ❌ GAP — same root cause |
| LinearRegression iris | ≤1e-9 mọi coef/intercept/preds | 1e-9 | ✅ PASS |

**Gap duy nhất: reduce ops.** Nguyên nhân đã biết từ research: NumJa sum tuần tự trái-qua-phải, NumPy dùng pairwise summation. Fix path Phase 3: tree-reduce hoặc Kahan summation (research/SUMMARY.md mục "Kahan").

## Deviations

| Plan said | Did | Why |
|---|---|---|
| `@Tag("golden")` (JUnit 5) | JUnit 4 `@Category(GoldenReferenceTest.Golden.class)` + marker interface | Project dùng JUnit 4.13.2 — @Tag không tồn tại. Surefire `-Dgroups` filter với JUnit 4 categories không hoạt động như kỳ vọng trong lần chạy đầu (0 tests); chạy bằng `-Dtest=GoldenReferenceTest`. Cần cấu hình surefire `<groups>` nếu muốn filter theo group thật — để lại note cho Phase 3. |
| Soft-fail qua "test pass về cơ chế" | mỗi test kết thúc `assertTrue(true)` sau khi log gap | Đúng ý đồ plan — mechanical pass, gap ghi nhận. |

## Files modified

- NEW: `scripts/golden/generate_golden.py`, 4 golden JSONs, `bench/GOLDEN.md`, `GoldenReferenceTest.java`
- MOD: `modules/sklearn/pom.xml` (thêm `org.json:json:20240303`, test scope)

## Lưu ý kích thước file

Golden JSON chứa full input arrays (matmul 256×256×2 inputs ≈ 131k số) → commit ~1.2M dòng JSON. Chấp nhận được vì cần reproduce input cùng seed; nếu repo phình, có thể chuyển sang lưu chỉ seed + regenerate lúc build.
