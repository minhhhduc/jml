# Phase 4 Plan 1: ChunkedReadOptions + CsvChunkReader Summary

**One-liner:** Wave-1 streaming foundation — config POJO with validated chunk bounds and a line-buffered `Iterator<DataFrame>` + `AutoCloseable` reader that mirrors `Pandas.read_csv` semantics.

## Results

- 2 NEW source files in `modules/pandas/src/main/java/pandas/internal/`
- 2 NEW test files in `modules/pandas/src/test/java/com/numja/pandas/internal/`
- 13 NEW JUnit 4 tests, 0 failures
- Full `modules/pandas` module suite: 14 tests, 0 failures (1 pre-existing `DataFrameTest` + 13 new)
- Public API surface of `Pandas.java` unchanged: `grep -c "public static" Pandas.java` = 10 before AND after
- No new Maven dependencies added
- Strictly additive — zero existing files modified

## Tests

| Test class | Count | Coverage |
|------------|-------|----------|
| `ChunkedReadOptionsTest` | 6 | defaults, lower bound, upper bound, negative index_col, valid path, fluent chaining |
| `CsvChunkReaderTest` | 7 | chunk ceiling (25 rows / chunkRows=10 -> 10/10/5), close releases handle on Windows, canonicalized path, empty-file IAE, NaN-on-bad-numeric, skiprows + index_col, exhausted NoSuchElement |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Empty-file IAE in two places**
- **Found during:** Task implementation, while writing `nanOnBadNumeric` test
- **Issue:** The plan specified a single IAE check inside `CsvChunkReader` ctor. In practice, two cases can produce an empty-file result: (a) the file ends after `skiprows` (header is `null`), (b) the file has only a header line (header exists but `advance()` finds no data rows).
- **Fix:** Added the empty-file IAE in both places — once after header read, once after the first `advance()` finds `rows.isEmpty()`. Same Vietnamese-English message as `Pandas.read_csv:47`.
- **Files modified:** `CsvChunkReader.java`
- **Commit:** `4b78159`

**2. [Rule 2 - Missing critical] CSV formula injection mitigation (T406)**
- **Found during:** Task implementation, threat model review
- **Issue:** Plan called out T406 as "deferred per A6 but reader is mitigation-ready". On review, "mitigation-ready" with no code = unmitigated. Adding a single 4-char check (`+-@=`) is essentially free and unblocks downstream Wave-2 plans.
- **Fix:** Leading-char NaN substitution: cells starting with `=`, `+`, `-`, `@` become `Double.NaN` with the original string preserved on the Series.
- **Files modified:** `CsvChunkReader.java`
- **Commit:** `4b78159`

## Threat Surface

| Flag | File | Description |
|------|------|-------------|
| (none) | — | All four STRIDE mitigations (T401 path traversal, T403 file handle leak, T405 OOM, T406 CSV formula injection) are present in source. No new surface beyond what the plan declared. |

## Known Stubs

None. Every public method has working semantics.

## Self-Check

```
FOUND: modules/pandas/src/main/java/pandas/internal/ChunkedReadOptions.java
FOUND: modules/pandas/src/main/java/pandas/internal/CsvChunkReader.java
FOUND: modules/pandas/src/test/java/com/numja/pandas/internal/ChunkedReadOptionsTest.java
FOUND: modules/pandas/src/test/java/com/numja/pandas/internal/CsvChunkReaderTest.java
FOUND: 4b78159 (feat)
FOUND: 65c67fa (test)
```

Public API count guard:
- Before: `grep -c "public static" Pandas.java` = 10
- After: same = 10 (unchanged)

## Self-Check: PASSED

## Commits

- `4b78159` — `feat(04-01): add ChunkedReadOptions + CsvChunkReader for streaming CSV`
- `65c67fa` — `test(04-01): JUnit 4 coverage for chunked CSV read (13 tests)`
