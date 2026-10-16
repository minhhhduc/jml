# Phase 4: Adaptive Memory Model - Pattern Map

**Mapped:** 2026-08-28
**Files analyzed:** 11 (8 NEW + 2 additive + 1 test infra)
**Analogs found:** 6 / 6 (100% coverage)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `modules/pandas/src/main/java/pandas/internal/ChunkedReadOptions.java` | config (POJO) | n/a (value carrier) | `modules/pandas/src/main/java/pandas/Pandas.java:16-28` (`ReadCsvOptions` inner class) | exact |
| `modules/pandas/src/main/java/pandas/internal/CsvChunkReader.java` | iterator (file I/O) | streaming + file-I/O | `modules/pandas/src/main/java/pandas/Pandas.java:34-112` (`read_csv(String, ReadCsvOptions)`) | exact |
| `modules/pandas/src/main/java/pandas/internal/RunningGroupAggregator.java` | utility (aggregation) | streaming + merge | `modules/pandas/src/main/java/pandas/GroupBy.java:14-24, 71-85` | exact |
| `modules/pandas/src/main/java/pandas/Pandas.java` (+1 method) | controller (additive) | file-I/O | `modules/pandas/src/main/java/pandas/Pandas.java:30-112` | exact (sibling) |
| `modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java` (+1 method) | estimator (additive) | streaming + per-class merge | `modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java:18-47` | exact (sibling) |
| `modules/sklearn/src/main/java/sklearn/pipeline/PandasPipeline.java` | builder | streaming + chained I/O | `modules/sklearn/src/main/java/sklearn/pipeline/Pipeline.java:1-85` | role-match (different style: builder vs reflective) |
| `modules/pandas/src/test/java/com/numja/pandas/internal/CsvChunkReaderTest.java` | test | streaming | `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java:27-32` (`@After` hygiene) | exact (hygiene pattern) |
| `modules/pandas/src/test/java/com/numja/pandas/internal/RunningGroupAggregatorTest.java` | test | merge | `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java:27-32` (`@After` hygiene) | exact (hygiene pattern) |
| `modules/pandas/src/test/java/com/numja/pandas/PandasStreamingTest.java` | test (cross-path) | streaming | existing `modules/pandas/src/test/java/com/numja/pandas/DataFrameTest.java` (JUnit 4 test class style) | exact (test style) |
| `modules/sklearn/src/test/java/sklearn/naive_bayes/GaussianNBPartialFitTest.java` | test (cross-path) | streaming | existing `modules/sklearn/src/test/java/sklearn/TestSklearnJUnit.java` (JUnit 4 style) | exact (test style) |
| `modules/sklearn/src/test/java/sklearn/pipeline/PandasPipelineTest.java` | test (integration) | streaming | existing `modules/sklearn/src/test/java/sklearn/TestSklearnJUnit.java` (JUnit 4 style) | exact (test style) |

**Note on test directory:** Existing pandas tests live at `modules/pandas/src/test/java/com/numja/pandas/` (mirrors numja-wrapper style). Wave 0 specifies `com/pandas/internal/` — the planner should use `com/numja/pandas/internal/` to match the existing module convention (one fewer top-level package break; consistent with DataFrameTest.java location).

---

## Pattern Assignments

### `modules/pandas/src/main/java/pandas/internal/ChunkedReadOptions.java` (config POJO)

**Analog:** `Pandas.ReadCsvOptions` (inner class at Pandas.java:16-28)

**Imports pattern** — mirror Pandas.java:1-6 (only stdlib):
```java
package pandas.internal;

import java.nio.file.Path;

public class ChunkedReadOptions {
    public String sep = ",";
    public int skiprows = 0;
    public Integer index_col = null;
    public int chunkRows;

    public ChunkedReadOptions chunkRows(int n) { this.chunkRows = n; return this; }
}
```

**Config POJO pattern** (Pandas.java:16-28):
```java
// Source: modules/pandas/src/main/java/pandas/Pandas.java:16-28
public static class ReadCsvOptions {
    public String sep = ",";
    public Integer header = 0;
    public String[] names = null;
    public int skiprows = 0;
    public Integer index_col = null;

    public ReadCsvOptions sep(String sep) { this.sep = sep; return this; }
    // ... other fluent setters return `this`
}
```

**Key conventions:**
- Public fields with sensible defaults (NOT final, NOT getters/setters)
- Fluent setters return `this` for builder-like chaining
- Top-level class with its own file (NOT a static inner class) since it's in a different package

**WR-05 carry-forward (defensive identity init):** Set `chunkRows = 0` default, then require non-zero at use-site — explicit ctor init not needed for an int but flag if someone adds `double budgetFraction` field (init to 0.5, NOT 0.0).

**Validation at ctor:**
```java
public ChunkedReadOptions() {
    // Validate post-construction in CsvChunkReader.open() so error has path context
}
```

**ASVS V5 + STRIDE block (mandated):** Include in javadoc — trust boundary is filesystem, threats = path traversal + OOM + CSV injection, mitigations = `path.normalize()` + `chunkRows` cap + leading-char scrub.

---

### `modules/pandas/src/main/java/pandas/internal/CsvChunkReader.java` (Iterator<DataFrame>, AutoCloseable)

**Analog:** `Pandas.read_csv(String, ReadCsvOptions)` body (Pandas.java:34-112)

**Imports pattern** (mirror Pandas.java:3-7 + add NIO + AutoCloseable):
```java
package pandas.internal;

import pandas.DataFrame;
import pandas.Series;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public final class CsvChunkReader implements Iterator<DataFrame>, AutoCloseable {
```

**Core BufferedReader + readLine loop** (Pandas.java:37-44):
```java
// Source: modules/pandas/src/main/java/pandas/Pandas.java:37-44
try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
    String line;
    int currentRow = 0;
    while ((line = br.readLine()) != null) {
        if (currentRow < options.skiprows) { currentRow++; continue; }
        if (!line.trim().isEmpty()) lines.add(line.split(options.sep));
        currentRow++;
    }
}
```

**Streaming variant — replace list accumulation with per-chunk flush:**
```java
// Streaming: read `chunkRows` lines, build DataFrame, return; loop until EOF.
while ((line = br.readLine()) != null) {
    if (currentRow < opts.skiprows) { currentRow++; continue; }
    if (line.trim().isEmpty()) continue;
    rows.add(parseRow(line));            // build into local List<double[]>
    if (rows.size() >= opts.chunkRows) break;
}
```

**Series assembly per chunk** (Pandas.java:101-109):
```java
// Source: modules/pandas/src/main/java/pandas/Pandas.java:101-109
Map<String, Series> seriesMap = new LinkedHashMap<>();
for (int c = 0; c < targetColsCount; c++) {
    double[] colDoubles = new double[dataRows];
    for (int r = 0; r < dataRows; r++) colDoubles[r] = data[r][c];
    Series s = isStringCol[c]
        ? new Series(finalColumns[c], colDoubles, stringData[c], index)
        : new Series(finalColumns[c], colDoubles, index);
    seriesMap.put(finalColumns[c], s);
}
return new DataFrame(seriesMap);
```

**Empty-file fail-fast guard** (Pandas.java:47):
```java
// Source: modules/pandas/src/main/java/pandas/Pandas.java:47
if (lines.isEmpty()) throw new IllegalArgumentException("File CSV rong hoac bi skip het data.");
```

**NaN-on-bad-numeric pattern** (Pandas.java:90-95):
```java
// Source: modules/pandas/src/main/java/pandas/Pandas.java:90-95
try { data[r][targetColCount] = Double.parseDouble(val); }
catch (NumberFormatException e) {
    data[r][targetColCount] = Double.NaN;
    stringData[targetColCount][r] = val;
    isStringCol[targetColCount] = true;
}
```

**`AutoCloseable` for file-handle hygiene** (ASVS V12 / STRIDE-T403):
```java
@Override public void close() throws IOException {
    if (br != null) br.close();
}
// Caller must use try-with-resources:
try (CsvChunkReader reader = new CsvChunkReader(path, opts)) {
    while (reader.hasNext()) { DataFrame chunk = reader.next(); /* ... */ }
}
```

**WR-05 defensive identity init** — `next = null; exhausted = false;` in field declarations; prime look-ahead in ctor via `advance()`.

---

### `modules/pandas/src/main/java/pandas/internal/RunningGroupAggregator.java` (merge-aware aggregator)

**Analog:** `GroupBy` (GroupBy.java:14-24 for index-map, GroupBy.java:71-85 for agg ops)

**Imports pattern** (mirror GroupBy.java:1-3):
```java
package pandas.internal;

import pandas.DataFrame;
import pandas.Series;
import java.util.*;

public final class RunningGroupAggregator {
```

**Index map pattern** (GroupBy.java:14-24):
```java
// Source: modules/pandas/src/main/java/pandas/GroupBy.java:14-24
public GroupBy(DataFrame df, String groupColumn) {
    this.df = df;
    this.groupColumn = groupColumn;
    this.groups = new LinkedHashMap<>();
    Series key = df.getColumn(groupColumn);
    for (int i = 0; i < key.size(); i++) {
        String k = key.getString(i);
        groups.computeIfAbsent(k, x -> new ArrayList<>()).add(i);
    }
}
```

**Per-group accumulator** (GroupBy.java:71-85):
```java
// Source: modules/pandas/src/main/java/pandas/GroupBy.java:71-85
private double computeAgg(double[] values, String func) {
    switch (func) {
        case "sum": { double s = 0; for (double v : values) s += v; return s; }
        case "mean": { double s = 0; for (double v : values) s += v; return s / values.length; }
        case "count": return values.length;
        case "min": { double m = Double.MAX_VALUE; for (double v : values) if (v < m) m = v; return m; }
        case "max": { double m = -Double.MAX_VALUE; for (double v : values) if (v > m) m = v; return m; }
        // ...
    }
}
```

**Streaming accumulator pattern (WR-05 explicit identity init):**
```java
private static final class RunningAgg {
    double sum, sumSq;
    long count;
    double min = Double.MAX_VALUE;    // WR-05: explicit identity, NOT field default 0.0
    double max = -Double.MAX_VALUE;   // WR-05: explicit identity, NOT field default 0.0

    void merge(double v) {
        if (Double.isNaN(v)) return;          // mirror Series.sum NaN-skip (Series.java:69)
        sum   += v;
        sumSq += v * v;
        count++;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    void merge(RunningAgg o) {                // associativity — merge two accumulators
        sum   += o.sum;  sumSq += o.sumSq;  count += o.count;
        if (o.min < min) min = o.min;
        if (o.max > max) max = o.max;
    }
}
```

**NaN-skip policy** (Series.java:68-69):
```java
// Source: modules/pandas/src/main/java/pandas/Series.java:68-69
public double sum() {
    double s = 0; for (double v : data) if (!Double.isNaN(v)) s += v; return s;
}
```

**Mean/std derivation at finalize (matches GroupBy.computeAgg):**
- `mean = sum / count` (identical to GroupBy.java:74)
- `std = sqrt(sumSq/count - mean²)` if `count > 1` else `0` (analog of GroupBy.java:78-82)

---

### `modules/pandas/src/main/java/pandas/Pandas.java` — ADDITIVE `read_csv_streaming(...)`

**Analog:** existing `read_csv(String, ReadCsvOptions)` (Pandas.java:34-112) — same parser logic, different output shape (Iterator<DataFrame> instead of single DataFrame).

**New method signature (additive, no existing signature touched):**
```java
// NEW — additive, public static
public static Iterator<DataFrame> read_csv_streaming(String filepath, ChunkedReadOptions options) throws IOException {
    Path p = java.nio.file.Paths.get(filepath).normalize();
    if (!Files.isReadable(p)) throw new IllegalArgumentException("Cannot read path: " + filepath);
    return new CsvChunkReader(p, options);
}
```

**Carry-forward patterns from Pandas.read_csv:**
- `try-with-resources` for BufferedReader (Pandas.java:37) → `AutoCloseable` on CsvChunkReader
- Skip skiprows + skip blank lines (Pandas.java:41-42) → mirrored in CsvChunkReader.advance()
- `path.normalize()` (ASVS V4) — new for Phase 4, not in existing read_csv
- `IllegalArgumentException` on empty file (Pandas.java:47) → mirrored in CsvChunkReader ctor

**Public API guard:** `grep -c "public static" Pandas.java` must remain unchanged (per STATE.md frozen-API lock at v0.2.0). Add exactly +1 method.

---

### `modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java` — ADDITIVE `partial_fit(...)`

**Analog:** existing `fit(NDArray, int[])` body (GaussianNB.java:18-47)

**Existing fit pattern — per-class running accumulator:**
```java
// Source: modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java:18-47
public GaussianNB fit(NDArray X, int[] y) {
    double[][] data = extract2D(X);
    int n = data.length, d = data[0].length;
    n_classes = 0;
    for (int l : y) if (l > n_classes) n_classes = l;
    n_classes++;

    means = new double[n_classes][d];
    variances = new double[n_classes][d];
    classPriors = new double[n_classes];
    int[] counts = new int[n_classes];
    for (int l : y) counts[l]++;
    for (int c = 0; c < n_classes; c++) classPriors[c] = (double) counts[c] / n;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) means[y[i]][j] += data[i][j];
    for (int c = 0; c < n_classes; c++)
        if (counts[c] > 0) for (int j = 0; j < d; j++) means[c][j] /= counts[c];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) {
            double diff = data[i][j] - means[y[i]][j];
            variances[y[i]][j] += diff * diff;
        }
    for (int c = 0; c < n_classes; c++)
        if (counts[c] > 0) for (int j = 0; j < d; j++) variances[c][j] = variances[c][j] / counts[c] + 1e-9;

    fitted = true;
    return this;
}
```

**New partial_fit method signature (additive, no existing signature touched):**
```java
// NEW — additive, public, returns this for chaining (matches fit() style)
public GaussianNB partial_fit(NDArray X, int[] y) {
    // Lazy init on first call; accumulate thereafter (STRIDE-T402: stale state defense)
    if (!initialized) init(X, y);
    else accumulate(X, y);
    fitted = false;  // require finalize() or predict() lazy-divide
    return this;
}

public GaussianNB finalize() { divide(); fitted = true; return this; }
```

**Refactor opportunity (do this):** Extract `init()`, `accumulate(X, y)`, `divide()` from existing `fit` body — `fit` then calls `init` + `accumulate` + `divide`. `partial_fit` calls `init` (first call only) + `accumulate`. Avoids duplicate accumulator math.

**WR-05 carry-forward (defensive identity init for RunningAgg-style fields):** Already present — `means = new double[n_classes][d]` zeroes correctly via Java array default; the analogue in partial_fit is `initialized = false` initially.

**Cross-path equivalence invariant** (RESEARCH Pitfall 3): `partial_fit(X1,y1) + partial_fit(X2,y2) + finalize()` must equal `fit(X_full, y_full)` to within `1e-9 * n_classes`.

---

### `modules/sklearn/src/main/java/sklearn/pipeline/PandasPipeline.java` (builder)

**Analog:** existing `Pipeline.java` (Pipeline.java:1-85) — same package, same role (orchestrator). **Different style:** builder (fluent) instead of reflective.

**Imports pattern** (mirror Pipeline.java:1-7 + add NIO):
```java
package sklearn.pipeline;

import pandas.DataFrame;
import pandas.internal.ChunkedReadOptions;
import pandas.internal.CsvChunkReader;
import java.io.IOException;
import java.nio.file.Path;
```

**Fluent-setter pattern** (mirror Pandas.ReadCsvOptions fluent style, Pipeline.java:1-85 for package conventions):
```java
public final class PandasPipeline {
    private Path path;
    private ChunkedReadOptions opts = new ChunkedReadOptions();
    private PartialFitCapable estimator;

    public PandasPipeline load(Path p) { this.path = p; return this; }
    public PandasPipeline chunk(int rows) { this.opts.chunkRows = rows; return this; }
    public PandasPipeline partialFit(PartialFitCapable e) { this.estimator = e; return this; }

    public <E extends PartialFitCapable> E run() throws IOException {
        if (path == null) throw new IllegalArgumentException("load(path) required.");
        if (estimator == null) throw new IllegalArgumentException("partialFit(estimator) required.");
        try (CsvChunkReader reader = new CsvChunkReader(path, opts)) {
            while (reader.hasNext()) {
                DataFrame chunk = reader.next();
                estimator.partial_fit_chunk(/* extract X, y from chunk */);
            }
        }
        estimator.finalize();
        @SuppressWarnings("unchecked") E e = (E) estimator;
        return e;
    }
}
```

**`<10` line caller pattern (USE-02 success criterion):**
```java
GaussianNB model = new PandasPipeline()
    .load(Paths.get("big.csv"))
    .chunk(10_000)
    .partialFit(new GaussianNB())
    .run();
```

**Differences from existing Pipeline.java:**
- No reflection (existing Pipeline.java:25-39 uses `Method.invoke`)
- No `List<Object> steps` (single estimator, not chain)
- File-based input (CsvChunkReader) vs NDArray input

**Naming/style conventions borrowed from Pipeline.java:**
- `package sklearn.pipeline;`
- Same javadoc preamble style (`/** ... */`)
- `throw new RuntimeException("...", e)` for wrapped errors

---

### Test files (`CsvChunkReaderTest`, `RunningGroupAggregatorTest`, `PandasStreamingTest`, `GaussianNBPartialFitTest`, `PandasPipelineTest`)

**Analog for test hygiene:** `ParallelOpsTest.java:27-32` — `@After` reset pattern

```java
// Source: modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java:27-32
@After
public void reset() {
    ParallelOps.resetThresholdForTesting();
}
```

**Imports + JUnit 4 conventions** (mirror existing test files):
```java
package ...;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class XxxTest {
    @Before public void setUp() throws Exception { /* temp file or fixture */ }
    @After  public void tearDown() throws Exception { /* cleanup temp file */ }

    @Test public void smallFile_inMemoryPathUnchanged() { /* ... */ }
}
```

**Cross-path equivalence test pattern** (RESEARCH Validation Map):
```java
@Test public void twoPartialFits_equalOneFit() {
    int n = 1000, d = 4;
    NDArray X1 = /* rows 0..499 */; int[] y1 = /* ... */;
    NDArray X2 = /* rows 500..999 */; int[] y2 = /* ... */;
    NDArray Xfull = /* rows 0..999 */; int[] yfull = /* ... */;

    GaussianNB a = new GaussianNB();
    a.partial_fit(X1, y1).partial_fit(X2, y2).finalize();

    GaussianNB b = new GaussianNB();
    b.fit(Xfull, yfull);

    double[] pA = a.predict(Xfull), pB = b.predict(Xfull);
    assertArrayEquals(pB, pA, 0);  // categorical predictions must match
}
```

**Test directory convention:** existing pandas tests live at `modules/pandas/src/test/java/com/numja/pandas/` — use same path for new tests, NOT `com/pandas/` as Wave 0 literal text says.

---

## Shared Patterns

### Pattern 1: WR-05 Defensive Identity Init (Phase 2/3 carry-forward)

**Source:** `modules/numja/src/main/java/numja/core/ParallelOps.java:317-320` (MinTask ctor)

**Apply to:** `CsvChunkReader` (exhausted=false, next=null in field decls), `RunningGroupAggregator.RunningAgg` (min=MAX_VALUE, max=-MAX_VALUE in field decls), `GaussianNB` (initialized=false).

**Why:** Java field defaults (0.0 for double, false for boolean) are wrong identity elements for min/max/accumulator semantics. Explicit ctor or field-init prevents first-merge bugs.

```java
// Ponytail: min/max identity is non-zero — explicit field init prevents silent drift
private double min = Double.MAX_VALUE;
private double max = -Double.MAX_VALUE;
```

---

### Pattern 2: WR-03 `@After` Reset Hygiene (Phase 2/3 carry-forward)

**Source:** `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java:27-32`

**Apply to:** Every test class that mutates package-private test hooks (e.g., `ChunkedReadOptions` defaults, `Runtime.getRuntime().maxMemory()` stubs).

```java
@After
public void tearDown() {
    // reset test-only state to avoid leaking across test classes (T-2-05)
}
```

---

### Pattern 3: ASVS L1 + STRIDE Threat Model Block

**Source:** project constraint (per STATE.md / RESEARCH §Security Domain)

**Apply to:** Every NEW public class javadoc. Trust boundary = filesystem → in-process objects. Threats = STRIDE table from RESEARCH §Known Threat Patterns.

**Required javadoc preamble:**
```java
/**
 * <one-line summary>
 *
 * Trust boundary: filesystem (caller-controlled path) → in-process DataFrame/NDArray.
 * STRIDE threats: [path traversal, OOM, CSV formula injection, file handle leak].
 * Mitigations: path.normalize() + Files.isReadable() check (V4);
 *               chunkRows cap (V12); leading-char NaN substitution (V5);
 *               AutoCloseable + try-with-resources (V12).
 */
```

---

### Pattern 4: IllegalArgumentException + descriptive message

**Source:** `Pandas.java:47` (`"File CSV rong hoac bi skip het data."`), `DataFrame.java:23-26`, `Series.java:48`.

**Apply to:** Every input validation point in new classes. Vietnamese-English hybrid messages accepted (matches existing style).

```java
if (chunkRows < 1) throw new IllegalArgumentException("chunkRows must be >= 1, got " + chunkRows);
```

---

### Pattern 5: try-with-resources for file/IO handles

**Source:** `Pandas.java:37`, `DataFrame.to_csv` (DataFrame.java:246-259).

**Apply to:** `CsvChunkReader` (must be AutoCloseable), `PandasPipeline.run()` (wraps reader).

```java
try (CsvChunkReader reader = new CsvChunkReader(path, opts)) {
    while (reader.hasNext()) { /* ... */ }
}  // closes br even on early break (STRIDE-T403)
```

---

### Pattern 6: NaN-skip policy for aggregations

**Source:** `Series.java:68-69` (`sum`), `Series.java:82-84` (`min`), `GroupBy.computeAgg` (implicit — `values` array pre-built by `Series.get()` which can be NaN).

**Apply to:** `RunningGroupAggregator` (skip NaN in `merge(double)`).

```java
if (Double.isNaN(v)) return;  // mirror Series.sum NaN-skip
```

---

### Pattern 7: `LinkedHashMap` for stable order + `computeIfAbsent`

**Source:** `Pandas.java:101`, `GroupBy.java:17`, `GroupBy.java:22`.

**Apply to:** `RunningGroupAggregator` (Map<String, RunningAgg>), `CsvChunkReader` index lookup (Map<String, Integer> if needed).

```java
groups.computeIfAbsent(k, x -> new RunningAgg()).merge(v);
```

---

## No Analog Found

None — every new file has an exact or role-match analog in the existing codebase. Pattern coverage is complete.

---

## Metadata

**Analog search scope:** `modules/pandas/src/main/java/pandas/`, `modules/pandas/src/test/java/com/numja/pandas/`, `modules/sklearn/src/main/java/sklearn/`, `modules/sklearn/src/test/java/sklearn/`, `modules/numja/src/main/java/numja/core/`, `modules/numja/src/test/java/com/numja/core/`.

**Files scanned:** 17 (Pandas.java, GroupBy.java, DataFrame.java, Series.java, DataFrameStats.java, GaussianNB.java, KMeans.java, LinearRegression.java, Pipeline.java, ParallelUtils.java, NumJa.java, ArrayOps.java, ParallelOps.java, NDArray.java, PandasBench.java, SklearnBench.java, ParallelOpsTest.java).

**Pattern extraction date:** 2026-08-28.

**Carry-forward patterns verified:** WR-05 (defensive identity init), WR-03 (`@After` reset hygiene), ASVS L1 + STRIDE block (project constraint).

**Public API lock verified:** `Pandas.read_csv(String)`, `Pandas.read_csv(String, ReadCsvOptions)`, `GaussianNB.fit(NDArray, int[])` signatures UNCHANGED. New methods are strict additions (`read_csv_streaming`, `partial_fit`, `finalize`).

---

## PATTERN MAPPING COMPLETE

6 / 6 files have exact or role-match analogs; 6 patterns identified for cross-cutting reuse.
