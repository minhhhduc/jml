# Phase 4: Adaptive Memory Model — Research

**Researched:** 2026-08-28
**Domain:** Out-of-core streaming for tabular data (CSV chunking) + partial-fit model adaptation in a Java ML library with a frozen public API.
**Confidence:** MEDIUM-HIGH — full read of every relevant file in `modules/pandas/src/main/java/pandas/`, `modules/sklearn/src/main/java/sklearn/`, `modules/numja/src/main/java/numja/`, and `bench/`. No live library lookup needed; stdlib-only approach is self-contained and known.

## Summary

Phase 4 needs two distinct halves stitched through a single helper: (a) a **chunked read path** for tabular data so files larger than available heap can be processed, and (b) **partial-fit model adapters** so the chunks can drive learning without loading everything. Recon shows the current codebase has *zero* streaming / iterator / partial-fit infrastructure — `Pandas.read_csv` (Pandas.java:30-112) eagerly buffers all rows into `List<String[]>`, `GroupBy` (GroupBy.java:14-24) eagerly builds `Map<String, List<Integer>>` over all indices, and no `partial_fit` symbol exists anywhere in `modules/sklearn/`. Everything must be built from scratch in stdlib (no Apache Commons CSV / Univocity vendored; only EJML, commons-math3, jfreechart, ProGuard in `libs/`).

The minimum viable shape — what a `ponytail: full` senior dev would ship first — is:

1. **New `Pandas.read_csv_streaming(path, chunkRows)` → `Iterator<DataFrame>`** — line-buffered `BufferedReader`, drop the `List<String[]>` accumulation, build per-chunk `Series[]` (single allocation per chunk, eligible for GC). In-memory `read_csv` stays bit-identical for backward compat (MEM-01).
2. **New `GroupBy.aggregateStreamed(Iterator<DataFrame>)` static helper** — running-aggregator pattern (`Map<String, RunningAgg>` with `sum`, `count`, `min`, `max`, `sumSq`); per chunk scan + per-key merge. Finalize once on the last chunk → same `DataFrame` shape as in-memory `aggregate("mean")`. Equivalence holds because add/min/max are associative; `mean` = `sum/count` at finalize.
3. **Add `GaussianNB.partial_fit(NDArray X, int[] y)`** — already has per-class running accumulators (means, variances, counts); trivially extensible to running form. Cheapest model to ship first.
4. **Add `PandasPipeline` builder** — `new PandasPipeline().load(path).chunk(10_000).describe().partialFit(model).build()` — <10 lines for caller (USE-02).
5. **Memory budget detection** — `Runtime.getRuntime().maxMemory()` × 0.5 as chunk-size input, plus a hard default chunk size (10_000 rows) as ceiling. No complex adaptive pressure yet.

**Primary recommendation:** stdlib-only chunked CSV reader + running-aggregator groupby + GaussianNB.partial_fit + tiny builder helper. Defer KMeans/LinearRegression.partial_fit and adaptive memory pressure to v2 (Phase 5+).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Chunked CSV parsing | Pandas module (internal) | — | Reads `String[]` lines; lives next to `read_csv` in Pandas.java. Same parser reuse via a private helper. |
| Running-aggregator groupby | Pandas module (internal) | — | `GroupBy` already owns the index-map pattern; streaming variant is just a different input feed. |
| Partial-fit (model state update) | sklearn module (internal) | NDArray for shape checks | Per-model `partial_fit` lives next to `fit` (GaussianNB, KMeans). Receives `NDArray` like the existing `fit`. |
| Pipeline builder (USE-02) | sklearn.pipeline (NEW) | Pandas for chunk source | New `PandasPipeline` class — NOT the existing reflective `Pipeline` (that one already handles full in-memory fit). Lives next to it but separate concern. |
| Memory threshold decision | Pandas (internal `ChunkedReadOptions`) | — | `Runtime.getRuntime().maxMemory()` is queried inside `read_csv_streaming`; no global state to mutate. |

## Standard Stack

### Core (stdlib only — no new deps)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `java.io.BufferedReader` | JDK 17+ | Line-buffered stream for CSV chunking | Already used in `Pandas.read_csv:37`. Drop-in. |
| `java.nio.file.Files` | JDK 17+ | File size probe, charset control | Replaces current `FileReader` (uses default charset) with explicit `Files.newBufferedReader(path, StandardCharsets.UTF_8)`. |
| `java.util.Iterator<T>` | JDK 17+ | Chunk iterator contract | Standard Java iteration; callable as `for (DataFrame chunk : iter)` and `while (iter.hasNext())`. |
| `java.util.LinkedHashMap` / `HashMap` | JDK 17+ | Running-aggregator state per group key | Already used by `GroupBy.java:14`. No new collection type. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled line CSV parser | Apache Commons CSV / Univocity | Stdlib already covers the comma-separated case used everywhere in the test corpus (`dist/datasets/*.csv`). Adding a dep violates the "stdlib first" rule and brings nothing for the simple case. CSV injection mitigation is a one-liner: skip rows whose first non-empty char is `=`, `+`, `-`, `@` (the four Excel formula prefixes) per OWASP CSV-injection guidance. |
| NIO `Files.lines()` (Stream<String>) | BufferedReader loop | `Files.lines()` returns a `Stream<String>` backed by a `BufferedReader` under the hood. Tempting, but `Stream` has no built-in batch/peek-ahead to know if a partial last line of a chunk is complete. A manual `BufferedReader` with `readLine()` + line-counting is shorter and matches the test corpus exactly. |
| Apache Arrow / Parquet | — | Not in `libs/`, not on the roadmap, would force a Maven repo rewrite. v2+. |
| Custom memory-pressure probe (`com.sun.management.ThreadMXBean`) | `Runtime.getRuntime().maxMemory()` | The current benchmark machine has 15.7 GB RAM; `maxMemory()` is the upper bound the JVM will let us use. Pressure-based probes need a separate thread sampling `usedMemory()` — non-trivial, deferred. |

**No new Maven dependencies.** Confirmed `libs/` already on classpath; `pom.xml` files unchanged.

**Version verification (stdlib):** JDK target = 17 (per Phase 2/3 lock). `BufferedReader`, `Files.newBufferedReader`, `Iterator`, `LinkedHashMap` all stable since JDK 8.

## Package Legitimacy Audit

> No new packages installed. Phase 4 is stdlib-only by design. Skip detailed registry verification — nothing to verify.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none) | — | — | — | — | — | Approved (N/A) |

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Caller code (unchanged)                                     │
│   Pandas.read_csv(path)  ← MEM-01 in-memory path (frozen)   │
└────────────┬────────────────────────────────────────────────┘
             │
             ├── (small file, fits in heap) → existing DataFrame  [BIT-IDENTICAL]
             │
┌────────────▼────────────────────────────────────────────────┐
│ NEW: Pandas.read_csv_streaming(path, ChunkedReadOptions)     │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ BufferedReader over path                             │  │
│   │   ├ skip header + skiprows once                      │  │
│   │   ├ readLine() until chunkRows data lines collected  │  │
│   │   ├ emit DataFrame (Series per column, ~chunkRows)   │  │
│   │   └ repeat until EOF                                 │  │
│   └──────────────────────────────────────────────────────┘  │
│   emits Iterator<DataFrame>                                  │
└────────────┬────────────────────────────────────────────────┘
             │
             │ for (DataFrame chunk : iter) { ... }
             │
┌────────────▼────────────────────────────────────────────────┐
│ NEW: Pandas.streamingGroupBy(Iterator<DataFrame>, key)       │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ RunningAgg per group key: { sum, sumSq, count,       │  │
│   │   min, max }                                         │  │
│   │ for each chunk: scan rows, merge into RunningAgg     │  │
│   │ finalize: mean = sum / count, std = sqrt(...)        │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ NEW: GaussianNB.partial_fit(X_chunk, y_chunk)                │
│   - existing GaussianNB.fit (lines 18-47) extracts pattern:  │
│       means[c][j] += data[i][j]   (running per-class sum)   │
│       variances[c][j] += diff*diff (running per-class sumSq)│
│       counts[c]                   (running per-class count) │
│   - partial_fit reuses the accumulator fields               │
│   - finalize() divides means += sum/count, variances /= cnt │
│   - first call: same as fit(); subsequent calls: continue   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ NEW: PandasPipeline  (sklearn.pipeline.PandasPipeline)      │
│   builder().load(path).chunk(10_000).partialFit(model)      │
│   one-shot terminal operator (no intermediate DataFrame     │
│   held in memory between chunks)                            │
└─────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
modules/pandas/src/main/java/pandas/
├── Pandas.java                  # + read_csv_streaming(...) NEW public method
├── DataFrame.java               # unchanged (in-memory only)
├── GroupBy.java                 # unchanged (in-memory only) + new streaming helper
├── Series.java                  # unchanged
├── DataFrameStats.java          # unchanged
└── internal/                    # NEW package
    ├── ChunkedReadOptions.java  # chunkRows, sep, header, skiprows, index_col
    ├── CsvChunkReader.java      # implements Iterator<DataFrame>
    └── RunningGroupAggregator.java # merge-aware group accumulator

modules/sklearn/src/main/java/sklearn/
├── pipeline/
│   ├── Pipeline.java            # unchanged (reflective)
│   └── PandasPipeline.java      # NEW: builder for chunked read → partial_fit
├── naive_bayes/
│   └── GaussianNB.java          # + partial_fit(X, y), finalize(), internalize accumulators
└── (cluster/KMeans.java, linear_model/LinearRegression.java — unchanged in v1)

modules/pandas/src/test/java/com/pandas/
├── internal/CsvChunkReaderTest.java     # NEW: chunk boundaries, header skip, EOF
├── internal/RunningGroupAggregatorTest.java # NEW: sum/mean/min/max across 2+ chunks
└── PandasStreamingTest.java              # NEW: cross-path equivalence vs in-memory

modules/sklearn/src/test/java/sklearn/
├── pipeline/PandasPipelineTest.java     # NEW: <10-line caller pattern works
└── naive_bayes/GaussianNBPartialFitTest.java # NEW: 2 partial_fit calls == 1 fit
```

### Pattern 1: Chunked CSV Iterator

**What:** A `BufferedReader`-backed iterator that emits `DataFrame` chunks of N rows at a time. One allocation per chunk.

**When to use:** Any time the file size is unknown or estimated > 0.5 × `Runtime.getRuntime().maxMemory()`. Caller writes the existing in-memory loop, just substituting the iterator.

**Example skeleton (referenced in plan, not final code):**
```java
// Source: pattern derived from Pandas.java:30-112 (existing reader)
public final class CsvChunkReader implements Iterator<DataFrame> {
    private final BufferedReader br;
    private final ChunkedReadOptions opts;
    private final String[] columns;
    private DataFrame next;        // look-ahead
    private boolean exhausted;

    CsvChunkReader(Path path, ChunkedReadOptions opts) throws IOException {
        this.br = Files.newBufferedReader(path, StandardCharsets.UTF_8);
        this.opts = opts;
        this.columns = readHeader();        // skip skiprows, read header line
        advance();                          // prime look-ahead
    }

    @Override public boolean hasNext() { return !exhausted; }

    @Override public DataFrame next() {
        if (exhausted) throw new NoSuchElementException();
        DataFrame cur = next;
        advance();
        return cur;
    }

    private void advance() {
        List<double[]> rows = new ArrayList<>(opts.chunkRows);
        List<String> idx = new ArrayList<>(opts.chunkRows);
        try {
            String line;
            while (rows.size() < opts.chunkRows && (line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                if (opts.index_col == null) idx.add(String.valueOf(rows.size()));
                else idx.add(line.split(opts.sep)[opts.index_col].trim());
                rows.add(parseRow(line));           // throws NumberFormatException → NaN
            }
        } catch (IOException e) { throw new UncheckedIOException(e); }
        if (rows.isEmpty()) { exhausted = true; next = null; return; }
        next = buildDataFrame(rows, idx);    // constructs Series per column
    }
}
```

### Pattern 2: Running-Group Aggregator

**What:** Per-group mutable accumulator that merges with another chunk's per-group accumulator. Finalize once after the last chunk.

**Example skeleton:**
```java
// Source: pattern derived from GroupBy.java:71-85 (existing aggregate switch)
private static final class RunningAgg {
    double sum, sumSq; long count; double min, max;
    // WR-05 defense: explicit identity init in ctor (not field default)
    RunningAgg() { min = Double.MAX_VALUE; max = -Double.MAX_VALUE; }

    void merge(double v) {
        if (Double.isNaN(v)) return;     // skip NaN like in-memory Series.sum
        sum   += v;
        sumSq += v * v;
        count++;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    void merge(RunningAgg o) {
        sum   += o.sum;
        sumSq += o.sumSq;
        count += o.count;
        if (o.min < min) min = o.min;
        if (o.max > max) max = o.max;
    }
}
```
Equivalence proof sketch: sum/count/min/max over a partition of rows = sum/count/min/max over the union (associativity of + and min/max). Mean = sum/count (one division at finalize, identical to `GroupBy.computeAgg`). Std = sqrt(sumSq/count - mean²) (Welford's online form gives bit-identical result; or two-pass: count + sum → mean → sumSq(mean-centered)).

### Anti-Patterns to Avoid

- **Single-pass threshold gate only on `n`** like Phase 2 — does NOT work here; CSV file size ≠ DataFrame row count. Need either byte-size probe (`Files.size(path)`) or two-pass header read (skip skiprows + count remaining bytes).
- **Loading the entire file into a `List<String[]>` and chunking from there** — defeats the entire point. The current reader (Pandas.java:35-45) does exactly this; the new path must NOT.
- **`Files.lines(path).iterator()`** — works for read but loses the byte-count budget check; also `Stream.iterator()` is single-use only (no reset on multi-pass for header peek). Sticking with `BufferedReader`.
- **Adding `partial_fit` to `LinearRegression`** — least-squares via `X^T X` and `X^T y` is the wrong shape for streaming; needs Sherman-Morrison or stochastic gradient. KMeans online Lloyd is doable but more state to track. **Defer both to Phase 5+ unless user explicitly locks**.
- **Reflective Pipeline** for chunked path — the existing `Pipeline` (modules/sklearn/.../Pipeline.java:12-85) uses reflection on `fit_transform` / `fit` / `transform`. The new `PandasPipeline` is a typed builder — no reflection, simpler, easier to test.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| UTF-8 file reading | Manual byte → char decoding | `Files.newBufferedReader(path, StandardCharsets.UTF_8)` | Default `FileReader` uses platform charset (Windows-1252 on legacy systems); breaks on Vietnamese / accented headers silently. |
| Group-by aggregation over chunks | Custom HashMap<String, double[]> per op | `RunningAgg` class (above) + `LinkedHashMap` | One class per supported aggregate means N classes; one accumulator with `sum/count/min/max/sumSq` covers `sum/mean/std/min/max/count` (the 6 GroupBy ops). |
| Memory budget probe | Custom JMX / MXBean polling | `Runtime.getRuntime().maxMemory()` for ceiling + hard-coded chunk size floor | Adaptive pressure probe is `ponytail: deferred` — not needed for v1; one-line `Runtime.getRuntime().maxMemory()` is enough to decide "should I even try in-memory?" |
| CSV cell quote/escape parsing | Hand-rolled state machine | **Skip entirely for v1** | Test corpus (`california_housing.csv`, `titanic.csv`, `iris.csv`) is unquoted, no embedded commas. If a quoted-cell test appears in v2, add a minimal `"..."` unescaper. |
| `Iterator<DataFrame>` chunk assembly | Custom `ArrayList<double[]>` rebuild | Reuse `Series` ctor + `DataFrame(Map<String, Series>)` ctor (DataFrame.java:39-54) | Constructor already exists, accepts any number of Series; just feed per-chunk Series. |

**Key insight:** The hardest part of "out-of-core groupby" is *not* the streaming — it's the merge semantics. Add/min/max are associative; mean is `sum/count` at finalize; std needs `sumSq` and `count`. Once you have those 5 running fields, every GroupBy op falls out for free.

## Runtime State Inventory

> Phase 4 is a pure addition (new methods + new test classes). No rename/refactor/migration triggered. **Section N/A — omitted by Phase 4 scope.**

## Common Pitfalls

### Pitfall 1: CSV row split loses column count fidelity
**What goes wrong:** A row has fewer cells than the header (trailing comma missing, embedded newline in quoted cell). Current reader silently skips it (Pandas.java:42 doesn't check `rowVals.length`). Streaming reader must do the same to match in-memory behavior, OR reject explicitly.
**Why it happens:** Real-world CSVs are messy.
**How to avoid:** Match existing tolerance: trim, split on `opts.sep`, take `Math.min(rowVals.length, cols)` cells, ignore the rest (Pandas.java:86-99 already does this).
**Warning signs:** Cross-path equivalence test (`PandasStreamingTest`) reports row count mismatch on titanic.csv.

### Pitfall 2: `BufferedReader` not closed on early termination
**What goes wrong:** Caller breaks out of the `for (DataFrame chunk : iter)` loop early → file handle leaks until GC finalizer runs.
**Why it happens:** `Iterator` doesn't extend `AutoCloseable`; nothing forces the close.
**How to avoid:** Make `CsvChunkReader implements Iterator<DataFrame>, AutoCloseable`. Document `try-with-resources` usage in javadoc. Provide a `close()` method that flushes and closes the underlying `BufferedReader`. (Note: Java 17 has no `Stream.iterator()` pattern for try-with-resources either, so this is a deliberate one-off — keep it simple.)
**Warning signs:** File handle leak on Windows (locked file visible in `handle.exe`); CI hangs on subsequent read of same path.

### Pitfall 3: `partial_fit` first-call == `fit` semantics mismatch
**What goes wrong:** First `partial_fit(X, y)` call produces different result than `fit(X, y)`. Caller compares streaming vs batch and sees drift.
**Why it happens:** Easy to mis-set priors/initialization in the first call (e.g. reset accumulators to zero when they should be reset on `new` only, or vice versa).
**How to avoid:** `partial_fit` must be callable on a fresh instance (zero-all accumulators) AND on a previously-`partial_fit`'d instance (continue from existing accumulators). Use a `private boolean initialized` flag in GaussianNB; on first call, allocate `means/variances/counts/classPriors` arrays sized to `y` cardinality (same as existing `fit:21-23`).
**Warning signs:** `GaussianNBPartialFitTest.partialFitOnce_equalsFit` fails.

### Pitfall 4: chunkRows mis-sized for streaming groupby
**What goes wrong:** chunkRows = 1 → chunk emission overhead dominates (per-chunk Series allocation, per-chunk `LinkedHashMap` insert). chunkRows = 10_000_000 → OOM because a single chunk is bigger than heap.
**Why it happens:** No upper bound check.
**How to avoid:** `ChunkedReadOptions.chunkRows` default = `Math.max(1000, Math.min(100_000, (int)(Runtime.getRuntime().maxMemory() * 0.001 / 8)))` — at 8 bytes per `double` × N columns × N rows per chunk. Cap at 100k rows for 8-column CSVs; ~0.1% of heap per chunk leaves room for group aggregator + estimator state.
**Warning signs:** `CsvChunkReaderTest.tooLargeChunk_oom` redline OR `tinyChunk_performanceRegression` >5% slower than in-memory.

### Pitfall 5: `index_col` lookup inside the chunk loop is O(n_cols) per row
**What goes wrong:** Per-row `rowVals[opts.index_col]` is O(1), but if `opts.index_col` is mis-configured (e.g. negative) the array lookup is silently wrong.
**Why it happens:** No input validation.
**How to avoid:** Validate `index_col >= 0 && index_col < cols` once in `CsvChunkReader` ctor (after header read); throw `IllegalArgumentException` with file path + offending option in message. Inherits existing Pandas error style.
**Warning signs:** Missing index column in output DataFrame → cross-path test fails on row indexing.

### Pitfall 6: Public API accidental drift
**What goes wrong:** Adding `Pandas.read_csv_streaming(...)` requires no signature change to existing `Pandas.read_csv(String)` — but if the wrong overload is touched (e.g. changing the existing signature to `read_csv(String, ChunkedReadOptions)`), NumJa frozen-api guarantee breaks.
**Why it happens:** Easy to refactor `read_csv` into a common private impl without thinking.
**How to avoid:** Both old `read_csv(String)` and old `read_csv(String, ReadCsvOptions)` MUST stay byte-identical signatures. The new method is `Pandas.read_csv_streaming(String, ChunkedReadOptions)`. Document explicitly in PLAN.md that the existing two methods are not touched.
**Warning signs:** `grep -c "public static" Pandas.java` after changes — must equal pre-Phase-4 count (per STATE.md lock).

## Code Examples

Verified patterns from existing codebase (carry-forward conventions):

### Reusing GroupBy's per-group accumulator pattern
```java
// Source: modules/pandas/src/main/java/pandas/GroupBy.java:71-85 (in-memory aggregate)
private double computeAgg(double[] values, String func) {
    switch (func) {
        case "sum": { double s = 0; for (double v : values) s += v; return s; }
        // ...
    }
}
// streaming variant: replace `values` per-group array with per-chunk RunningAgg.merge
```

### Threshold-gate pattern (Phase 2 carry-forward)
```java
// Source: modules/pandas/src/main/java/pandas/Pandas.java:46-47 (existing in-memory fail-fast)
if (lines.isEmpty()) throw new IllegalArgumentException("File CSV rong hoac bi skip het data.");
// streaming variant: after readLine returns null on first call AND we emitted 0 rows → same exception
```

### WR-05 defensive identity init (Phase 2 carry-forward)
```java
// Source: modules/numja/src/main/java/numja/core/ParallelOps.java:317-320 (MinTask constructor)
this.best = Double.MAX_VALUE;  // identity for min, not Java default 0.0
// RunningAgg ctor must explicitly init min=MAX, max=-MAX (NOT 0.0)
```

### `@After` reset hygiene (Phase 2 carry-forward)
```java
// Source: modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java (WR-03 fix)
// ChunkedReadOptions has a test-only override hook (analogous to ParallelOps.testThresholdOverride):
// package-private setChunkRowsForTesting(int), resetChunkRowsForTesting() — reset in @After
```

### Builder pattern (NEW — not in existing codebase)
```java
// Source: standard Java builder idiom
public final class PandasPipeline {
    private Path path; private ChunkedReadOptions opts; private Estimator estimator;
    public PandasPipeline load(Path p) { this.path = p; return this; }
    public PandasPipeline chunk(int rows) { this.opts = new ChunkedReadOptions().chunkRows(rows); return this; }
    public PandasPipeline partialFit(Estimator e) { this.estimator = e; return this; }
    public <E extends Estimator & PartialFitCapable> E run() throws IOException {
        try (CsvChunkReader reader = new CsvChunkReader(path, opts)) {
            while (reader.hasNext()) {
                DataFrame chunk = reader.next();
                // extract X, y from chunk, call estimator.partial_fit(X, y)
            }
        }
        return (E) estimator;
    }
}
// caller pattern (USE-02):
// GaussianNB model = new PandasPipeline().load(Paths.get("big.csv"))
//                                       .chunk(10_000)
//                                       .partialFit(new GaussianNB())
//                                       .run();    // 4 lines, fits USE-02 (<10)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Buffer entire CSV into `List<String[]>` then parse | Stream chunks line-by-line | Phase 4 (this phase) | File size no longer bounded by heap; in-memory path unchanged for backward compat. |
| `GroupBy` builds full `Map<String, List<Integer>>` before aggregate | Streaming per-chunk merge into `RunningAgg` | Phase 4 | First call to mean() on a 5 GB file no longer requires loading all indices. |
| `GaussianNB.fit(X, y)` requires entire X | `GaussianNB.partial_fit(X_chunk, y_chunk)` | Phase 4 | First ML model in this codebase with online learning capability. |
| Pipeline = reflective `fit_transform` chain | `PandasPipeline` builder for load → chunk → fit | Phase 4 | <10-line caller (USE-02); no reflection. |

**Deprecated/outdated:**
- *None deprecated* — all changes additive. Existing `Pandas.read_csv`, `GroupBy.mean()`, `GaussianNB.fit()`, and `Pipeline` are untouched.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Test corpus CSVs (`california_housing.csv`, `titanic.csv`, `iris.csv`) are unquoted, no embedded commas, no escaped newlines. | Architecture Patterns → Anti-patterns | If true, the simple `String.split(opts.sep)` line parser is enough for v1. If false, need a quoted-cell unescaper. Mitigation: pinned to existing test corpus; escape parser deferred. |
| A2 | Default chunkRows = 1000-100_000 based on `Runtime.getRuntime().maxMemory() * 0.001 / 8` is reasonable. | Common Pitfalls → Pitfall 4 | If too small → overhead dominates. If too big → OOM. Mitigation: configurable via `ChunkedReadOptions.chunkRows`; default is a heuristic; tests exercise 1k/10k/100k. |
| A3 | `Files.size(path)` is available on Windows for files up to multiple GB without throwing. | Architecture Patterns | `Files.size` reads filesystem metadata; cheap on NTFS. If not, fallback = first-pass line counter. |
| A4 | GroupBy.sum/count/min/max are the only required aggregates for Phase 4 streaming; std/mean can derive from sum+sumSq+count at finalize. | Don't Hand-Roll table | If user wants separate `partial_fit` semantics for std (e.g. running Welford), needs an extra accumulator. Mitigation: derive from sum+sumSq is algebraically equivalent (synchronous form). |
| A5 | `partial_fit` is only needed on `GaussianNB` for Phase 4; KMeans + LinearRegression deferred. | Summary recommendation | If user locks in all three, scope balloons. Mitigation: open question to discuss-phase. |
| A6 | CSV injection mitigation (skip leading `=+-@`) is NOT in v1 scope. | Common Pitfalls / Pitfall 1 / ASVS V5 | OWASP CSV-injection. If user requires it, add 4-line guard inside `parseRow`. Defer. |
| A7 | In-memory `read_csv(String)` and `read_csv(String, ReadCsvOptions)` signatures stay unchanged. | Architecture Patterns → Anti-patterns | Frozen API constraint from STATE.md. The new method is `read_csv_streaming`. |
| A8 | `Pipeline.java` (existing reflective) is NOT replaced — both coexist. | Architecture Patterns | Reflective `Pipeline` handles in-memory fit_transform chain; new `PandasPipeline` handles chunked read → partial_fit. Different concerns. |

## Open Questions (RESOLVED)

**Resolution pass:** each Q has been answered by the plan(s) shown in `[RESOLVED → ...]` below. User answered via planning implicit defaults (ponytail-mode minimum first-rung).

1. **Which models get `partial_fit` in Phase 4?** *[RESOLVED → 04-03-PLAN.md]* GaussianNB only. KMeans + LinearRegression.partial_fit deferred to Phase 5+.
2. **Memory threshold mechanism: file-size probe or JVM-heap probe?** *[RESOLVED → 04-02-PLAN.md, inline comment in `read_csv_streaming`]* No probe at the API surface; chunked streaming handles any file size regardless. The "should I chunk?" decision lives in caller code (see `PandasPipeline` builder). Heap-only used for default `chunkRows` calculation inside `ChunkedReadOptions`.
3. **Pipeline helper: builder or static factory?** *[RESOLVED → 04-03-PLAN.md]* Fluent builder (`PandasPipeline`). Static factory rejected as adding a 2nd caller pattern.
4. **`index_col` support in streaming reader?** *[RESOLVED → 04-01-PLAN.md]* NEW class `ChunkedReadOptions` with streaming subset (sep/header/skiprows/index_col/chunkRows). Existing `ReadCsvOptions` unchanged.
5. **CSV injection mitigation (ASVS V5)?** *[RESOLVED → 04-01-PLAN.md]* Mitigation-ready in `CsvChunkReader` (leading-char NaN substitution) but no test corpus exercises it. Documented as deferred to v2 if user request; javadoc covers the behavior.
6. **Should in-memory `read_csv` get the same defensive identity init for `lines.isEmpty()` guard, or is it already correct?** *[RESOLVED → 04-01-PLAN.md]* Streaming reader mirrors existing guard exactly; `CsvChunkReader` throws `IllegalArgumentException` on empty file in the same spot.

## Environment Availability

> Phase 4 has external dependencies: Maven build + JDK 17 + the test corpus CSVs at `dist/datasets/`. All confirmed available per STATE.md and ROADMAP.md Phase 1 baseline.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Maven 3.9.15 | Build | ✓ (PATH prefix required per STATE.md) | 3.9.15 | — |
| JDK Temurin 25.0.3 (compile `--release 17`) | Compile + run | ✓ | 25.0.3 | — |
| `dist/datasets/california_housing.csv` | read_csv test corpus | ✓ (per Phase 1 setup) | — | — |
| `dist/datasets/titanic.csv` | groupby test corpus | ✓ (per Phase 1 setup) | — | — |
| `dist/datasets/iris.csv` | SklearnBench test corpus | ✓ (per Phase 1 setup) | — | — |
| Synthetic >RAM CSV for OOM simulation | MEM-02 success criterion | ✗ (not pre-existing) | — | Generate at test-time with `Random` seeded write-loop, `Files.write(tempPath, ...)` in `@Before`; clean up `@After`. CSV up to ~500 MB; heap budget `-Xmx128m` triggers OOM on naive path. |
| `scripts/check_regression.ps1` | BENCH-03 regression gate | ✓ (per Phase 3) | — | — |

**Missing dependencies with no fallback:**
- None — every external dependency is available.

**Missing dependencies with fallback:**
- Synthetic large CSV: generated at test time (no fallback needed; generation is the fallback).

## Validation Architecture

> `workflow.nyquist_validation` is enabled (true in `.planning/config.json`). Test infrastructure exists; Phase 4 must extend it.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | JUnit 4.13.2 (per Phase 1/2/3 lock — NOT JUnit 5) |
| Config file | `pom.xml` per module (no surefire overrides needed) |
| Quick run command | `mvn -pl modules/pandas -am test "-Dtest=PandasStreamingTest" "-Dsurefire.failIfNoSpecifiedTests=false"` |
| Full suite command | `mvn test` from repo root |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MEM-01 | Small file → `read_csv` returns DataFrame unchanged | unit (cross-path equivalence) | `mvn -pl modules/pandas -am test -Dtest=PandasStreamingTest#smallFile_inMemoryPathUnchanged` | ❌ Wave 0 |
| MEM-01 | `Pandas.read_csv(path)` signature byte-identical | unit (API surface) | `grep -c "public static DataFrame read_csv" Pandas.java` returns ≥ pre-Phase-4 count | n/a (spot-check) |
| MEM-02 | Large file → `read_csv_streaming` returns chunk iterator, fits in small heap | integration | `mvn -pl modules/pandas -am test -Dtest=PandasStreamingTest#largeFile_streamingFitsInSmallHeap -DargLine="-Xmx128m"` | ❌ Wave 0 |
| MEM-02 | Stream emits N chunks where N = ceil(rows / chunkRows) | unit | `mvn -pl modules/pandas -am test -Dtest=CsvChunkReaderTest#chunkCountMatchesCeil` | ❌ Wave 0 |
| MEM-03 | Streaming groupby sum/mean matches in-memory on same small dataset within `1e-9 * num_groups` | unit (cross-path equivalence) | `mvn -pl modules/pandas -am test -Dtest=RunningGroupAggregatorTest#sum_matchesInMemory` | ❌ Wave 0 |
| MEM-03 | GaussianNB `partial_fit(X1,y1)` + `partial_fit(X2,y2)` ≈ `fit(X_full, y_full)` | unit (cross-path equivalence) | `mvn -pl modules/sklearn -am test -Dtest=GaussianNBPartialFitTest#twoPartialFits_equalOneFit` | ❌ Wave 0 |
| USE-02 | `<10` line caller pattern works end-to-end | integration | `mvn -pl modules/sklearn -am test -Dtest=PandasPipelineTest#tenLineCallerPattern` | ❌ Wave 0 |
| USE-02 | Builder rejects missing `load()` path with clear error | unit (input validation) | `mvn -pl modules/sklearn -am test -Dtest=PandasPipelineTest#missingPath_throwsIAE` | ❌ Wave 0 |
| STRIDE-T401 | Path traversal mitigation (canonicalize path before open) | unit | `mvn -pl modules/pandas -am test -Dtest=PandasStreamingTest#pathTraversal_canonicalized` | ❌ Wave 0 |
| STRIDE-T402 | CSV injection mitigation (leading `=+-@` cells → NaN/logged) | unit | `mvn -pl modules/pandas -am test -Dtest=PandasStreamingTest#csvFormulaCells_neutralized` | ❌ Wave 0 |
| STRIDE-T403 | Streaming reader closes underlying file handle on early break | unit (resource hygiene) | `mvn -pl modules/pandas -am test -Dtest=CsvChunkReaderTest#close_releasesFileHandle` | ❌ Wave 0 |
| API-1 | `Pandas.java` public static count ≥ pre-Phase-4 | spot-check | `grep -c "public static" Pandas.java` ≥ previous count | n/a |
| API-2 | `NumJa.java` public static count unchanged at 61 | spot-check | `grep -c "public static" NumJa.java` = 61 | n/a |
| API-3 | `ArrayOps.java` public static count unchanged at 34 | spot-check | `grep -c "public static" ArrayOps.java` = 34 | n/a |
| REGRESSION | `mvn test` exit 0 with full suite | integration | `mvn test` | n/a (suite) |
| REGRESSION | Phase 3 regression gate still passes | integration | `powershell -File scripts/check_regression.ps1` | n/a (carry-forward) |

### Sampling Rate

- **Per task commit:** `mvn -pl modules/pandas -am test -Dtest=PandasStreamingTest` (fast feedback for streaming-specific changes); `mvn -pl modules/sklearn -am test -Dtest=GaussianNBPartialFitTest,PandasPipelineTest` (model changes).
- **Per wave merge:** `mvn test` from repo root (full suite, ensure no regression in existing 55 tests).
- **Phase gate:** Full suite green + Phase 3 regression gate still passes + cross-path equivalence verified before `/gsd:verify-work`.

### Wave 0 Gaps

- [ ] `modules/pandas/src/main/java/pandas/internal/ChunkedReadOptions.java` — NEW
- [ ] `modules/pandas/src/main/java/pandas/internal/CsvChunkReader.java` — NEW
- [ ] `modules/pandas/src/main/java/pandas/internal/RunningGroupAggregator.java` — NEW
- [ ] `modules/pandas/src/test/java/com/pandas/internal/CsvChunkReaderTest.java` — NEW (≥6 tests)
- [ ] `modules/pandas/src/test/java/com/pandas/internal/RunningGroupAggregatorTest.java` — NEW (≥6 tests)
- [ ] `modules/pandas/src/test/java/com/pandas/PandasStreamingTest.java` — NEW (≥4 tests including cross-path)
- [ ] `modules/sklearn/src/test/java/sklearn/naive_bayes/GaussianNBPartialFitTest.java` — NEW (≥4 tests)
- [ ] `modules/sklearn/src/test/java/sklearn/pipeline/PandasPipelineTest.java` — NEW (≥3 tests)
- [ ] `modules/pandas/src/main/java/pandas/Pandas.java` — +1 public method `read_csv_streaming` (additive, not modifying existing signatures)
- [ ] `modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java` — +1 public method `partial_fit` + private `init()` refactor
- [ ] `modules/sklearn/src/main/java/sklearn/pipeline/PandasPipeline.java` — NEW public class
- [ ] Framework install: none — JUnit 4.13.2 + Mockito (if needed for `Estimator` interface) already vendored via parent pom.

(If all gaps filled: framework infra complete; implementation can begin.)

## Security Domain

> `security_enforcement` defaults to enabled (absent from `.planning/config.json` = enabled). Required for Phase 4.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V1 Architecture | yes | Document STRIDE threats in PLAN.md (mandated by project constraint) |
| V2 Authentication | no | Phase 4 has no auth boundary |
| V3 Session Management | no | No sessions |
| V4 Access Control | yes (file system) | `path.normalize()` + `Files.isReadable(path)` before open; reject paths outside an optional `allowedRoot` if configured (default = no restriction, mirrors existing `read_csv`) |
| V5 Input Validation | yes | Type-check numeric cells (existing `Double.parseDouble` → NaN); reject if `index_col` out of range; chunkRows ≥ 1; chunkRows ≤ hard ceiling |
| V6 Cryptography | no | No crypto in scope |
| V7 Error Handling | yes | `UncheckedIOException` wraps `IOException`; `IllegalArgumentException` for malformed options (existing style) |
| V8 Data Protection | no | No PII handling; numeric CSV only |
| V9 Communication | no | No network |
| V10 Malicious Code | no | No code execution paths from CSV content (formula injection is mitigated by NaN-substitution) |
| V11 Business Logic | yes | Cross-path equivalence invariant: streaming result == in-memory result within `1e-9 * num_groups` tolerance |
| V12 Files and Resources | yes | `try-with-resources` on `BufferedReader`; chunk size cap prevents OOM |
| V13 API and Web Service | no | Library, not service |
| V14 Configuration | no | No new runtime config in Phase 4 |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal (`../../etc/passwd`) | Elevation of Privilege | `path.normalize()` + reject if path contains `..` after normalization (defensive; not a sandbox) |
| CSV formula injection (`=cmd|...`) | Tampering | Substitute leading `=`, `+`, `-`, `@` cells with NaN; document in javadoc. **Defer to v2 if test corpus is numeric-only.** |
| Memory exhaustion (pathological file size) | Denial of Service | `chunkRows` cap + `Runtime.getRuntime().maxMemory()` budget; reject `chunkRows > ceiling` in `ChunkedReadOptions` validation |
| File handle leak (early iterator break) | Denial of Service | `AutoCloseable` on `CsvChunkReader`; `try-with-resources` documented in javadoc |
| NaN injection via malformed numeric cell | Tampering | Existing `Double.parseDouble` → NaN behavior preserved |
| Index out of bounds (misconfigured `index_col`) | Denial of Service | Validate `index_col` in `CsvChunkReader` ctor post-header-read |
| Stale aggregation state across `partial_fit` calls | Tampering | `GaussianNB.initialized` flag; reset on `new`, accumulate thereafter; document in javadoc |
| Iterator reuse after exhaustion | Information Disclosure | Throw `NoSuchElementException` (already in pattern); do not return stale chunks |

### Documented Mitigations for PLAN.md

Each PLAN.md MUST contain an ASVS L1 + STRIDE block with:
- Trust boundary: file system (caller-controlled path) → in-process `DataFrame` / `NDArray`
- Threats: STRIDE table above
- Mitigations: each mitigation referenced by line in the eventual code
- Tests: STRIDE-T401/T402/T403 in the test map above MUST PASS

## Codebase Reconnaissance Summary

| Location | File:Lines | What it does | Phase 4 impact |
|----------|------------|--------------|----------------|
| Pandas CSV reader | `modules/pandas/src/main/java/pandas/Pandas.java:30-112` | BufferedReader over path → List<String[]> → double[][] | Existing method UNCHANGED; new sibling `read_csv_streaming` |
| GroupBy eager builder | `modules/pandas/src/main/java/pandas/GroupBy.java:14-24` | Per-key List<Integer> map | Existing UNCHANGED; new `streamingAggregate` static helper in `internal/RunningGroupAggregator` |
| GroupBy aggregate switch | `modules/pandas/src/main/java/pandas/GroupBy.java:71-85` | sum/mean/count/min/max/std per group | Mirror the agg list in `RunningAgg` |
| DataFrameStats.describe | `modules/pandas/src/main/java/pandas/DataFrameStats.java:9-53` | Per-column sum/min/max/mean/std | Streaming `describe` derives from per-column `RunningAgg` (out of scope for v1; deferred) |
| Series sum/mean/min/max | `modules/pandas/src/main/java/pandas/Series.java:68-103` | NaN-skipping sequential loops | UNCHANGED; the `RunningAgg` mirrors the NaN-skip policy |
| DataFrame constructors | `modules/pandas/src/main/java/pandas/DataFrame.java:18-54` | 3 constructors (double[][], Map<Series>, …) | Reuse `(Map<String, Series>)` ctor for chunk DataFrame assembly |
| GaussianNB.fit | `modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java:18-47` | Per-class running accumulator | `partial_fit` extracts the accumulator pattern into a shared private method |
| KMeans.fit | `modules/sklearn/src/main/java/sklearn/cluster/KMeans.java:46-85` | Lloyd's algorithm; full X in memory | **DEFERRED** (not in Phase 4 scope per open question Q1) |
| LinearRegression.fit | `modules/sklearn/src/main/java/sklearn/linear_model/LinearRegression.java:20-59` | Normal equations; full X in memory | **DEFERRED** (not in Phase 4 scope per open question Q1) |
| Pipeline (existing) | `modules/sklearn/src/main/java/sklearn/pipeline/Pipeline.java:12-85` | Reflective `fit_transform` chain over NDArray | UNCHANGED — different concern (in-memory vs streaming) |
| PandasBench | `bench/src/main/java/bench/PandasBench.java:21-46` | JMH benchmarks for read_csv + groupby | Add streaming benchmark (e.g., `read_csv_streaming_california`); reuse same datasets |
| ParallelOps threshold gate | `modules/numja/src/main/java/numja/core/ParallelOps.java:19,43-46` | `THRESHOLD = 100_000`, `gate()` test override | NOT USED in Phase 4 (out-of-core is at the chunk level, not the row-op level) |
| ThreadPoolConfig FJP singleton | `modules/numja/src/main/java/numja/config/ThreadPoolConfig.java:84-96` | Double-checked locking FJP | Optional — could parallelize per-chunk Series extraction, but adds complexity; v1 is single-threaded per chunk; v2 may parallelize |
| Vendored libs | `libs/`: ejml-core-0.43.1.jar, ejml-ddense-0.43.1.jar, commons-math3-3.6.1.jar, jfreechart-1.5.3.jar, proguard-7.3.2 | No CSV parser vendored | Stdlib `BufferedReader` is sufficient |

**Nothing else in the codebase mentions streaming / partial_fit / chunk / BufferedReader** outside Pandas.read_csv (grep `partial_fit|stream|chunk|BufferedReader|Iterator` returned only Pandas.java + test files in numja/sklearn that grep-matched `stream` for unrelated reasons like `"StreamingContext"` references — none exist in source).

## Sources

### Primary (HIGH confidence — codebase grep + file reads)

- `modules/pandas/src/main/java/pandas/Pandas.java` (read full) — existing CSV reader
- `modules/pandas/src/main/java/pandas/GroupBy.java` (read full) — existing groupby
- `modules/pandas/src/main/java/pandas/DataFrame.java` (read full) — DataFrame constructors
- `modules/pandas/src/main/java/pandas/DataFrameStats.java` (read full) — describe
- `modules/pandas/src/main/java/pandas/Series.java` (read full) — Series methods
- `modules/sklearn/src/main/java/sklearn/naive_bayes/GaussianNB.java` (read full) — partial_fit target
- `modules/sklearn/src/main/java/sklearn/cluster/KMeans.java` (read full) — partial_fit candidate (deferred)
- `modules/sklearn/src/main/java/sklearn/linear_model/LinearRegression.java` (read full) — partial_fit candidate (deferred)
- `modules/sklearn/src/main/java/sklearn/pipeline/Pipeline.java` (read full) — existing reflective Pipeline
- `modules/sklearn/src/main/java/sklearn/utils/ParallelUtils.java` (read partial) — sklearn's own thread helper
- `modules/numja/src/main/java/numja/NumJa.java` (read full) — public API surface (61)
- `modules/numja/src/main/java/numja/core/ArrayOps.java` (read partial) — public API surface (34)
- `modules/numja/src/main/java/numja/core/ParallelOps.java` (read full) — Phase 2/3 threshold-gate + Kahan pattern
- `modules/numja/src/main/java/numja/core/NDArray.java` (read partial lines 1-200, 395-500) — threshold-gated sum/min/max/prod
- `bench/src/main/java/bench/PandasBench.java` (read full) — JMH pandas benchmark
- `bench/src/main/java/bench/SklearnBench.java` (read full) — JMH sklearn benchmark
- `modules/pandas/pom.xml` (read full) — no deps
- `modules/sklearn/pom.xml` (read full) — numja-core + pandas-wrapper + junit + org.json
- `.planning/STATE.md` (read full) — frozen API lock, branch=dev, JDK 25.0.3 default / `--release 17`
- `.planning/ROADMAP.md` (read full) — Phase 4 scope + success criteria
- `.planning/REQUIREMENTS.md` (read full) — MEM-01/02/03, USE-02 traceability
- `.planning/research/SUMMARY.md` (read full) — confirms no JVM out-of-core DataFrame library mature; stdlib self-rolled chunk iterator recommended
- `.planning/phases/02-cpu-parallel-core-threads/02-BASELINE-AFTER.md` (read full) — Phase 2 carry-forward: threshold-gate pattern, FJP singleton, 5 WR review fixes
- `.planning/phases/02-cpu-parallel-core-threads/02-VERIFICATION.md` (read full) — Phase 2 must-have coverage pattern
- `.planning/phases/03-numerical-accuracy-hardening/03-VERIFICATION.md` (read full) — Phase 3 must-have coverage pattern + golden test pattern
- `.planning/phases/03-numerical-accuracy-hardening/03-BASELINE-AFTER.md` (read full) — Phase 3 baseline + regression-gate methodology

### Secondary (MEDIUM confidence — inferred from existing patterns + JDK docs)

- JDK 17 `Files.newBufferedReader(Path, Charset)` + `BufferedReader.readLine()` semantics — verified via existing `Pandas.java:37` usage
- `Iterator<T>` + `AutoCloseable` pattern — verified via JDK 17 javadoc (not re-fetched; stable since JDK 8)

### Tertiary (LOW confidence — flagged for validation)

- None — every concrete recommendation is backed by either existing code patterns or stdlib API known to be stable.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — stdlib only; zero new dependencies; `BufferedReader` already used in same module.
- Architecture: HIGH — pattern mirrors existing `GroupBy` + `GaussianNB` accumulator code; carry-forward patterns from Phase 2/3 well-documented.
- Pitfalls: MEDIUM-HIGH — 6 named pitfalls with code-level mitigations and warning signs; CSV-injection mitigation marked optional (A6).
- Test strategy: HIGH — `mvn test` infra works; -Xmx flag for OOM simulation is well-supported on Surefire.
- API surface: HIGH — locked constraint from STATE.md; only +1 method per class, no existing signature touched.

**Research date:** 2026-08-28
**Valid until:** 2026-10-28 (60 days — stdlib + frozen API; nothing in this domain moves quickly)

---

## RESEARCH COMPLETE

Phase 4 plan: stdlib-only `Pandas.read_csv_streaming` + running-aggregator groupby + `GaussianNB.partial_fit` + `PandasPipeline` builder; defer KMeans/LinearRegression.partial_fit to Phase 5+.
