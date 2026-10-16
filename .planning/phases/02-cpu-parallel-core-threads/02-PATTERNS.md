# Phase 2: CPU Parallel Core (Threads) - Pattern Map

**Mapped:** 2026-08-27
**Files analyzed:** 8 (3 NEW source, 3 NEW test, 2 MODIFY, 1 NEW doc)
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `modules/numja/src/main/java/numja/core/ParallelOps.java` (NEW) | utility | elementwise/reduce via FJP | `numja/core/NDArray.java` (sequential loops) + `sklearn/utils/ParallelUtils.java` (anti-pattern) | role+anti-pattern |
| `modules/numja/src/main/java/numja/config/ThreadPoolConfig.java` (MODIFY) | config | singleton accessor | self (existing `getInstance()` lines 28-33) | exact |
| `modules/numja/src/main/java/numja/core/NDArray.java` (MODIFY) | core | elementwise/reduce | self (`add` line 151-160, `sum` line 329-335, `min` line 347-355) | exact |
| `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java` (NEW) | test | unit | `modules/numja/src/test/java/com/numja/core/NDArrayTest.java` | exact |
| `modules/numja/src/test/java/com/numja/core/ParallelElementwiseTest.java` (NEW) | test | unit | `modules/numja/src/test/java/com/numja/core/NDArrayTest.java` | exact |
| `modules/numja/src/test/java/com/numja/core/ParallelRegressionTest.java` (NEW) | test | timing/regression | `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java` (soft-fail harness) + `NDArrayTest` (JUnit 4 basics) | role-match |
| `bench/src/main/java/bench/CoreBench.java` (MODIFY) | bench | JMH | self (`ElemState` lines 64-77 + `add_elementwise` lines 79-82) | exact |
| `.planning/phases/02-cpu-parallel-core-threads/02-BASELINE-AFTER.md` (NEW) | doc | reference | `.planning/phases/01-baseline-benchmark/BASELINE.md` | exact |

> **Conventions note (test package path):** main sources use package `numja.core`. Existing tests already live under `com/numja/core/` (see `modules/numja/src/test/java/com/numja/core/NDArrayTest.java:1` declaring `package numja.core;` from path `com/numja/core/` — this is a long-standing path/package mismatch in the repo). New tests should follow the existing test-side path: `modules/numja/src/test/java/com/numja/core/` (the executor's package declaration stays `package numja.core;`). Don't introduce a new `numja/` test root.

## Pattern Assignments

### `modules/numja/src/main/java/numja/core/ParallelOps.java` (NEW — utility)

**Analogs:** `NDArray.java` (sequential elementwise/reduce body to keep below threshold) + `ThreadPoolConfig.java` (singleton-style lazy init) + `ParallelUtils.java` (anti-pattern to AVOID — per-call pool).

**Package + imports** (mirrors `NDArray.java:1-6`):
```java
package numja.core;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
// NDArray/ThreadPoolConfig live in same module — no cross-module import needed
```

**Singleton-ish threshold constants + dispatcher** (mirrors the lazy-init spirit of `ThreadPoolConfig.java:28-33`; do NOT mirror the per-call pool of `ParallelUtils.java:55-69`):
```java
// Threshold matching the success criterion: <100k stays sequential.
public static final int THRESHOLD = 100_000;

// Leaf CUTOFF inside RecursiveAction — RESEARCH §Open Questions #1 suggests 16k
// doubles ≈ 128KB ≈ L1 cache; executor should expose as a constant for sweep.
static final int LEAF_CUTOFF = 16_384;

public static void elementwiseBinary(double[] a, double[] b, double[] out,
                                     DoubleBinaryOperator op) {
    final int n = a.length;
    if (n < THRESHOLD) {                       // gate, like NDArray.add line 157-158
        for (int i = 0; i < n; i++) out[i] = op.applyAsDouble(a[i], b[i]);
        return;
    }
    ForkJoinPool pool = ThreadPoolConfig.getInstance().getForkJoinPool();
    pool.invoke(new ElementwiseTask(a, b, out, op, 0, n));
}
```

**RecursiveAction skeleton** (analog: none — first FJP usage in repo; pattern lifted from JDK `ForkJoinPool` docs):
```java
private static final class ElementwiseTask extends RecursiveAction {
    private final double[] a, b, out;
    private final DoubleBinaryOperator op;
    private final int lo, hi;
    ElementwiseTask(double[] a, double[] b, double[] out,
                    DoubleBinaryOperator op, int lo, int hi) {
        this.a = a; this.b = b; this.out = out; this.op = op; this.lo = lo; this.hi = hi;
    }
    @Override protected void compute() {
        if (hi - lo <= LEAF_CUTOFF) {
            for (int i = lo; i < hi; i++) out[i] = op.applyAsDouble(a[i], b[i]);
            return;
        }
        int mid = (lo + hi) >>> 1;             // avoid (lo+hi)/2 overflow
        invokeAll(new ElementwiseTask(a, b, out, op, lo, mid),
                  new ElementwiseTask(a, b, out, op, mid, hi));
    }
}
```

**Sequential loop body to mirror** (`NDArray.java:329-335` for sum, `NDArray.java:347-355` for min, `NDArray.java:360-368` for max) — `ParallelOps` should reuse the exact same loop as the sub-threshold branch so both paths produce bit-identical results below threshold:
```java
// from NDArray.java:329-335 — reuse verbatim in ParallelOps.sumSequential
double sum = 0;
for (int i = 0; i < data.numRows * data.numCols; i++) {
    sum += data.data[i];
}
return sum;
```

**Public surface:** package-private static methods only (`NDArray` calls them). RESEARCH §Pitfall 7 explicitly forbids adding `public static addParallel(...)` to `ArrayOps` — keep new entry point hidden.

**Error handling:** No checked exceptions; `IllegalArgumentException` for shape/length mismatch happens upstream in `NDArray.add` (line 152-155). `ParallelOps` trusts its inputs.

---

### `modules/numja/src/main/java/numja/config/ThreadPoolConfig.java` (MODIFY — add `getForkJoinPool()`)

**Analog:** self, specifically the existing `getInstance()` at lines 28-33 (lazy singleton + `synchronized`). The new accessor MUST follow the same lazy-init pattern so the FJP is created exactly once and shared.

**Insertion point:** after `get_threads()` line 73-75, before `setThreads`.

**Pattern to copy** (lines 28-33):
```java
public static synchronized ThreadPoolConfig getInstance() {
    if (instance == null) {
        instance = new ThreadPoolConfig();
    }
    return instance;
}
```

**New field + accessor** (lazy FJP, sized to the already-60%-capped `currentThreads`):
```java
// Field added at top of class alongside `private boolean autoOptimize;`
private volatile ForkJoinPool forkJoinPool;          // volatile + lazy = safe publication

public ForkJoinPool getForkJoinPool() {
    ForkJoinPool p = forkJoinPool;
    if (p == null) {
        synchronized (this) {
            p = forkJoinPool;
            if (p == null) {
                p = new ForkJoinPool(currentThreads);   // RESEARCH §Pattern 3
                forkJoinPool = p;
            }
        }
    }
    return p;
}
```

**Add import** at top of file (lines 1-4 already import `java.util.*`-adjacent types):
```java
import java.util.concurrent.ForkJoinPool;
```

**Limitation comment** (matches RESEARCH §Open Questions #4):
```java
// NOTE: setThreads() mutates currentThreads but does NOT recreate the pool.
// Matches existing set_threads behavior (mutates integer only).
// If resize-after-create is needed, add a resetForkJoinPool() and call it from setThreads.
```

---

### `modules/numja/src/main/java/numja/core/NDArray.java` (MODIFY — gate elementwise + reduce)

**Analog:** self. The change is a thin wrapper around the existing bodies — preserve the original loop as the sub-threshold branch (RESEARCH §Pattern 1) so behaviour for n < 100k is bit-identical to v0.1.0.

**Imports to add** (top of file, lines 1-6):
```java
import numja.config.ThreadPoolConfig;     // not yet imported in NDArray.java
import numja.core.ParallelOps;            // same package — import still required
```

**Method body to wrap** — `add` (line 151-160):
```java
public NDArray add(NDArray other) {
    if (!Arrays.equals(shape, other.shape)) {                       // unchanged
        throw new IllegalArgumentException("Shape mismatch: " +
            Arrays.toString(shape) + " vs " + Arrays.toString(other.shape));
    }
    DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
    int n = data.numRows * data.numCols;
    if (n < ParallelOps.THRESHOLD) {
        CommonOps_DDRM.add(data, other.data, result);               // keep existing path
    } else {
        ParallelOps.elementwiseBinary(data.data, other.data, result.data, Double::sum);
    }
    return new NDArray(result, shape.clone());
}
```

**Same wrapping recipe applies to** `subtract` (177-185), `multiply` (197-205), `divide` (219-229), `power` (241-247), `abs` (252-258), `sqrt` (263-269), `exp` (274-280), `log` (285-291), `sin` (296-302), `cos` (307-313), `tan` (318-324), `add(double)` (165-172 scalar).

**Reduce methods to wrap** — `sum` (line 329-335):
```java
public double sum() {
    int n = data.numRows * data.numCols;
    if (n < ParallelOps.THRESHOLD) {
        double s = 0;
        for (int i = 0; i < n; i++) s += data.data[i];             // unchanged
        return s;
    }
    return ParallelOps.sum(data.data);
}
```

**Same wrapping for** `min` (347-355), `max` (360-368). `mean` (340-342) already delegates to `sum()` — no change needed.

**Conventions to preserve** (from existing file):
- Javadoc style: `/** one-line description */` above each public method (e.g., line 148-150 for `add`). Add a one-liner to new `ParallelOps` methods too.
- Throw `IllegalArgumentException` for shape/length mismatch (line 153) — do NOT introduce a new exception type.
- Result wrapper: `new NDArray(result, shape.clone())` (line 159) — keep the `shape.clone()` to defend against caller mutation.

---

### `modules/numja/src/test/java/com/numja/core/ParallelOpsTest.java` (NEW)

**Analog:** `modules/numja/src/test/java/com/numja/core/NDArrayTest.java` (whole file, 25 lines) — JUnit 4, simple `assertEquals` with delta.

**Style to mirror** (`NDArrayTest.java:1-23`):
```java
package numja.core;

import numja.core.NDArray;
import numja.core.ParallelOps;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ParallelOpsTest {
    @Test
    public void elementwiseBinary_matchesSequential() {
        NDArray a = new NDArray(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        NDArray b = new NDArray(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        // exercise both sub-threshold (n<8) and a synthetic above-threshold case
        // via direct ParallelOps call with a double[] of length 200_000+
        // ...
        assertEquals(9.0, result.getData().get(0,0), 1e-9);
    }
}
```

**Conventions:**
- JUnit 4 (`org.junit.Test`, `org.junit.Assert.*`) per `modules/numja/pom.xml:35-37`.
- Delta `1e-9` matches existing test style (NDArrayTest.java:15-22).
- Path `com/numja/core/` (not `numja/core/`) matches existing test layout.

---

### `modules/numja/src/test/java/com/numja/core/ParallelElementwiseTest.java` (NEW)

**Analog:** `NDArrayTest.java` (whole file) — same JUnit 4 + delta style.

**Pattern to copy** (NDArrayTest.java:11-22):
```java
NDArray a = new NDArray(1.0, 2.0, 3.0);
NDArray b = new NDArray(4.0, 5.0, 6.0);

NDArray sum = a.add(b);
assertEquals(5.0, sum.getData().get(0,0), 1e-9);
```

**Scope:** exercise `NumJa.add` / `NumJa.multiply` / `NumJa.divide` / `NumJa.exp` against golden values at both n < THRESHOLD (sub-100k) and n > THRESHOLD (e.g., 1_000_000 doubles — fits in 8MB, well under CI heap). Reuse seed `44L`/`45L` from `CoreBench.ElemState` (lines 74-75) so Phase 2 numbers are reproducible.

**Critical assertion:** at n > THRESHOLD, `parallel_path` and `sequential_path` must agree within `1e-13` relative error (RESEARCH §Pitfall 5, matches `GoldenReferenceTest` tolerance per `STATE.md`).

---

### `modules/numja/src/test/java/com/numja/core/ParallelRegressionTest.java` (NEW)

**Analog A (timing harness):** `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java:38-60` — JUnit 4 + `@Category` marker + soft-fail accumulation pattern. We can adopt the soft-fail style so CI noise doesn't block merge, OR we can hard-fail since this is a hard success criterion.

**Analog B (JUnit basics):** `NDArrayTest.java` (whole file).

**Skeleton** (combines both):
```java
package numja.core;

import numja.NumJa;
import numja.core.NDArray;
import org.junit.Test;
import static org.junit.Assert.*;

public class ParallelRegressionTest {

    private static NDArray randomVec(int n, long seed) {     // copy from CoreBench.java:32-37
        java.util.Random rng = new java.util.Random(seed);
        double[] a1 = new double[n];
        for (int i = 0; i < n; i++) a1[i] = rng.nextDouble();
        return new NDArray(a1);
    }

    @Test
    public void largeArray_isAtLeast2xFaster() {
        NDArray a = randomVec(10_000_000, 44L);
        NDArray b = randomVec(10_000_000, 45L);
        // warm up JIT
        for (int i = 0; i < 5; i++) NumJa.add(a, b);
        // time sequential path (force THRESHOLD = MAX via reflection or env)
        // time parallel path (default)
        // assert ratio >= 2.0  (RESEARCH success criterion #1)
    }

    @Test
    public void smallArray_under10PercentRegression() {
        // n = 10_000 and 100_000; assert parallel/sequential <= 1.10
    }
}
```

**Test forcing sequential path:** add a package-private setter on `ParallelOps` (`setThresholdForTesting(int)`) or use a system property `numja.parallel.threshold` — must be read once at class init, not per-call, to avoid benchmarking overhead.

**Conventions:**
- JUnit 4 (same as other numja tests).
- Delta `1e-13` for value comparisons (matches `GoldenReferenceTest` per STATE.md).
- No `@Category` needed unless we want to filter — keep it simple.
- Path `com/numja/core/`.

---

### `bench/src/main/java/bench/CoreBench.java` (MODIFY — add `SmallArrayState` + `add_elementwise_small`)

**Analog:** self — copy `ElemState` (lines 64-77) and the existing `add_elementwise` benchmark (lines 79-82) one-for-one, just change the `@Param` values.

**Insertion point:** after line 87 (end of `multiply_elementwise`), before the reduce section.

**State class to add** (mirror `ElemState` lines 64-77):
```java
@State(Scope.Thread)
public static class SmallArrayState {
    @Param({"10000", "100000"})                  // RESEARCH success criterion #2
    public int n;

    NDArray a;
    NDArray b;

    @Setup(Level.Trial)
    public void setUp() {
        a = randomVec(n, 47L);                    // different seeds to avoid cache pollution
        b = randomVec(n, 48L);
    }
}

@Benchmark
public void add_elementwise_small(SmallArrayState s, Blackhole bh) {
    bh.consume(NumJa.add(s.a, s.b));
}
```

**Conventions to preserve** (existing file):
- `@BenchmarkMode(Mode.AverageTime)` + `@OutputTimeUnit(MILLISECONDS)` + `@Warmup(5,1)` + `@Measurement(5,1)` + `@Fork(1)` at class level (lines 17-21). Do NOT modify — RESEARCH §Pitfall 3 says current forks=1 is acceptable here; bumping is out of scope unless we re-baseline.
- `randomVec` helper is `private static` (line 32). Reuse, don't re-declare.
- Seeds are `44L`/`45L` for large ElemState (line 74-75); use `47L`/`48L` for SmallArrayState to keep baseline-vs-Phase-2 reproducible AND distinguishable.

**Optional follow-on (only if RESEARCH §Pitfall 3 demands it):** change `@Fork(1)` to `@Fork(3)` — this affects EVERY existing benchmark and would force re-baseline of `01-BASELINE.md`. Out of scope unless explicitly requested.

---

### `.planning/phases/02-cpu-parallel-core-threads/02-BASELINE-AFTER.md` (NEW)

**Analog:** `.planning/phases/01-baseline-benchmark/BASELINE.md` (whole file, 72 lines) — same structure, same machine-metadata table, same result table format.

**Sections to copy verbatim from `BASELINE.md`:**
- Machine metadata table (lines 12-21) — same machine.
- "Kết quả" result table format (lines 23-27) — Mode `avgt`, ms/op, Error = 99.9% CI.

**Sections to ADD:**
- New "Core ops (parallel path)" subsection showing elementwise/reduce at n=10⁶ and n=10⁷ under Phase 2.
- "Small-array regression" subsection from `add_elementwise_small` at n=10⁴ / n=10⁵ — assert ≤10% regression vs Phase 1 baseline (89.38ms at 10⁷ → must be ≥2x faster, so ≤44ms; at 10⁵ must be ≤1.10× of sub-100k sequential path).
- "Methodology" subsection noting hybrid P/E-core variance (RESEARCH §Pitfall 3) — same power profile, ≥3 forks if re-baselined.

**Reproduce command** (copy from `BASELINE.md:64-68`):
```powershell
$env:Path = 'C:\Users\Admin\.maven\maven-3.9.15\bin;' + $env:Path
mvn -pl bench -am package -DskipTests
java -jar bench/target/benchmarks.jar -f 1 -wi 3 -i 3 -w 1s -r 1s
java -jar bench/target/benchmarks.jar "CoreBench.add_elementwise_small" -p n=10000
```

## Shared Patterns

### Lazy singleton (apply to: `ThreadPoolConfig.getForkJoinPool()`)

**Source:** `ThreadPoolConfig.java:28-33`
```java
public static synchronized ThreadPoolConfig getInstance() {
    if (instance == null) {
        instance = new ThreadPoolConfig();
    }
    return instance;
}
```
**Apply to:** new `getForkJoinPool()` — same lazy + synchronized shape; one FJP for the JVM lifetime. Don't create a new FJP per call (RESEARCH §Pattern 3 anti-pattern note).

### Threshold-gated dispatch (apply to: every modified `NDArray` method)

**Source:** `NDArray.java:151-160` (current `add`)
```java
DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
int n = data.numRows * data.numCols;
// OLD: CommonOps_DDRM.add(data, other.data, result);
// NEW:
if (n < ParallelOps.THRESHOLD) {
    CommonOps_DDRM.add(data, other.data, result);
} else {
    ParallelOps.elementwiseBinary(data.data, other.data, result.data, Double::sum);
}
return new NDArray(result, shape.clone());
```
**Apply to:** add, subtract, multiply, divide, power, abs, sqrt, exp, log, sin, cos, tan, sum, min, max. Scalar overloads (e.g., `add(double)`, `multiply(double)`) also gate on `n`.

### Sequential loop as sub-threshold branch (apply to: `ParallelOps.elementwiseBinary`, `sum`, `min`, `max`)

**Source:** `NDArray.java:329-335` (sum), `NDArray.java:347-355` (min), `NDArray.java:360-368` (max)
```java
double s = 0;
for (int i = 0; i < data.numRows * data.numCols; i++) {
    s += data.data[i];
}
return s;
```
**Apply to:** keep these exact loops as the sub-threshold branch in both `NDArray` AND as `ParallelOps.sumSequential` so both paths produce bit-identical output for n < THRESHOLD (otherwise existing tests like `NDArrayTest.arithmeticAndReductions` at line 21 may flip).

### Error handling (apply to: `ParallelOps`, modified `NDArray`)

**Source:** `NDArray.java:152-155`
```java
if (!Arrays.equals(shape, other.shape)) {
    throw new IllegalArgumentException("Shape mismatch: " +
        Arrays.toString(shape) + " vs " + Arrays.toString(other.shape));
}
```
**Apply to:** shape/length checks at the `NDArray` API boundary (unchanged). `ParallelOps` does NOT throw — trusts inputs validated upstream.

### Javadoc style (apply to: all new files)

**Source:** `NDArray.java:148-150`
```java
/**
 * Element-wise addition
 */
public NDArray add(NDArray other) {
```
**Apply to:** new public/package-private methods in `ParallelOps`. One-line description, no `@param`/`@return` — match the existing terse style. Note: `NumJa.java` (lines 21-25) and `ArrayOps.java` are richer (full `@param`/`@return`) — `ParallelOps` follows the lighter NDArray style since it's an internal utility.

### Test layout (apply to: `ParallelOpsTest`, `ParallelElementwiseTest`, `ParallelRegressionTest`)

**Source:** `modules/numja/src/test/java/com/numja/core/NDArrayTest.java:1-23`
```java
package numja.core;

import numja.core.NDArray;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NDArrayTest {
    @Test
    public void arithmeticAndReductions() {
        NDArray a = new NDArray(1.0, 2.0, 3.0);
        ...
        assertEquals(5.0, sum.getData().get(0,0), 1e-9);
    }
}
```
**Apply to:** path `modules/numja/src/test/java/com/numja/core/`, package declaration `package numja.core;`, JUnit 4 (`junit:4.13.2`), delta `1e-9` for value checks / `1e-13` for parallel-vs-sequential regression.

### Random-vec helper for tests (apply to: `ParallelElementwiseTest`, `ParallelRegressionTest`)

**Source:** `CoreBench.java:32-37`
```java
private static NDArray randomVec(int n, long seed) {
    java.util.Random rng = new java.util.Random(seed);
    double[] a1 = new double[n];
    for (int i = 0; i < n; i++) a1[i] = rng.nextDouble();
    return new NDArray(a1);
}
```
**Apply to:** copy verbatim into test classes (or share via a `TestUtils` class — but RESEARCH doesn't call for a new util, and a 5-line helper per test class is the lazy option; match the bench-file precedent of `private static` helpers in each test).

### ForkJoinPool sizing (apply to: `ThreadPoolConfig.getForkJoinPool()`)

**Source:** `ThreadPoolConfig.java:66-68`
```java
public int getCurrentThreads() {
    return currentThreads;
}
```
**Apply to:** `new ForkJoinPool(currentThreads)` — already capped at 60% of system max via `calculateOptimalThreads()` line 38-40. Don't call `Runtime.getRuntime().availableProcessors()` directly — that defeats the existing config knob.

## No Analog Found

None — every new file has a same-role or self analog in the repo. JDK stdlib (`ForkJoinPool`, `RecursiveAction`) provides the parallel primitive; the repo provides the sequential loop, the singleton pattern, the JMH harness shape, the test style, and the baseline doc layout.

## Metadata

**Analog search scope:**
- `modules/numja/src/main/java/numja/**/*.java`
- `modules/numja/src/test/java/com/numja/**/*.java`
- `modules/sklearn/src/main/java/sklearn/utils/ParallelUtils.java`
- `modules/sklearn/src/test/java/sklearn/accuracy/GoldenReferenceTest.java`
- `bench/src/main/java/bench/CoreBench.java`
- `.planning/phases/01-baseline-benchmark/BASELINE.md`

**Files scanned:** 9 source files + 1 doc file (no CLAUDE.md / no skills directory in repo).

**Pattern extraction date:** 2026-08-27
