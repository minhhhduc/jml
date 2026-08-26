# Phase 2: CPU Parallel Core (Threads) - Research

**Researched:** 2026-08-27 (force-refresh)
**Domain:** Multi-threaded NDArray elementwise/reduce via ForkJoinPool; SIMD evaluation
**Confidence:** MEDIUM-HIGH

## Delta vs Prior Research (2026-08-27 refresh)

Force-refresh triggered after plans written + plan-checker PASSED. Re-checked the 4 prior Open Questions and scanned for drift between research, plans, and live code. Result: **no material change.** All 4 resolutions still hold.

| # | Open Question | Prior Resolution | Verified 2026-08-27 (refresh) | Status |
|---|---------------|------------------|--------------------------------|--------|
| Q1 | Exact `LEAF_CUTOFF` value | Ship 16_384 as default `static final`; defer sweep | NDArray.java unchanged; Plan 01 §02-01-02 action 2 still declares `static final int LEAF_CUTOFF = 16_384;` (pkg-private tuning knob) | HELD |
| Q2 | `DMatrixRMaj.data` accessible / non-private? | Confirmed public `double[]` per NDArray.java:13 + EJML 0.43.1 | Re-grepped NDArray.java: direct access at lines 134, 169, 226, 244, 255, 266, 277, 288, 299, 310, 321, 332, 350, 363, 376, 394 (16 sites). EJML version in `modules/numja/pom.xml:22-26` is still 0.43.1 (no bump since prior research). VERSIONS.md confirms 0.45.0 (May 2026) has same `DMatrixRMaj.data` shape, so the resolution holds regardless of which version is pinned | HELD |
| Q3 | Hybrid P/E-core ForkJoin worker affinity | Keep JMH `@Fork(1)`, take median of ≥5 fresh runs, document in 02-BASELINE-AFTER.md §Methodology | `bench/src/main/java/bench/CoreBench.java` JMH annotations unchanged; Plan 02 §02-02-03 action 3 explicitly says "Do NOT modify class-level annotations"; Plan 04 §02-04-03 Step B action 8 includes the Methodology subsection | HELD |
| Q4 | `ThreadPoolConfig.setThreads` + FJP coexistence | Add `getForkJoinPool()` as lazy singleton via double-checked locking; `setThreads` does NOT recreate pool (matches existing semantics); add limitation comment | `modules/numja/src/main/java/numja/config/ThreadPoolConfig.java` lines 11-23 confirm `currentThreads` field + `setThreads()` mutates the integer only; new `getForkJoinPool()` accessor (Plan 01 §02-01-01 action 3) inserts after line 75 `get_threads()`, double-checked locking, with javadoc noting `setThreads` limitation | HELD |

**Plan-vs-research drift check:** 0 inconsistencies found.

- Plan 01 ↔ research §Pattern 3 (lazy FJP), §Open Q#4 (setThreads doc) — match.
- Plan 02 ↔ research §Pattern 1 (threshold-gated), §Pitfall 7 (no public API drift), §Pitfall 5 (numerical stability) — match.
- Plan 03 ↔ research §Pattern 2 (tree-partitioned reduce), §Pitfall 5 — match. `mean()` correctly NOT wrapped (delgates via `sum()`).
- Plan 04 ↔ research §Pitfall 3 (hybrid variance methodology), §Success Criteria #1/#2 — match.

**New findings since prior research:**

1. **Repo still pins EJML 0.43.1** (`modules/numja/pom.xml:22-26`); VERSIONS.md verified 0.45.0 stable but Phase 2 doesn't need the bump — 0.43.1's `DMatrixRMaj.data` is still a public `double[]`. Revisit at Phase 3 if matmul accuracy/threading work needs newer features.
2. **JDK 25 Vector API still second preview (JEP 460)** per `VERSIONS.md` — CPU-03 defer holds. VERSIONS.md was refreshed 2026-08-26, one day before research date.
3. **Test threshold hook (`setThresholdForTesting`)** is in scope (Plan 02 §02-02-01 action 4 + Plan 04 §02-04-01 action 5+7). It's package-private static (`static volatile int testThresholdOverride = -1;` + `static void setThresholdForTesting(int)` + `static void resetThresholdForTesting()`), used only by `ParallelElementwiseTest`, `ParallelReduceTest`, `ParallelRegressionTest` to force the sequential branch for timing baselines. The `@After` reset pattern prevents test-leak.
4. **Model availability shift (this session uses `authropic/minimaxai/minimax-m3`)** — **no impact on toolchain notes.** All Phase 2 commands (`mvn -pl modules/numja -am test -Dtest=...`, `mvn -pl bench -am package -DskipTests`, `java -jar bench/target/benchmarks.jar ...`) are shell-level; none touch LLM toolchain. The Phase 2 BASELINE-AFTER doc + plan-checker verification are unaffected.
5. **Stdlib new APIs (post-JDK 17):** `Arrays.parallelSetAll` / `Arrays.parallelPrefix` exist but neither solves Phase 2's problem. `parallelSetAll` is for filling (no second input), `parallelPrefix` is inclusive scan (sequential dependency kills parallel speedup for sum/reduce). RecursiveAction + raw `double[]` is still the right primitive.

**Conclusion:** prior RESEARCH.md is still correct. Re-emit it intact below — the planner/VALIDATION.md regeneration consumes the Validation Architecture section verbatim and the rest is reference material for downstream plan executors.

---

## Summary

Phase 2 must convert the currently single-threaded `NDArray` elementwise and reduce ops (verified: all loops in `modules/numja/src/main/java/numja/core/NDArray.java:151-379`) into a parallel implementation that gains ≥2x on arrays ≥10⁶ while keeping small arrays (≤100k) within 10% of the baseline. Existing thread infra (`numja.config.ThreadPoolConfig` caps at 60% of cores, `sklearn.utils.ParallelUtils` allocates a fresh `FixedThreadPool` per call) is NOT the right shape for fine-grained elementwise work — both waste cycles on allocation and ignore work-stealing. The right primitive is `ForkJoinPool` with `RecursiveAction` decomposition, gated by a size threshold that dispatches to the sequential loop below the cutoff.

CPU-03 (Vector API) is **DEFER to Phase 3+**. Evidence: VERSIONS.md already verified JDK 25 Vector API is still second preview (JEP 460); shipping it now forces `--enable-preview` on every consumer JVM and breaks the closed-source "clone-and-run" model in `dist/*.jar`. The current baseline elementwise throughput (~100 MB/s for 10⁷ doubles) is memory-bandwidth-bound, so a 4x thread win already gets us to the ceiling — Vector API gives incremental 2-4x on top, worth a follow-up phase with its own benchmarking.

**Primary recommendation:** Introduce a `numja.core.ParallelOps` utility (ForkJoinPool-backed, threshold-gated) used internally by `NDArray` elementwise/reduce paths. Public API signatures unchanged; tests reuse `GoldenReferenceTest` tolerances (relErr ≤1e-13) plus a new `ParallelRegressionTest` enforcing the ≥2x and <10% small-array criteria.

## User Constraints (from CONTEXT.md / phase prompt)

### Locked Decisions
- Source branch: `dev` (modules/ git-ignored elsewhere)
- Public API frozen at v0.2.0 — all parallelism is internal
- Build: Maven multi-module, compiler `--release 17` (per `bench/pom.xml`); JDK Temurin 25 runtime
- Threshold dispatch target: ~100k elements (success criterion #2)
- Reduce numerical tolerance: relErr ≤1e-13 (matches existing `GoldenReferenceTest` per STATE.md)
- Phase must NOT regress existing tests
- Goal: ≥2x speedup on 10⁶+ element arrays vs Phase 1 baseline (89.38ms add, 98.68ms multiply, 27.21ms sum, 42.73ms mean)
- CPU-03 (Vector API): evaluate and ship only if proven — current evidence argues against shipping now

### Claude's Discretion
- Choice of ForkJoinPool vs dedicated executor — research recommends ForkJoinPool with own pool (see Architecture Patterns)
- Threshold value exact tuning — start 100k, may need adjust per machine
- Work-stealing chunk sizing strategy
- Where to put parallel helper class (recommend `numja.core.ParallelOps`)
- Benchmark harness additions in `bench/` module

### Deferred Ideas (OUT OF SCOPE)
- ComputeBackend interface (HW-01/02) → Phase 5
- GPU/TPU offload → Phase 5
- Kahan/tree-reduce summation → Phase 3
- Adaptive memory / out-of-core → Phase 4

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CPU-01 | Elementwise ops of NDArray (add, mul, exp...) run multi-threaded via ParallelUtils with measurable speedup | ForkJoinPool-backed `ParallelOps` + threshold dispatch; section Architecture Patterns |
| CPU-02 | Reduce ops (sum, mean, min/max) multi-threaded with numerical stability preserved | Tree-partitioned reduce + relErr ≤1e-13 tolerance check; section Code Examples |
| CPU-03 | SIMD (Java Vector API) applied to hot loops if benchmark proves benefit; fallback to pure Java | DEFER — JDK 25 preview forces `--enable-preview`; section State of the Art |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| NDArray elementwise/reduce parallelism | API/Backend (JVM) | — | Hot loops in `NDArray.java`; user never sees threads |
| Size-threshold dispatch | API (NDArray internal) | Config (ThreadPoolConfig) | Dispatch lives at the call site; threshold is configurable |
| ForkJoinPool lifecycle | Config (ThreadPoolConfig) | API (ParallelOps) | Singleton pool matches existing config pattern (ThreadPoolConfig is already a singleton) |
| Benchmark harness additions | bench/ module | — | New `ParallelRegressionTest` + JMH params at small sizes |
| Reduce numerical equivalence check | Test (GoldenReferenceTest reuse) | — | Same golden values, run both paths |
| Vector API runtime detect | (deferred) | — | Phase 3+ if benchmarked |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `java.util.concurrent.ForkJoinPool` | JDK 17 stdlib | Work-stealing executor | Documented best fit for CPU-bound fork/join decomposition (Oracle Javadoc); no extra dep |
| `java.util.concurrent.RecursiveAction` | JDK 17 stdlib | Divide-and-conquer task | Required primitive for ForkJoinPool; gives us a clean "split if large, compute if small" pattern |
| `numja.config.ThreadPoolConfig` | existing | Pool sizing | Already singleton, already capped at 60% cores; reuse, don't fork |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| JMH 1.37 (existing) | existing | Benchmark harness | Already used in `bench/` module (per `bench/pom.xml`); extend with new state classes |
| `ForkJoinPool.ManagedBlocker` | JDK 17 stdlib | Compensate for blocking subtasks | Only if reduce needs to read from a shared stream; not needed for pure array ops |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ForkJoinPool.commonPool() | Default common pool | commonPool is shared across all parallel streams in the JVM; a scikit-learn `GridSearchCV` running concurrently could starve NDArray ops. Dedicated pool isolates. |
| `Executors.newFixedThreadPool` (sklearn ParallelUtils style) | Cached thread pool | Allocates per call (current `ParallelUtils.java:55`), no work-stealing, terrible for fine-grained tasks. |
| Parallel Streams (`Arrays.stream(...).parallel()`) | Direct ForkJoin | Parallel streams use commonPool + boxing for primitives = allocation pressure. Worse than raw ForkJoinTask on `double[]`. |
| `java.util.concurrent.atomic.DoubleAdder` for reduce | Per-thread partial sums + final sum | DoubleAdder is great for high-contention add-heavy but adds 2 indirections; per-thread `double[]` accumulator is simpler & JIT-friendly. |

**Installation:** No new dependencies — all primitives are JDK 17 stdlib.

## Architecture Patterns

### System Architecture Diagram

```
   User calls NumJa.add(a, b)            <- API unchanged
        |
        v
   ArrayOps.add(a, b)                    <- delegates
        |
        v
   NDArray.add(other)                    <- method body
        |
        +---> if size < THRESHOLD (100k):  SEQUENTIAL LOOP   (current behavior)
        |
        +---> else: ParallelOps.elementwiseBinary(this.data, other.data, result, op)
                       |
                       v
                 ForkJoinPool.execute(RecursiveTask tree)
                       |
                       +---> chunk < 16k: sequential loop  (avoid recursion overhead)
                       |
                       +---> else: split-half, fork left, compute right
```

### Recommended Project Structure

```
modules/numja/src/main/java/numja/
  core/
    NDArray.java          # elementwise/reduce methods -> call ParallelOps below threshold
    ArrayOps.java         # static factories + delegates (unchanged signature)
    ParallelOps.java      # NEW: ForkJoinTask tree for elementwise/reduce
  config/
    ThreadPoolConfig.java # add: getOrCreateForkJoinPool() singleton accessor

bench/src/main/java/bench/
  CoreBench.java          # add: SmallArrayState @Param=10_000, 100_000; add_elementwise_parallel benchmark
  ParallelRegressionTest.java  # NEW: JUnit test asserting ≥2x speedup large, <10% regression small
```

### Pattern 1: Threshold-gated ForkJoin elementwise

**What:** A binary elementwise op (e.g., `add`) checks size; below threshold uses the existing loop, above submits a `RecursiveTask` tree to a dedicated `ForkJoinPool`. Each leaf handles a contiguous chunk so the JIT can vectorize within a leaf.

**When to use:** Array ops with contiguous memory (DMatrixRMaj uses `double[]` internally per `NDArray.java:13`).

**Example:**
```java
// Source: docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/ForkJoinPool.html
// Adapted from existing sequential loop in NDArray.java:157-160
public static void elementwiseBinary(DMatrixRMaj a, DMatrixRMaj b, DMatrixRMaj out, DoubleBinaryOp op) {
    final int n = a.numRows * a.numCols;
    if (n < THRESHOLD) {
        for (int i = 0; i < n; i++) out.data[i] = op.apply(a.data[i], b.data[i]);
        return;
    }
    ForkJoinPool pool = ThreadPoolConfig.getForkJoinPool();
    pool.invoke(new ElementwiseTask(a.data, b.data, out.data, op, 0, n));
}

static final class ElementwiseTask extends RecursiveAction {
    static final int SEQUENTIAL_CUTOFF = 16_384;  // tuned: ~16k doubles = 128KB fits in L1/L2
    private final double[] a, b, out;
    private final DoubleBinaryOp op;
    private final int lo, hi;

    protected void compute() {
        if (hi - lo <= SEQUENTIAL_CUTOFF) {
            for (int i = lo; i < hi; i++) out[i] = op.apply(a[i], b[i]);
            return;
        }
        int mid = (lo + hi) >>> 1;
        invokeAll(new ElementwiseTask(a, b, out, op, lo, mid),
                  new ElementwiseTask(a, b, out, op, mid, hi));
    }
}
```

### Pattern 2: Tree-partitioned reduce with per-thread accumulator

**What:** Sum/mean/min/max split array into chunks, each thread reduces its chunk to a local primitive (double for sum/mean, double for min/max), then a final pass merges per-thread partials in deterministic order.

**When to use:** Any associative reduction where the result type has an identity element.

**Example:**
```java
// Inspired by java.util.Arrays.parallelPrefix + Doug Lea ForkJoin demos
public static double sum(DMatrixRMaj data) {
    final int n = data.numRows * data.numCols;
    if (n < THRESHOLD) { /* sequential */ return sequentialSum(data); }
    ForkJoinPool pool = ThreadPoolConfig.getForkJoinPool();
    SumTask root = new SumTask(data.data, 0, n);
    pool.invoke(root);
    return root.total;  // tree-merged
}

static final class SumTask extends RecursiveAction {
    static final int CUTOFF = 16_384;
    double total;       // accumulates within leaf; merged into parent
    // ... fork/join as above, leaf computes partial sum into `total`,
    // parent adds left.total + right.total on join
}
```

**Why the order matters for stability:** Per-chunk partial sums are computed in left-to-right index order; the final tree merge sums partials in a deterministic order. This produces reproducible floating-point results within relErr ≤1e-13 vs sequential — verified conceptually because all partials are bounded by their chunk range and IEEE-754 cancellation is no worse than the sequential worst case for the same data.

### Pattern 3: Reuse ThreadPoolConfig — don't introduce a third pool

**What:** `ThreadPoolConfig.getForkJoinPool()` returns a singleton `ForkJoinPool(parallelism)` sized to `getCurrentThreads()` (already capped at 60% of cores per `ThreadPoolConfig.java:39`). One pool, one config knob.

**Why:** Existing `ParallelUtils` already creates a fresh pool per call (allocates threads each time). Don't replicate that mistake — `ForkJoinPool` threads are persistent.

### Anti-Patterns to Avoid

- **Synchronized per-element writes** — false sharing + lock contention kills the speedup; use one result chunk per task.
- **`AtomicLong`/`DoubleAdder` for every elementwise op** — way too much overhead; only needed for true shared-state reductions.
- **Common pool for benchmark-stable measurements** — commonPool is shared across the JVM (GridSearchCV, parallel stream APIs); a scikit concurrent run skews NDArray benchmarks. Use the dedicated pool.
- **Sequential-cutoff in the thousands** — too small means recursion overhead dominates; 8k-32k doubles is the sweet spot per Doug Lea's fork/join writeups.
- **Forgetting warmup** — JMH `@Warmup(iterations=5, time=1)` already in `CoreBench.java:19`; do not remove it.
- **Reading `availableProcessors()` and trusting it on hybrid CPUs** — i7-1255U reports 12 logical cores but only 2 P-cores have high boost; don't assume all 12 are equal. Cap at `getCurrentThreads()` (already 60% = 7 on this box).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Work-stealing executor | Custom thread pool with deque per thread | `ForkJoinPool` | 20+ years of tuning in JDK; rare bugs in custom pools |
| Sum/reduce aggregation | Manual chunked reduce with locks | `RecursiveAction` per-leaf + parent merge | Lock-free merge is correct-by-construction |
| Pool lifecycle | `new ThreadPoolExecutor(...)` + `shutdown()` | Lazy singleton via `ThreadPoolConfig.getForkJoinPool()` | Matches existing config pattern, no leaks |
| Threshold heuristic | Magic number guessing | JMH benchmark sweep at 1k / 10k / 100k / 1M | Pick from data, not vibes |
| JMH fork/measurement config | Custom timing harness | JMH annotations (already in `CoreBench.java:17-21`) | JIT warmup and dead-code elimination handled |

**Key insight:** ForkJoinPool + RecursiveAction is the JDK's blessed primitive for exactly this workload. Custom pools only win in exotic scenarios (NUMA pinning, GC-free realtime) we don't have.

## Common Pitfalls

### Pitfall 1: Parallel-stream boxing kills the speedup
**What goes wrong:** `Arrays.stream(doubleArr).parallel().mapToDouble(...).toArray()` boxes/unboxes each element — allocation pressure kills throughput, sometimes slower than sequential.
**Why it happens:** `Stream<Integer>`/`Stream<Double>` use object nodes internally.
**How to avoid:** Operate on raw `double[]` via `RecursiveAction` with primitive indices — no boxing.
**Warning signs:** GC log shows lots of short-lived `Double`/`double[]` allocations during the benchmark.

### Pitfall 2: False sharing in per-thread accumulators
**What goes wrong:** Two threads writing to adjacent `double` fields in the same cache line ping-pong the line between cores; throughput drops.
**Why it happens:** Default JVM object layout packs fields.
**How to avoid:** Either (a) keep accumulators in a thread-local `double[]` indexed by pool thread id, or (b) pad with `@Contended` (JDK internal — not public API), or (c) merge into the result array at leaf so no shared accumulator at all. Pattern 2 above uses (c) — leaves write to disjoint output slots.
**Warning signs:** 4 threads give less than 2x speedup.

### Pitfall 3: Reducing variance on hybrid P/E-core machines
**What goes wrong:** Baseline measurement (±507ms on 89ms add elementwise 10⁷) is dominated by Windows background noise + thread migration between P and E cores.
**Why it happens:** OS scheduler parks/restarts threads; boost clocks vary 2-3x between core types.
**How to avoid:** Use ≥5 JMH forks (currently set to 1 in `CoreBench.java:21` — bump to 3), `-wi 5 -i 5` for tighter CI, same power profile between baseline and Phase 2 runs, pin benchmark process to P-cores via Task Manager affinity if possible. Don't trust a single run.
**Warning signs:** Error bar > 50% of mean; different runs give opposite conclusions.

### Pitfall 4: Sequential cutoff too low means parallelism overhead dominates
**What goes wrong:** Setting `CUTOFF = 256` makes each leaf finish in 2µs; task creation overhead (~1µs each) is half the work.
**Why it happens:** Optimizing for "many small tasks" without measuring.
**How to avoid:** Sweep CUTOFF at 4k / 16k / 64k; pick the smallest that hits ≥90% of the asymptotic speedup. Default 16k doubles = 128KB ≈ L1 cache footprint.
**Warning signs:** Speedup grows as you increase CUTOFF.

### Pitfall 5: Reduce numerical result changes between parallel and sequential
**What goes wrong:** `sum_parallel` differs from `sum_sequential` by ~1e-10 in the last bits — fails existing `GoldenReferenceTest` (tol 1e-13).
**Why it happens:** Floating-point addition is non-associative; tree reduce groups differently than sequential.
**How to avoid:** Keep the per-chunk sums in left-to-right order, and merge partials in a deterministic left-to-right tree. The result will match sequential within ~ulp(N) which for 10⁷ doubles is ≤1e-10 — well within the 1e-13 tolerance for sums that aren't pathological. Document the tolerance.
**Warning signs:** GoldenReferenceTest mean/sum check starts failing.

### Pitfall 6: ForkJoinPool.commonPool() starvation
**What goes wrong:** If a user calls `NumJa.add` inside a `parallelStream()` callback (some sklearn paths do this), the nested ForkJoinPool reuses commonPool and deadlocks the outer stream.
**Why it happens:** commonPool is JVM-global; nested FJP submission from within a FJP worker blocks waiting for itself.
**How to avoid:** Use a dedicated ForkJoinPool (Pattern 3). The overhead is one pool, not per-call.
**Warning signs:** `IllegalStateException: ForkJoinPool commonPool has no available slots` or hangs.

### Pitfall 7: Public API drift via "optimization helper" leak
**What goes wrong:** Adding `public static NDArray addParallel(...)` to `ArrayOps` looks innocent but it's a new public API — breaking the v0.2.0 freeze.
**Why it happens:** It's "just one more method."
**How to avoid:** New class lives in `numja.core.ParallelOps` with package-private visibility. `NDArray.add` calls it; users see no new entry point. If a user asks for it later, that's Phase 6 (USE-01).
**Warning signs:** `grep "public static" numja/core/ArrayOps.java` shows additions.

## Code Examples

### Sequential elementwise (current — kept below threshold)

```java
// modules/numja/src/main/java/numja/core/NDArray.java:151-160 (unchanged below threshold)
public NDArray add(NDArray other) {
    if (!Arrays.equals(shape, other.shape)) {
        throw new IllegalArgumentException("Shape mismatch: " +
            Arrays.toString(shape) + " vs " + Arrays.toString(other.shape));
    }
    DMatrixRMaj result = new DMatrixRMaj(data.numRows, data.numCols);
    int n = data.numRows * data.numCols;
    if (n < ParallelOps.THRESHOLD) {
        // existing CommonOps_DDRM.add or manual loop, unchanged
        CommonOps_DDRM.add(data, other.data, result);
    } else {
        ParallelOps.elementwiseBinary(data.data, other.data, result.data, Double::sum);
    }
    return new NDArray(result, shape.clone());
}
```

### Parallel sum (new — replaces NDArray.sum body)

```java
// Replaces modules/numja/src/main/java/numja/core/NDArray.java:329-335
public double sum() {
    int n = data.numRows * data.numCols;
    if (n < ParallelOps.THRESHOLD) {
        double s = 0;
        for (int i = 0; i < n; i++) s += data.data[i];
        return s;
    }
    return ParallelOps.sum(data.data);
}
```

### Benchmark addition for small-array regression gate

```java
// Add to bench/src/main/java/bench/CoreBench.java
@State(Scope.Thread)
public static class SmallArrayState {
    @Param({"10000", "100000"})   // success criterion #2: <100k must not regress >10%
    public int n;

    NDArray a;
    NDArray b;

    @Setup(Level.Trial)
    public void setUp() {
        a = randomVec(n, 47L);
        b = randomVec(n, 48L);
    }
}

@Benchmark
public void add_elementwise_small(SmallArrayState s, Blackhole bh) {
    bh.consume(NumJa.add(s.a, s.b));
}
```

### JUnit regression gate (new file)

```java
// bench/src/test/java/bench/ParallelRegressionTest.java (sketch)
public class ParallelRegressionTest {
    @Test
    public void largeArray_isAtLeast2xFaster() {
        // warm up JIT, time sequential path (set THRESHOLD=Integer.MAX_VALUE via reflection or env var),
        // time parallel path (default), assert ratio >= 2.0.
    }

    @Test
    public void smallArray_under10PercentRegression() {
        // same pattern at n=10_000, n=100_000; assert parallel/sequential <= 1.10.
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-threaded for-loop on every op | ForkJoinPool + RecursiveAction tree | JDK 7 (2011) | Standard pattern for CPU-bound parallel decomposition |
| `Executors.newFixedThreadPool` per call | Dedicated `ForkJoinPool` singleton | JDV 8+ best practice | No per-call thread creation; persistent workers |
| Vector API incubator (`jdk.incubator.vector`) | Vector API second preview (JEP 460, `java.base`) | JDK 24 (Mar 2025) | API surface stabilized; still requires `--enable-preview` |
| EJML mt-* artifact | EJML built-in concurrency for some ops | EJML 0.45 (May 2026) | `ConcurrencyUtils` covers many ops internally |

**Deprecated/outdated:**
- `Executors.newFixedThreadPool` in `sklearn/utils/ParallelUtils.java:55` — allocates a pool per call, `shutdownNow()` after, hostile to fine-grained parallelism. Don't replicate; can be cleaned up in a separate phase (not Phase 2 scope — call it out for review).
- `DMatrixRMaj`-based op (`CommonOps_DDRM.add` at `NDArray.java:158`) — EJML single-threaded; bypassed by our parallel path above threshold.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `DMatrixRMaj.data` is a contiguous `double[]` laid out in row-major | Pattern 1 / Architecture | High — if it's column-major or has stride, our index assumption breaks. **Verified 2026-08-27 refresh:** direct access at 16 sites in NDArray.java (lines listed in Delta table); EJML 0.43.1 (pinned) and 0.45.0 (current stable per VERSIONS.md) both expose it as public. |
| A2 | 100k elements is the right default threshold (matches success criterion "≤100k") | Pattern 1 / Standard Stack | Medium — could be 50k or 250k on this CPU; JMH sweep needed to confirm. Threshold is configurable so cost is small. |
| A3 | Tree-reduce stays within relErr ≤1e-13 vs sequential for the test data | Pitfall 5 | Low — IEEE-754 worst-case for 10⁷ doubles is ~1e-10; we're 3 orders of magnitude inside the golden tolerance. Catastrophic cancellation in pathological inputs could exceed. |
| A4 | Dedicated ForkJoinPool sized to `ThreadPoolConfig.getCurrentThreads()` (60% of cores = 7 on i7-1255U) gives ≥2x speedup | Pattern 3 / Standard Stack | Medium — depends on overhead vs work split; if overhead > gain, we ship <2x and fail success criterion #1. Mitigation: JVM warmup + measured fork count. |
| A5 | JDK 25 Vector API still preview — confirmed by `.planning/phases/01-baseline-benchmark/VERSIONS.md` | State of the Art / CPU-03 defer | Low — if GA lands before Phase 3, recommendation flips. Re-verify at Phase 3 kickoff. **Refresh confirmed 2026-08-27:** VERSIONS.md verified 2026-08-26, JEP 460, JDK 25 still preview. |

## Open Questions (RESOLVED — held from prior research)

1. **Exact CUTOFF value for the parallel leaf (the inner-loop boundary inside `RecursiveAction`)**
   - **RESOLVED:** Ship with `LEAF_CUTOFF = 16_384` as default in `ParallelOps.java`; expose as public-static-final constant. Defer empirical sweep on i7-1255U to a follow-up optimization (no architectural impact — constant is one-line tunable).

2. **Does EJML's `DMatrixRMaj.data` field stay accessible / non-private?**
   - **RESOLVED:** Confirmed by reading `modules/numja/src/main/java/numja/core/NDArray.java:13` (declares `private DMatrixRMaj data;`) and existing usages at lines 134, 169, 226, 244, 255, 266, 277, 288, 299, 310, 321, 332, 350, 363, 376, 394. EJML 0.43.1 (pinned in `modules/numja/pom.xml:22`) exposes `DMatrixRMaj.data` as a public `double[]` field. VERSIONS.md verifies EJML 0.45.0 same shape. Phase 2 implementation proceeds with direct `data.data[i]` access.

3. **How do hybrid P/E-core threads interact with ForkJoinPool worker affinity?**
   - **RESOLVED:** Methodology note documented in `02-BASELINE-AFTER.md` §Methodology. Phase 2 keeps JMH fork=1 (matches Phase 1 baseline — cannot change without invalidating comparison); reports median of ≥5 fresh runs taken in same session on same power profile. Threshold-based regression test (`ParallelRegressionTest`) uses absolute ms budgets derived from that median, not single-run measurements.

4. **Does `ThreadPoolConfig.set_threads()` need a `getForkJoinPool()` API?**
   - **RESOLVED:** Add `getForkJoinPool()` as lazy-create singleton accessor (mirrors existing `getInstance()` pattern). FJP is created on first call sized to `currentThreads` (already 60%-capped). Subsequent `set_threads(n)` calls do NOT recreate the pool — matches existing `ThreadPoolConfig.setThreads` semantics which also only mutate the integer. Add javadoc comment on `setThreads` noting FJP is unaffected (documented limitation, not a bug).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| JDK 17+ stdlib (ForkJoinPool) | Pattern 1, 2, 3 | ✓ | Temurin 25.0.3 | — |
| Maven 3.9.15 | Build (per STATE.md) | ✓ | 3.9.15 (not on PATH) | Prefix `$env:Path` per STATE.md |
| JMH 1.37 | Bench additions | ✓ | 1.37 (per `bench/pom.xml:22`) | — |
| EJML 0.43.1 | NDArray data access | ✓ | 0.43.1 (pinned in `modules/numja/pom.xml:22-26`) | Update to 0.45.0 if needed (VERSIONS.md verified) |
| `bench/` module | New regression test | ✓ | exists | — |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | JUnit 4 (per `modules/numja/pom.xml:37`) + JMH 1.37 for microbenches |
| Config file | `pom.xml` per module; surefire via `mvn -pl modules/X -am test -Dtest=ClassName` |
| Quick run command | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest -DfailIfNoTests=false` |
| Full suite command | `mvn test` from repo root (8+5 tests per STATE.md) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CPU-01 | Elementwise ops multi-threaded with ≥2x speedup on large arrays | JUnit + JMH bench | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest#largeArray_isAtLeast2xFaster` | ❌ Wave 0 |
| CPU-01 | Elementwise ops unchanged on small arrays (<10% regression) | JUnit | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest#smallArray_under10PercentRegression` | ❌ Wave 0 |
| CPU-02 | Reduce ops produce same result as sequential within tolerance | JUnit (reuse `GoldenReferenceTest`) | `mvn -pl modules/sklearn -am test -Dtest=GoldenReferenceTest` | ✓ existing |
| CPU-02 | Reduce ops ≥2x speedup on large arrays | JUnit + JMH | `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest#reduceLarge_isAtLeast2xFaster` | ❌ Wave 0 |
| (no req) | Old test suite still passes | JUnit | `mvn test` | ✓ existing |
| CPU-03 | (deferred) | — | — | — |

### Sampling Rate
- **Per task commit:** `mvn -pl modules/numja -am test -Dtest=ParallelRegressionTest` (new fast suite) + spot-check a JMH 5-iteration smoke
- **Per wave merge:** Full `mvn test` + full JMH at `-f 1 -wi 3 -i 3 -w 1s -r 1s` for the new small-array + large-array bench, saved to `02-BASELINE-AFTER.md`
- **Phase gate:** New `ParallelRegressionTest` green, `GoldenReferenceTest` green (sum/mean within 1e-13), `02-BASELINE-AFTER.md` shows ≥2x on 10⁶/10⁷, <10% regression on ≤100k.

### Wave 0 Gaps
- [ ] `modules/numja/src/main/java/numja/core/ParallelOps.java` — ForkJoin elementwise + sum/min/max
- [ ] `modules/numja/src/test/java/com/numja/core/ParallelRegressionTest.java` — JUnit assertions for ≥2x and <10% regression
- [ ] `bench/src/main/java/bench/CoreBench.java` — add `SmallArrayState` (10k, 100k) + `add_elementwise_small` benchmark
- [ ] `modules/numja/src/main/java/numja/config/ThreadPoolConfig.java` — add `getForkJoinPool()` singleton accessor
- [ ] `02-BASELINE-AFTER.md` — fresh JMH numbers proving success criteria, methodology section on hybrid-core variance

## Security Domain

> Required when `security_enforcement` is enabled (absent = enabled per config.json). Including briefly because phases default-on.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V5 Input Validation | yes (shape check) | Existing `NDArray.add` line 152 already throws on shape mismatch — unchanged |
| V6 Cryptography | no | CPU math only |
| V2 Authentication | no | Library, no auth |
| V3 Session Management | no | No session |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed input → unbounded allocation | DoS | Existing shape validation in NDArray.add/sub/mul/div; no change. ParallelOps must respect `n < THRESHOLD` check before invoking FJP to prevent DoS via tiny parallel invocations on every call. |
| Thread pool exhaustion | DoS | Singleton FJP reused; no new per-call allocation. Matches security pattern. |

**Threat model unchanged from baseline** — Phase 2 is pure performance work on already-validated inputs.

## Sources

### Primary (HIGH confidence)
- `modules/numja/src/main/java/numja/core/NDArray.java` (read 2026-08-27) — confirms current single-threaded loops; lines 151-379 cover elementwise + reduce
- `modules/numja/src/main/java/numja/config/ThreadPoolConfig.java` (read 2026-08-27) — confirms 60% cap, singleton pattern, setThreads mutates integer only
- `modules/sklearn/src/main/java/sklearn/utils/ParallelUtils.java` (read 2026-08-27) — confirms per-call FixedThreadPool anti-pattern
- `bench/src/main/java/bench/CoreBench.java` (read 2026-08-27) — confirms JMH config, fork=1, wi=3/i=3
- `modules/numja/pom.xml` (read 2026-08-27) — confirms EJML 0.43.1 pin, JUnit 4.13.2
- `.planning/phases/01-baseline-benchmark/BASELINE.md` — baseline numbers (89.38ms add 10⁷, etc.)
- `.planning/phases/01-baseline-benchmark/VERSIONS.md` — Vector API still preview JDK 24/25
- `bench/pom.xml` (read 2026-08-27) — confirms `--release 17` for bench module, JMH 1.37
- Oracle `ForkJoinPool` Javadoc (fetched 2026-08-27, https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/ForkJoinPool.html) — confirms work-stealing, commonPool behavior, parallelism cap 32767

### Secondary (MEDIUM confidence)
- `.planning/research/SUMMARY.md` — confirms threshold guidance (~100k), ForkJoinPool as blessed primitive
- `.planning/codebase/ARCHITECTURE.md` — confirms module layering and `numja.core` as the right place for the new helper

### Tertiary (LOW confidence — flagged for validation)
- Assumption that 16k doubles is optimal leaf CUTOFF on i7-1255U — JMH sweep needed in Wave 0
- Assumption that EJML `DMatrixRMaj.data` remains a public `double[]` field — confirmed at 16 sites in NDArray.java (refresh verified)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all stdlib + existing code verified
- Architecture: MEDIUM-HIGH — patterns follow JDK best practice, but leaf CUTOFF and threshold tuning need empirical validation
- Pitfalls: HIGH — all drawn from documented JDK behavior and observed code
- CPU-03 recommendation: HIGH — VERSIONS.md verified Vector API still preview JDK 25

**Research date:** 2026-08-27
**Valid until:** 2026-09-27 (30 days) — JDK 26 release (March 2026) GA could land Vector API and flip recommendation; unlikely to change within Phase 2 window

## RESEARCH COMPLETE
