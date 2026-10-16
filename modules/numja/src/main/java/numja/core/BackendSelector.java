package numja.core;

import java.util.Set;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Single source of truth for the active {@link ComputeBackend}.
 *
 * <p>Defaults to {@link CpuThreadBackend#getInstance()}. {@link #setBackend(String)}
 * honors an explicit allowlist + opt-in flag (HW-01 threat model). Unknown names
 * — even with the opt-in flag set — fall back silently to {@code CpuThreadBackend}
 * and emit a warning to {@code System.err}. This class never reflectively loads an
 * arbitrary class name: there is no class-loader dispatch, only a closed
 * {@code Set<String>} membership check.
 *
 * <p>The {@link #THRESHOLD_GPU} constant is declared for 05-02 (where dispatch
 * branches on size). It is intentionally unused in 05-01: the GPU path is never
 * auto-promoted in this phase.
 *
 * <p>The static wrappers {@link #sum}, {@link #prod}, {@link #min}, {@link #max},
 * {@link #elementwiseBinary}, {@link #elementwiseUnary}, {@link #scalarBinary},
 * and {@link #matmul} are the HW-02 gated dispatchers: each calls
 * {@link #checkSize(long)} before delegating to {@link #get()}(). Named after
 * the {@link ComputeBackend} methods so callers swap a one-line site without
 * renaming.
 */
public final class BackendSelector {

    /**
     * Cell-count ceiling above which a GPU backend would be considered in 05-02.
     * Declared here per research; not consulted in 05-01.
     */
    public static final int THRESHOLD_GPU = 4_096_000;

    /**
     * HW-02 DoS guard ceiling for the dispatcher. 10^9 doubles = 8 GB, far beyond
     * any single JVM op but small enough to allow legitimate matmul of e.g.
     * 31622^2 = ~10^9 cells. Aligns in spirit with
     * {@code pandas.internal.ChunkedReadOptions.MAX_CHUNK_ROWS = 100_000} (a
     * per-chunk row ceiling) but is per-op, not per-chunk. Enforced by
     * {@link #checkSize(long)}; oversized n throws
     * {@link IllegalArgumentException} before any backend call.
     */
    public static final long MAX_DISPATCH_N = 1_000_000_000L;

    /**
     * Test-only seam (read by {@link #activeCeiling()}): when the JVM system
     * property {@code numja.backend.maxDispatchN} is set to a parseable long,
     * the dispatcher uses that value instead of {@link #MAX_DISPATCH_N}.
     * Tests set this so they can exercise the ceiling throw without
     * allocating gigabyte-scale buffers. Production callers leave the
     * property unset; {@code checkSize} then reads the real ceiling.
     */

    private static final Set<String> ALLOWLIST = Set.of("cpu-thread");

    private static volatile ComputeBackend active = CpuThreadBackend.getInstance();

    private BackendSelector() {}

    /** Returns the currently active backend. Never {@code null}. */
    public static ComputeBackend get() {
        return active;
    }

    /**
     * Attempts to set the active backend to the named one. Honors an explicit
     * allowlist and the {@code numja.backend.allow} opt-in flag. Unknown names
     * — or a missing opt-in flag — fall back to {@link CpuThreadBackend} and
     * log a warning. Never throws on an unknown name; never reflectively loads
     * a class.
     */
    public static void setBackend(String name) {
        ComputeBackend fallback = CpuThreadBackend.getInstance();
        boolean allowed = "true".equals(System.getProperty("numja.backend.allow"));
        if (allowed && name != null && ALLOWLIST.contains(name)) {
            if ("cpu-thread".equals(name)) {
                active = fallback;
                return;
            }
        }
        // Fallback path: opt-in missing, name unknown, or name null.
        if (name != null && !ALLOWLIST.contains(name)) {
            System.err.println("[BackendSelector] unknown backend name '" + name
                + "'; falling back to cpu-thread");
        }
        active = fallback;
    }

    // ---------------------------------------------------------------------
    // HW-02 gated dispatchers. Each calls checkSize(long) before delegating
    // to the active backend. Same name as the ComputeBackend method so the
    // call site swap is one identifier.
    // ---------------------------------------------------------------------

    public static void elementwiseBinary(double[] a, double[] b, double[] out, DoubleBinaryOperator op) {
        checkSize(a.length);
        get().elementwiseBinary(a, b, out, op);
    }

    public static void elementwiseUnary(double[] in, double[] out, DoubleUnaryOperator op) {
        checkSize(in.length);
        get().elementwiseUnary(in, out, op);
    }

    public static void scalarBinary(double[] a, double scalar, double[] out, DoubleBinaryOperator op) {
        checkSize(a.length);
        get().scalarBinary(a, scalar, out, op);
    }

    public static double sum(double[] data) {
        checkSize(data.length);
        return get().sum(data);
    }

    public static double prod(double[] data) {
        checkSize(data.length);
        return get().prod(data);
    }

    public static double min(double[] data) {
        checkSize(data.length);
        return get().min(data);
    }

    public static double max(double[] data) {
        checkSize(data.length);
        return get().max(data);
    }

    /**
     * Dispatcher wrapper for matmul. {@code n} is the output cell count
     * ({@code aRows * bCols}); the caller's input arrays need only be sized
     * for the actual matmul body — but the checkSize trip rejects the call
     * before any allocation or EJML work, so we don't blow up on hostile
     * dim products.
     */
    public static void matmul(double[] a, int aRows, int aCols,
                              double[] b, int bRows, int bCols,
                              double[] out) {
        checkSize((long) aRows * bCols);
        get().matmul(a, aRows, aCols, b, bRows, bCols, out);
    }

    /**
     * Throws {@link IllegalArgumentException} when {@code n} exceeds the
     * active ceiling (production: {@link #MAX_DISPATCH_N}; test override:
     * {@code numja.backend.maxDispatchN}). Long-typed to avoid overflow on
     * dim-product inputs like {@code aRows * bCols}.
     */
    private static void checkSize(long n) {
        long ceiling = activeCeiling();
        if (n > ceiling) {
            throw new IllegalArgumentException(
                "dispatch n=" + n + " exceeds MAX_DISPATCH_N=" + ceiling);
        }
    }

    private static long activeCeiling() {
        // System-property read on every checkSize call so tests can toggle the
        // override by setProperty/clearProperty without restarting the JVM.
        // Property read cost is negligible compared to the FJP work that
        // follows; the alternative (caching the override) would couple test
        // ordering to BackendSelector state in ways that hurt isolation.
        String prop = System.getProperty("numja.backend.maxDispatchN");
        if (prop != null) {
            try {
                return Long.parseLong(prop);
            } catch (NumberFormatException ignored) {
                // Fall through to the production ceiling on a malformed value.
            }
        }
        return MAX_DISPATCH_N;
    }
}
