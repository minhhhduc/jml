package com.numja.core;

import numja.core.ArrayOps;
import numja.core.BackendSelector;
import numja.core.ComputeBackend;
import numja.core.NDArray;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Golden-value tests pinning the pre-refactor numeric path after wiring
 * {@link ArrayOps#dot} through {@link BackendSelector}.
 *
 * <p>Both {@code ArrayOps.dot} and {@code CpuThreadBackend.matmul} route to the
 * same EJML {@code CommonOps_DDRM.mult} call, so the post-refactor output is
 * bit-identical to the pre-refactor inline EJML path. These tests assert that
 * equivalence on a fixed-seed 64x64 matmul; if a future backend changes the
 * numerics they fail fast.
 *
 * <p>The HW-02 ceiling test (see T-05-HW-02 in {@code 05-02-PLAN.md} threat
 * model) is colocated here because the size ceiling and the dispatch wiring
 * are the same change.
 */
public class CpuThreadBackendGoldenTest {

    private String savedAllow;
    private String savedBackendName;

    @Before
    public void saveSystemProps() {
        savedAllow = System.getProperty("numja.backend.allow");
        savedBackendName = System.getProperty("numja.backend.name");
    }

    @After
    public void restoreSystemProps() {
        if (savedAllow == null) {
            System.clearProperty("numja.backend.allow");
        } else {
            System.setProperty("numja.backend.allow", savedAllow);
        }
        if (savedBackendName == null) {
            System.clearProperty("numja.backend.name");
        } else {
            System.setProperty("numja.backend.name", savedBackendName);
        }
        // Reset selector so we don't leak state to neighbors.
        BackendSelector.setBackend("cpu-thread");
    }

    /** Builds a deterministic n*n row-major double[] from seed 42L. */
    private static double[] seededMatrix(int n, long seed) {
        Random rng = new Random(seed);
        double[] a = new double[n * n];
        for (int i = 0; i < a.length; i++) a[i] = rng.nextDouble();
        return a;
    }

    /**
     * Pre-refactor reference: inline EJML mult on raw arrays.
     * Both paths call the same {@code CommonOps_DDRM.mult} so this is the
     * stable reference that survives any BackendSelector refactor.
     */
    private static double[] inlineEjlMult(double[] a, double[] b, int n) {
        DMatrixRMaj ma = new DMatrixRMaj(n, n, true, a);
        DMatrixRMaj mb = new DMatrixRMaj(n, n, true, b);
        DMatrixRMaj mo = new DMatrixRMaj(n, n);
        CommonOps_DDRM.mult(ma, mb, mo);
        return mo.data;
    }

    /**
     * Test 1: {@link ArrayOps#dot} on a 64x64 fixed-seed matmul must equal the
     * inline EJML reference bit-for-bit. The reference is recomputed each run
     * from the unchanged inline code path so it cannot drift.
     */
    @Test
    public void goldenDotMatchesPreRefactor() {
        final int n = 64;
        double[] rawA = seededMatrix(n, 42L);
        double[] rawB = seededMatrix(n, 43L);
        double[] expected = inlineEjlMult(rawA, rawB, n);

        NDArray a = new NDArray(new DMatrixRMaj(n, n, true, rawA.clone()), new int[]{n, n});
        NDArray b = new NDArray(new DMatrixRMaj(n, n, true, rawB.clone()), new int[]{n, n});
        NDArray actual = ArrayOps.dot(a, b);

        assertNotNull("dot returned null", actual);
        assertArrayEquals(
            "ArrayOps.dot output drifted from inline EJML reference",
            expected, actual.getData().data, 0.0);
    }

    /**
     * Test 2: Calling {@link BackendSelector#get()}().{@code matmul} directly
     * must also match the inline EJML reference. This isolates the interface
     * path itself (not just {@code ArrayOps.dot}'s wrapper) so any future
     * regression in the dispatch wrapper is caught independently.
     */
    @Test
    public void dotMatmulViaBackendSelector() {
        final int n = 64;
        double[] rawA = seededMatrix(n, 42L);
        double[] rawB = seededMatrix(n, 43L);
        double[] expected = inlineEjlMult(rawA, rawB, n);

        double[] out = new double[n * n];
        ComputeBackend backend = BackendSelector.get();
        backend.matmul(rawA, n, n, rawB, n, n, out);

        assertArrayEquals(
            "BackendSelector matmul output drifted from inline EJML reference",
            expected, out, 0.0);
    }

    /**
     * Test 3: HW-02 ceiling guard. A matmul whose output cell count
     * ({@code aRows * bCols}) exceeds {@link BackendSelector#MAX_DISPATCH_N}
     * must be rejected by the dispatcher before any backend work runs. We
     * pass small underlying arrays with a large {@code bCols} so the check
     * trips on the dim product alone — the matmul body never executes, so we
     * don't allocate gigabyte-scale buffers.
     */
    @Test
    public void dotDispatchCeilingThrows() {
        // aRows=1, bCols=MAX_DISPATCH_N + 1 → product = MAX_DISPATCH_N + 1 > MAX_DISPATCH_N.
        int aRows = 1, aCols = 1, bRows = 1;
        int bCols = (int) (BackendSelector.MAX_DISPATCH_N + 1L);
        double[] a = new double[1];
        double[] b = new double[1];
        double[] out = new double[1];
        try {
            // Call the HW-02 gated wrapper, not the raw ComputeBackend matmul —
            // the wrapper is where checkSize lives. The raw interface call would
            // skip the guard and let the backend try to allocate a 10^9-cell
            // output matrix (OOM).
            BackendSelector.matmul(a, aRows, aCols, b, bRows, bCols, out);
            fail("expected IllegalArgumentException for n > MAX_DISPATCH_N, got none");
        } catch (IllegalArgumentException ex) {
            String msg = ex.getMessage();
            assertTrue("message must reference 'exceeds', got: " + msg,
                msg != null && msg.contains("exceeds"));
            assertTrue("message must reference 'MAX_DISPATCH_N', got: " + msg,
                msg != null && msg.contains("MAX_DISPATCH_N"));
        }
    }

    /**
     * Test 4: Reduce ops wired in Task 2. {@link ArrayOps#sum} on a fixed-seed
     * 10_000-element array must equal {@link BackendSelector#get()}().{@code sum}
     * on the same underlying data. Tolerance accounts for FJP tree-merge
     * scheduling variance between two separate {@code ParallelOps.sum}
     * invocations: per-leaf Kahan is deterministic, but the tree merge order
     * across ForkJoinPool worker threads is not. {@code O(log n * eps)}
     * bounds the drift; we use 1e-9 which is well above that for n = 10_000.
     */
    @Test
    public void reduceOpsRoutedThroughBackend() {
        final int n = 10_000;
        double[] data = seededMatrix(n, 99L);
        NDArray arr = new NDArray(new DMatrixRMaj(n, 1, true, data.clone()), new int[]{n});

        double expectedSum = BackendSelector.get().sum(arr.getData().data);
        assertEquals("ArrayOps.sum drifted from BackendSelector.sum",
            expectedSum, ArrayOps.sum(arr), 1e-9);

        double expectedMin = BackendSelector.get().min(arr.getData().data);
        assertEquals("ArrayOps.min drifted from BackendSelector.min",
            expectedMin, ArrayOps.min(arr), 0.0);

        double expectedMax = BackendSelector.get().max(arr.getData().data);
        assertEquals("ArrayOps.max drifted from BackendSelector.max",
            expectedMax, ArrayOps.max(arr), 0.0);

        double expectedProd = BackendSelector.get().prod(arr.getData().data);
        assertEquals("ArrayOps.prod drifted from BackendSelector.prod",
            expectedProd, ArrayOps.prod(arr), 1e-9);
    }

    /**
     * Test 5: NDArray instance reduce methods must equal the corresponding
     * {@link BackendSelector} calls. Verifies the dispatch swap at the
     * NDArray instance-method boundary (sum/min/max/prod at lines 402/424/441/459).
     * Sum/prod use the FJP tolerance per Test 4; min/max are tree-merge
     * associative so exact == is safe.
     */
    @Test
    public void ndarrayInstanceOpsRoutedThroughBackend() {
        final int n = 10_000;
        double[] data = seededMatrix(n, 99L);
        NDArray arr = new NDArray(new DMatrixRMaj(n, 1, true, data.clone()), new int[]{n});

        double[] flat = arr.getData().data;
        assertEquals("NDArray.sum drifted from BackendSelector.sum",
            BackendSelector.get().sum(flat), arr.sum(), 1e-9);
        assertEquals("NDArray.min drifted from BackendSelector.min",
            BackendSelector.get().min(flat), arr.min(), 0.0);
        assertEquals("NDArray.max drifted from BackendSelector.max",
            BackendSelector.get().max(flat), arr.max(), 0.0);
        assertEquals("NDArray.prod drifted from BackendSelector.prod",
            BackendSelector.get().prod(flat), arr.prod(), 1e-9);
    }

    /**
     * Test 6: HW-02 ceiling guard on reduce ops. sum/min/max/prod each throw
     * when the input length exceeds the active ceiling. The test uses a
     * tiny ceiling override (via the {@code numja.backend.maxDispatchN}
     * system property, picked up by {@code BackendSelector.checkSize}) so we
     * don't have to allocate a >10^9-element array in JVM heap. The
     * dispatcher's checkSize consults the override when the property is set;
     * production callers leave it unset, so this only affects test runs.
     */
    @Test
    public void dispatchCeilingEnforcedForReductions() {
        final long tinyCeiling = 100L;
        System.setProperty("numja.backend.maxDispatchN", Long.toString(tinyCeiling));
        try {
            double[] oversized = new double[(int) tinyCeiling + 1];
            try {
                BackendSelector.sum(oversized);
                fail("sum must throw on n > ceiling");
            } catch (IllegalArgumentException ex) {
                assertTrue("sum message missing 'exceeds': " + ex.getMessage(),
                    ex.getMessage() != null && ex.getMessage().contains("exceeds"));
            }
            try {
                BackendSelector.min(oversized);
                fail("min must throw on n > ceiling");
            } catch (IllegalArgumentException ex) {
                assertTrue("min message missing 'exceeds': " + ex.getMessage(),
                    ex.getMessage() != null && ex.getMessage().contains("exceeds"));
            }
            try {
                BackendSelector.max(oversized);
                fail("max must throw on n > ceiling");
            } catch (IllegalArgumentException ex) {
                assertTrue("max message missing 'exceeds': " + ex.getMessage(),
                    ex.getMessage() != null && ex.getMessage().contains("exceeds"));
            }
            try {
                BackendSelector.prod(oversized);
                fail("prod must throw on n > ceiling");
            } catch (IllegalArgumentException ex) {
                assertTrue("prod message missing 'exceeds': " + ex.getMessage(),
                    ex.getMessage() != null && ex.getMessage().contains("exceeds"));
            }
        } finally {
            System.clearProperty("numja.backend.maxDispatchN");
        }
    }
}
