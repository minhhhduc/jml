package com.numja.core;

import numja.core.BackendSelector;
import numja.core.ComputeBackend;
import numja.core.CpuThreadBackend;
import numja.core.ParallelOps;

import org.junit.After;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

/**
 * Selector + backend contract coverage for Phase 5 (HW-01).
 *
 * Why this exists (vs in-process unit checks only): the abstract ComputeBackend
 * is the swap point for future GPU/TPU implementations. We need to lock down
 * three properties so 05-02 can wire dispatch without breaking the contract:
 *
 *  - defaultBackendIsCpuThread: a fresh JVM must return CpuThreadBackend singleton
 *    via BackendSelector.get(). 05-02 hot-path dispatch relies on this default.
 *
 *  - systemPropertyAllowlistHonored: setBackend(name) must consult the
 *    numja.backend.allow opt-in flag and a closed allowlist. Unknown names fall
 *    back to CpuThreadBackend (defends HW-01 EoP: no arbitrary class loading).
 *
 *  - cpuBackendNumericallyEquivalent: CpuThreadBackend must produce bit-identical
 *    results to ParallelOps for the same input. 1:1 delegation is the zero-
 *    behavior-change guarantee — if this breaks, 05-02 picks up silent drift.
 */
public class BackendSelectorTest {

    @After
    public void resetSelector() {
        // Restore default state so other tests in the JVM don't observe a polluted
        // backend. Clear both system properties last (mirror ParallelOps test hygiene).
        System.setProperty("numja.backend.allow", "true");
        BackendSelector.setBackend("cpu-thread");
        System.clearProperty("numja.backend.allow");
        System.clearProperty("numja.backend");
    }

    @Test
    public void defaultBackendIsCpuThread() {
        assertSame(CpuThreadBackend.getInstance(), BackendSelector.get());
    }

    @Test
    public void systemPropertyAllowlistHonored() {
        ComputeBackend cpuInstance = CpuThreadBackend.getInstance();

        // Opt-in flag + known name -> selector returns CpuThreadBackend.
        System.setProperty("numja.backend.allow", "true");
        BackendSelector.setBackend("cpu-thread");
        assertSame(cpuInstance, BackendSelector.get());

        // Opt-in flag + unknown name -> falls back to CpuThreadBackend, no throw.
        BackendSelector.setBackend("gpu-evil");
        assertSame(cpuInstance, BackendSelector.get());

        // Opt-in flag absent -> selector ignores name and keeps CpuThreadBackend.
        System.clearProperty("numja.backend.allow");
        BackendSelector.setBackend("cpu-thread");
        assertSame(cpuInstance, BackendSelector.get());
    }

    @Test
    public void cpuBackendNumericallyEquivalent() {
        final int n = 1000;
        Random rng = new java.util.Random(42L);
        double[] a = new double[n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            a[i] = rng.nextDouble();
            b[i] = rng.nextDouble();
        }

        // elementwiseBinary: same seed, same op, both below THRESHOLD -> sequential branch
        // on both paths. Expect bit-identical double[] output.
        double[] cpuOut = new double[n];
        double[] parOut = new double[n];
        ComputeBackend cpu = CpuThreadBackend.getInstance();
        cpu.elementwiseBinary(a, b, cpuOut, Double::sum);
        ParallelOps.elementwiseBinary(a, b, parOut, Double::sum);
        for (int i = 0; i < n; i++) {
            assertEquals("elementwiseBinary index " + i, parOut[i], cpuOut[i], 0.0);
        }

        // sum: same input, same sequential branch -> exact double equality.
        assertEquals(ParallelOps.sum(a), cpu.sum(a), 0.0);
    }
}
