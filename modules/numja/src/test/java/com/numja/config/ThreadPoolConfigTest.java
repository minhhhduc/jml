package numja.config;

import org.junit.After;
import org.junit.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Regression tests for {@link ThreadPoolConfig#getForkJoinPool()} (mitigates T-2-02 pool exhaustion).
 */
public class ThreadPoolConfigTest {

    @After
    public void restoreDefaults() {
        ThreadPoolConfig.getInstance().resetToDefaults();
    }

    @Test
    public void getForkJoinPool_isSingleton() {
        ThreadPoolConfig cfg = ThreadPoolConfig.getInstance();
        ForkJoinPool first = cfg.getForkJoinPool();
        ForkJoinPool second = cfg.getForkJoinPool();
        assertSame("getForkJoinPool() must return the same instance on repeated calls", first, second);
    }

    @Test
    public void getForkJoinPool_sizedToCurrentThreads() {
        ThreadPoolConfig cfg = ThreadPoolConfig.getInstance();
        ForkJoinPool pool = cfg.getForkJoinPool();
        assertEquals("FJP parallelism must equal currentThreads", cfg.getCurrentThreads(), pool.getParallelism());
    }

    @Test
    public void getForkJoinPool_survivesSetThreads() {
        ThreadPoolConfig cfg = ThreadPoolConfig.getInstance();
        ForkJoinPool original = cfg.getForkJoinPool();
        int originalParallelism = original.getParallelism();

        cfg.setThreads(1);
        ForkJoinPool afterSetThreads = cfg.getForkJoinPool();

        assertSame("setThreads must not recreate the FJP (matches existing setThreads semantics)", original, afterSetThreads);
        assertTrue("FJP parallelism must be >= 1 after setThreads(1)", afterSetThreads.getParallelism() >= 1);
        assertEquals("FJP parallelism must remain unchanged after setThreads(1) (documented limitation)", originalParallelism, afterSetThreads.getParallelism());
    }
}
