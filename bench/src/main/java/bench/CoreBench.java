package bench;

import numja.NumJa;
import numja.core.NDArray;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for core NumJa ops: matmul, elementwise add/mul, sum/mean.
 * Datasets are pre-allocated in setup so only the op itself is measured.
 *
 * One state class per input family (MatState / ElemState / ReduceState) so each
 * benchmark only runs its own @Param values — no cartesian product of unrelated params.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class CoreBench {

    private static NDArray randomMat(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[][] a2d = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) a2d[i][j] = rng.nextDouble();
        return new NDArray(a2d);
    }

    private static NDArray randomVec(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[] a1 = new double[n];
        for (int i = 0; i < n; i++) a1[i] = rng.nextDouble();
        return new NDArray(a1);
    }

    // ---------------- matmul ----------------

    @State(Scope.Thread)
    public static class MatState {
        @Param({"256", "1024"})
        public int n;

        NDArray a;
        NDArray b;

        @Setup(Level.Trial)
        public void setUp() {
            // Deterministic seeds so baseline numbers are reproducible.
            a = randomMat(n, 42L);
            b = randomMat(n, 43L);
        }
    }

    @Benchmark
    public void matmul_NxN(MatState s, Blackhole bh) {
        bh.consume(NumJa.matmul(s.a, s.b));
    }

    // ---------------- elementwise ----------------

    @State(Scope.Thread)
    public static class ElemState {
        @Param({"1000000", "10000000"})
        public int n;

        NDArray a;
        NDArray b;

        @Setup(Level.Trial)
        public void setUp() {
            a = randomVec(n, 44L);
            b = randomVec(n, 45L);
        }
    }

    @Benchmark
    public void add_elementwise(ElemState s, Blackhole bh) {
        bh.consume(NumJa.add(s.a, s.b));
    }

    @Benchmark
    public void multiply_elementwise(ElemState s, Blackhole bh) {
        bh.consume(NumJa.multiply(s.a, s.b));
    }

    // ---------------- reduce ----------------

    @State(Scope.Thread)
    public static class ReduceState {
        NDArray a;

        @Setup(Level.Trial)
        public void setUp() {
            a = randomVec(10_000_000, 46L);
        }
    }

    @Benchmark
    public void sum_reduce(ReduceState s, Blackhole bh) {
        bh.consume(NumJa.sum(s.a));
    }

    @Benchmark
    public void mean_reduce(ReduceState s, Blackhole bh) {
        bh.consume(NumJa.mean(s.a));
    }
}
