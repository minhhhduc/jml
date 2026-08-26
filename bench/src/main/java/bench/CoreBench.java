package bench;

import numja.NumJa;
import numja.core.NDArray;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for core NumJa ops: matmul, elementwise add/mul, sum/mean.
 * Datasets are pre-allocated in {@link #setUp()} so only the op itself is measured.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Thread)
public class CoreBench {

    // ---- matmul inputs ----
    @Param({"256", "1024"})
    public int matmulN;

    private NDArray matA;
    private NDArray matB;

    // ---- elementwise inputs ----
    @Param({"1000000", "10000000"})
    public int elemN;

    private NDArray elemA;
    private NDArray elemB;

    // ---- reduce inputs (double[10^7]) ----
    private NDArray reduceA;

    @Setup(Level.Trial)
    public void setUp() {
        // Deterministic seed so baseline numbers are reproducible.
        java.util.Random rng = new java.util.Random(42L);

        // matmul
        double[][] a2d = new double[matmulN][matmulN];
        double[][] b2d = new double[matmulN][matmulN];
        for (int i = 0; i < matmulN; i++) {
            for (int j = 0; j < matmulN; j++) {
                a2d[i][j] = rng.nextDouble();
                b2d[i][j] = rng.nextDouble();
            }
        }
        matA = new NDArray(a2d);
        matB = new NDArray(b2d);

        // elementwise
        double[] a1 = new double[elemN];
        double[] b1 = new double[elemN];
        for (int i = 0; i < elemN; i++) {
            a1[i] = rng.nextDouble();
            b1[i] = rng.nextDouble();
        }
        elemA = new NDArray(a1);
        elemB = new NDArray(b1);

        // reduce uses the larger elementwise array
        reduceA = elemN >= 10_000_000 ? elemA : new NDArray(makeArray(10_000_000, rng));
    }

    private static double[] makeArray(int n, java.util.Random rng) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.nextDouble();
        return a;
    }

    // ---------------- matmul ----------------

    @Benchmark
    public void matmul_NxN(Blackhole bh) {
        bh.consume(NumJa.matmul(matA, matB));
    }

    // ---------------- elementwise ----------------

    @Benchmark
    public void add_elementwise(Blackhole bh) {
        bh.consume(NumJa.add(elemA, elemB));
    }

    @Benchmark
    public void multiply_elementwise(Blackhole bh) {
        bh.consume(NumJa.multiply(elemA, elemB));
    }

    // ---------------- reduce ----------------

    @Benchmark
    public void sum_reduce(Blackhole bh) {
        bh.consume(NumJa.sum(reduceA));
    }

    @Benchmark
    public void mean_reduce(Blackhole bh) {
        bh.consume(NumJa.mean(reduceA));
    }
}
