package bench;

import pandas.DataFrame;
import pandas.Pandas;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * JMH benchmarks for pandas.DataFrame ops: read_csv + groupby mean.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Thread)
public class PandasBench {

    // Resource paths resolved from working dir. Baseline assumes `mvn` invoked
    // from repo root so relative paths resolve correctly.
    private static final Path CALIFORNIA = Paths.get("dist/datasets/california_housing.csv");
    private static final Path TITANIC = Paths.get("dist/datasets/titanic.csv");

    private DataFrame californiaDf;
    private DataFrame titanicDf;

    @Setup(Level.Trial)
    public void setUp() throws Exception {
        californiaDf = Pandas.read_csv(CALIFORNIA.toString());
        titanicDf = Pandas.read_csv(TITANIC.toString());
    }

    @Benchmark
    public void read_csv_california(Blackhole bh) throws Exception {
        bh.consume(Pandas.read_csv(CALIFORNIA.toString()));
    }

    @Benchmark
    public void groupby_mean_titanic(Blackhole bh) {
        bh.consume(titanicDf.groupby("sex").mean());
    }
}
