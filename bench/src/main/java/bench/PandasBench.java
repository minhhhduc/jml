package bench;

import pandas.DataFrame;
import pandas.Pandas;
import pandas.internal.ChunkedReadOptions;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;

/**
 * JMH benchmarks for pandas.DataFrame ops: read_csv + groupby mean + streaming variants.
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
    private ChunkedReadOptions streamingOpts;

    @Setup(Level.Trial)
    public void setUp() throws Exception {
        californiaDf = Pandas.read_csv(CALIFORNIA.toString());
        titanicDf = Pandas.read_csv(TITANIC.toString());
        streamingOpts = new ChunkedReadOptions().chunkRows(10_000);
    }

    @Benchmark
    public void read_csv_california(Blackhole bh) throws Exception {
        bh.consume(Pandas.read_csv(CALIFORNIA.toString()));
    }

    @Benchmark
    public void groupby_mean_titanic(Blackhole bh) {
        bh.consume(titanicDf.groupby("sex").mean());
    }

    @Benchmark
    public void read_csv_streaming_california(Blackhole bh) throws Exception {
        // ponytail: streaming reader NOT explicitly closed here — handle reclaimed by GC.
        // Add try-with-resources if this benchmark shows handle leak in process explorer.
        Iterator<DataFrame> it = Pandas.read_csv_streaming(CALIFORNIA.toString(), streamingOpts);
        int totalRows = 0;
        while (it.hasNext()) {
            DataFrame chunk = it.next();
            totalRows += chunk.shape()[0];
        }
        bh.consume(totalRows);
    }

    @Benchmark
    public void streaming_groupby_sum_titanic(Blackhole bh) throws Exception {
        Iterator<DataFrame> it = Pandas.read_csv_streaming(TITANIC.toString(), streamingOpts);
        DataFrame result = pandas.internal.RunningGroupAggregator.sum(it, "sex", "fare");
        bh.consume(result);
    }
}
