package bench;

import numja.NumJa;
import numja.core.NDArray;
import pandas.DataFrame;
import pandas.Pandas;
import pandas.Series;
import sklearn.cluster.KMeans;
import sklearn.linear_model.LinearRegression;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * JMH benchmarks for sklearn ops: LinearRegression fit/predict on
 * california_housing, KMeans fit on iris.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Thread)
public class SklearnBench {

    private static final Path CALIFORNIA = Paths.get("dist/datasets/california_housing.csv");
    private static final Path IRIS = Paths.get("dist/datasets/iris.csv");

    private NDArray californiaX;
    private double[] californiaY;
    private NDArray irisX;

    @Setup(Level.Trial)
    public void setUp() throws Exception {
        DataFrame calDf = Pandas.read_csv(CALIFORNIA.toString());
        // All numeric columns except the last (target = MedHouseVal)
        String[] cols = calDf.columns();
        int lastIdx = cols.length - 1;
        int n = calDf.shape()[0];

        double[][] xData = new double[n][lastIdx];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < lastIdx; j++) {
                xData[i][j] = calDf.getColumn(cols[j]).get(i);
            }
        }
        californiaX = new NDArray(xData);
        californiaY = new double[n];
        Series ySeries = calDf.getColumn(cols[lastIdx]);
        for (int i = 0; i < n; i++) californiaY[i] = ySeries.get(i);

        // iris: all 4 numeric columns
        DataFrame irisDf = Pandas.read_csv(IRIS.toString());
        String[] irisCols = irisDf.columns();
        int nFeat = irisCols.length - 1;
        int nIris = irisDf.shape()[0];
        double[][] ix = new double[nIris][nFeat];
        for (int i = 0; i < nIris; i++) {
            for (int j = 0; j < nFeat; j++) {
                ix[i][j] = irisDf.getColumn(irisCols[j]).get(i);
            }
        }
        irisX = new NDArray(ix);
    }

    @Benchmark
    public void linear_regression_fit(Blackhole bh) {
        bh.consume(new LinearRegression().fit(californiaX, californiaY));
    }

    @Benchmark
    public void kmeans_fit_iris(Blackhole bh) {
        bh.consume(new KMeans(3).fit(irisX));
    }
}
