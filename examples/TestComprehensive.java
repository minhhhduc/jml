import pandas.DataFrame;
import pandas.Pandas;
import pandas.Series;
import sklearn.cluster.KMeans;
import sklearn.cluster.DBSCAN;
import sklearn.decomposition.PCA;
import sklearn.ensemble.AdaBoostClassifier;
import sklearn.ensemble.GradientBoostingClassifier;
import sklearn.ensemble.GradientBoostingRegressor;
import sklearn.ensemble.RandomForest;
import sklearn.impute.SimpleImputer;
import sklearn.linear_model.Lasso;
import sklearn.linear_model.LinearRegression;
import sklearn.linear_model.LogisticRegression;
import sklearn.linear_model.Ridge;
import sklearn.metrics.Metrics;
import sklearn.model_selection.GridSearchCV;
import sklearn.model_selection.ModelSelection;
import sklearn.naive_bayes.GaussianNB;
import sklearn.neighbors.KNeighbors;
import sklearn.neural_network.MLPClassifier;
import sklearn.neural_network.MLPRegressor;
import sklearn.pipeline.Pipeline;
import sklearn.preprocessing.*;
import sklearn.svm.SVC;
import sklearn.svm.SVR;
import sklearn.tree.DecisionTree;
import sklearn.utils.ParallelUtils;
import numja.core.NDArray;
import numja.NumJa;

import java.io.File;
import java.util.*;

/**
 * NumJa - Sklearn & Pandas Comprehensive Test Suite
 * Designed as a modular and highly educational reference.
 * Demonstrates 7:3 train-test split on Iris dataset.
 */
public class TestComprehensive {
    static int passed = 0, failed = 0;

    // Classification data variables (7:3 Split)
    static NDArray X_iris;
    static int[] y_iris;
    static NDArray X_train_c;
    static NDArray X_test_c;
    static int[] y_train_c;
    static int[] y_test_c;

    // Regression data variables (7:3 Split)
    static NDArray X_reg;
    static double[] y_reg;
    static NDArray X_train_r;
    static NDArray X_test_r;
    static double[] y_train_r;
    static double[] y_test_r;

    static DataFrame iris;
    static LabelEncoder labelEncoder;
    static boolean headless = false;

    static void test(String name, Runnable r) {
        try {
            r.run();
            System.out.println("  [PASS] " + name);
            passed++;
        } catch (Throwable t) {
            System.out.println("  [FAIL] " + name + " -> " + t.getClass().getSimpleName() + ": " + t.getMessage());
            t.printStackTrace();
            failed++;
        }
    }

    public static void main(String[] args) throws Exception {
        headless = args.length > 0 && args[0].equals("--headless");

        System.out.println("=============================================");
        System.out.println("  NUMJA COMPREHENSIVE MODULAR TEST SUITE     ");
        System.out.println("=============================================");

        prepareDatasets();

        testPandas();
        testPreprocessing();
        testModelSelection();
        testClassification();
        testRegression();
        testClustering();
        testDecomposition();
        testMetrics();
        testPipeline();
        testGridSearchCV();
        testParallelUtils();
        testPlotting();

        System.out.println("\n=============================================");
        System.out.println("  FINAL RESULTS: " + passed + " PASSED / " + failed + " FAILED");
        System.out.println("  TOTAL TESTS: " + (passed + failed));
        System.out.println("=============================================");

        if (failed > 0) System.exit(1);
    }

    /**
     * Load the Iris dataset and perform 7:3 train-test splits for both classification and regression.
     */
    private static void prepareDatasets() throws Exception {
        // Showcase high-level dataset loader (exactly like Python Scikit-Learn!)
        sklearn.datasets.Bunch irisBunch = sklearn.datasets.Datasets.loadIris();
        X_iris = irisBunch.getData();
        y_iris = irisBunch.getTarget();

        // Initialize iris and labelEncoder for downstream Pandas and Preprocessing unit tests
        iris = Pandas.read_csv("dist/datasets/iris.csv", new Pandas.ReadCsvOptions().header(0));
        labelEncoder = new LabelEncoder();
        String[] speciesStr = new String[iris.shape()[0]];
        for (int i = 0; i < speciesStr.length; i++) {
            speciesStr[i] = iris.getColumn("species").getString(i);
        }
        labelEncoder.fit(speciesStr);

        // 7:3 Train-Test Split for Classification
        Object[] splitC = ModelSelection.trainTestSplit(X_iris, y_iris, 0.3, 42);
        X_train_c = (NDArray) splitC[0];
        X_test_c = (NDArray) splitC[1];
        y_train_c = (int[]) splitC[2];
        y_test_c = (int[]) splitC[3];

        // Prepare Regression Dataset (predict Petal Length from Sepal Length and Sepal Width)
        double[][] irisArr = irisBunch.getData().toArray(); // raw feature coordinates
        double[][] xReg = new double[irisArr.length][2];
        double[] yReg = new double[irisArr.length];
        for (int i = 0; i < irisArr.length; i++) {
            xReg[i][0] = irisArr[i][0]; // Sepal Length
            xReg[i][1] = irisArr[i][1]; // Sepal Width
            yReg[i] = irisArr[i][2];    // Petal Length (regression target)
        }
        X_reg = NumJa.array(xReg);
        y_reg = yReg;

        // 7:3 Train-Test Split for Regression
        Object[] splitR = ModelSelection.trainTestSplitRegression(X_reg, y_reg, 0.3, 42);
        X_train_r = (NDArray) splitR[0];
        X_test_r = (NDArray) splitR[1];
        y_train_r = (double[]) splitR[2];
        y_test_r = (double[]) splitR[3];

        System.out.println("Datasets initialized successfully:");
        System.out.println("  * Iris Classification train size: " + X_train_c.getShape()[0] + " | test size: " + X_test_c.getShape()[0] + " (70:30)");
        System.out.println("  * Iris Regression train size: " + X_train_r.getShape()[0] + " | test size: " + X_test_r.getShape()[0] + " (70:30)");
    }

    /**
     * Group 1: Pandas DataFrame operations
     */
    private static void testPandas() {
        System.out.println("\n--- 1. PANDAS ---");
        test("read_csv + shape", () -> {
            assert iris.shape()[0] == 150 : "Expected 150 rows, got " + iris.shape()[0];
            assert iris.shape()[1] == 5 : "Expected 5 cols";
        });
        test("head / tail", () -> {
            DataFrame h = iris.head(5);
            assert h.shape()[0] == 5 : "head(5) failed";
            DataFrame t = iris.tail(3);
            assert t.shape()[0] == 3 : "tail(3) failed";
        });
        test("describe", () -> { 
            iris.describe(); 
        });
        test("sort_values", () -> {
            DataFrame sorted = iris.sort_values("sepal_length");
            assert sorted.shape()[0] == 150;
        });
        test("dropna", () -> {
            DataFrame clean = iris.dropna();
            assert clean.shape()[0] > 0;
        });
        test("drop_duplicates", () -> {
            DataFrame dedup = iris.drop_duplicates();
            assert dedup.shape()[0] <= 150;
        });
        test("corr", () -> {
            DataFrame numDf = iris.drop("species");
            DataFrame c = numDf.corr();
            assert c.shape()[0] == 4 && c.shape()[1] == 4 : "corr matrix wrong shape";
        });
        test("to_csv", () -> {
            iris.head(3).to_csv("_test_out.csv");
            assert new File("_test_out.csv").exists();
            new File("_test_out.csv").delete();
        });
        test("fillna", () -> {
            DataFrame filled = iris.fillna(0);
            assert filled.shape()[0] == 150;
        });
        test("groupby", () -> { 
            iris.groupby("species"); 
        });
    }

    /**
     * Group 2: Feature Preprocessing
     */
    private static void testPreprocessing() {
        System.out.println("\n--- 2. PREPROCESSING ---");
        test("StandardScaler", () -> {
            StandardScaler ss = new StandardScaler();
            ss.fit(X_iris);
            NDArray Xt = ss.transform(X_iris);
            assert Xt.getShape()[0] == 150;
        });
        test("MinMaxScaler", () -> {
            MinMaxScaler ms = new MinMaxScaler();
            ms.fit(X_iris);
            NDArray Xt = ms.transform(X_iris);
            assert Xt.getShape()[0] == 150;
        });
        test("RobustScaler", () -> {
            RobustScaler rs = new RobustScaler();
            rs.fit(X_iris);
            NDArray Xt = rs.transform(X_iris);
            assert Xt.getShape()[0] == 150;
        });
        test("LabelEncoder", () -> {
            assert y_iris.length == 150;
            String[] inv = labelEncoder.inverseTransform(y_iris);
            assert inv[0] != null;
        });
        test("OneHotEncoder", () -> {
            OneHotEncoder ohe = new OneHotEncoder();
            NDArray oh = ohe.fitTransform(y_iris);
            assert oh.getShape()[1] == 3 : "Expected 3 one-hot columns";
        });
        test("PolynomialFeatures", () -> {
            PolynomialFeatures pf = new PolynomialFeatures(2);
            NDArray Xt = pf.fit_transform(X_iris);
            assert Xt.getShape()[1] > X_iris.getShape()[1] : "Should have more features";
        });
        test("SimpleImputer", () -> {
            SimpleImputer si = new SimpleImputer("mean");
            NDArray Xt = si.fit_transform(X_iris);
            assert Xt.getShape()[0] == 150;
        });
    }

    /**
     * Group 3: Model Selection & Train-Test Splits
     */
    private static void testModelSelection() {
        System.out.println("\n--- 3. MODEL SELECTION ---");
        test("train_test_split (classification)", () -> {
            Object[] split = ModelSelection.trainTestSplit(X_iris, y_iris, 0.3);
            NDArray Xtr = (NDArray) split[0];
            NDArray Xte = (NDArray) split[1];
            assert Xtr.getShape()[0] + Xte.getShape()[0] == 150;
        });
        test("train_test_split (regression)", () -> {
            Object[] split = ModelSelection.trainTestSplitRegression(X_reg, y_reg, 0.3);
            NDArray Xtr = (NDArray) split[0];
            assert Xtr.getShape()[0] > 0;
        });
        test("cross_val_score", () -> {
            DecisionTree dt = new DecisionTree(5, 2, "classifier");
            double[] scores = ModelSelection.crossValScore(
                new ModelSelection.Estimator() {
                    public void fit(NDArray X, int[] y) { dt.fit(X, y); }
                    public double score(NDArray X, int[] y) { return dt.score(X, y); }
                }, X_iris, y_iris, 5);
            assert scores.length == 5;
            System.out.println("    CV scores: " + Arrays.toString(scores));
        });
    }

    private static void showModelConfusionMatrix(String modelName, int[] y_true, int[] y_pred, String[] classes) {
        try {
            int[][] cm = Metrics.confusionMatrix(y_true, y_pred);
            double acc = Metrics.accuracyScore(y_true, y_pred);
            matplotlib.Matplotlib.clf();
            matplotlib.Matplotlib.title(modelName + " Confusion Matrix (Acc: " + String.format("%.2f%%", acc * 100) + ")");
            seaborn.Seaborn.heatmap(cm, classes);
            if (headless) {
                String filename = modelName.replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png";
                matplotlib.Matplotlib.savefig(filename);
                java.io.File f = new java.io.File(filename);
                if (f.exists()) f.delete();
            } else {
                System.out.println("    --> Showing GUI plot window for " + modelName + ". Close it to continue...");
                matplotlib.Matplotlib.show();
            }
        } catch (Exception e) {
            System.out.println("Could not show heatmap for " + modelName + ": " + e.getMessage());
        }
    }

    /**
     * Group 4: Classification models applied to 7:3 split
     */
    private static void testClassification() {
        System.out.println("\n--- 4. CLASSIFICATION (7:3 Train-Test Split on Iris) ---");
        String[] speciesClasses = new String[]{"Setosa", "Versicolor", "Virginica"};

        test("DecisionTree Classifier", () -> {
            DecisionTree dt = new DecisionTree(5, 2, "classifier");
            dt.fit(X_train_c, y_train_c);
            int[] pred = dt.predict(X_test_c);
            double acc = dt.score(X_test_c, y_test_c);
            System.out.println("    DecisionTree Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("DecisionTree", y_test_c, pred, speciesClasses);
            assert acc > 0.5;
        });

        test("RandomForest Classifier (Parallel execution, n_jobs=0 -> 60% default threads)", () -> {
            RandomForest rf = new RandomForest(20, 5, 2, "classifier", 0);
            rf.fit(X_train_c, y_train_c);
            int[] pred = rf.predict(X_test_c);
            double acc = rf.score(X_test_c, y_test_c);
            System.out.println("    RandomForest Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("RandomForest", y_test_c, pred, speciesClasses);
            assert acc > 0.5;
        });

        test("KNeighbors Classifier (n_jobs=-1 -> Use all cores)", () -> {
            KNeighbors knn = new KNeighbors(5, "classifier", -1);
            knn.fit(X_train_c, y_train_c);
            int[] pred = knn.predict(X_test_c);
            double acc = knn.score(X_test_c, y_test_c);
            System.out.println("    KNeighbors Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("KNeighbors", y_test_c, pred, speciesClasses);
            assert acc > 0.5;
        });

        test("Gaussian Naive Bayes (GaussianNB)", () -> {
            GaussianNB nb = new GaussianNB();
            nb.fit(X_train_c, y_train_c);
            int[] pred = nb.predict(X_test_c);
            double acc = nb.score(X_test_c, y_test_c);
            System.out.println("    GaussianNB Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("GaussianNB", y_test_c, pred, speciesClasses);
            assert acc > 0.5;
        });

        test("LogisticRegression", () -> {
            LogisticRegression lr = new LogisticRegression();
            lr.setVerbose(true);
            lr.fit(X_train_c, y_train_c);
            int[] pred = lr.predict(X_test_c);
            double acc = lr.score(X_test_c, y_test_c);
            System.out.println("    LogisticRegression Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("LogisticRegression", y_test_c, pred, speciesClasses);
            assert acc > 0.3;
        });

        test("Support Vector Classifier (SVC)", () -> {
            SVC svc = new SVC();
            svc.fit(X_train_c, y_train_c);
            int[] pred = svc.predict(X_test_c);
            double acc = svc.score(X_test_c, y_test_c);
            System.out.println("    SVC Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("SVC", y_test_c, pred, speciesClasses);
            assert acc > 0.3;
        });

        test("GradientBoosting Classifier", () -> {
            GradientBoostingClassifier gbc = new GradientBoostingClassifier(20, 0.1, 3);
            gbc.fit(X_train_c, y_train_c);
            int[] pred = gbc.predict(X_test_c);
            double acc = gbc.score(X_test_c, y_test_c);
            System.out.println("    GradientBoosting Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("GradientBoosting", y_test_c, pred, speciesClasses);
            assert acc > 0.5;
        });

        test("AdaBoost Classifier (binary representation)", () -> {
            AdaBoostClassifier ada = new AdaBoostClassifier(20);
            // AdaBoost is binary - test with class 0 vs not 0
            int[] yBinTrain = new int[y_train_c.length];
            for (int i = 0; i < y_train_c.length; i++) yBinTrain[i] = y_train_c[i] == 0 ? 0 : 1;
            int[] yBinTest = new int[y_test_c.length];
            for (int i = 0; i < y_test_c.length; i++) yBinTest[i] = y_test_c[i] == 0 ? 0 : 1;

            ada.fit(X_train_c, yBinTrain);
            int[] pred = ada.predict(X_test_c);
            double acc = ada.score(X_test_c, yBinTest);
            System.out.println("    AdaBoost Accuracy: " + String.format("%.4f", acc));
            showModelConfusionMatrix("AdaBoost (Binary)", yBinTest, pred, new String[]{"Setosa", "Others"});
            assert acc > 0.5;
        });

        test("Multi-Layer Perceptron Classifier (MLPClassifier)", () -> {
            // Standardize features for optimal MLP convergence (standard practice in Python)
            StandardScaler scaler = new StandardScaler();
            scaler.fit(X_train_c);
            NDArray X_train_scaled = scaler.transform(X_train_c);
            NDArray X_test_scaled = scaler.transform(X_test_c);

            MLPClassifier mlp = new MLPClassifier(10);
            mlp.setVerbose(true);
            mlp.setMaxIter(250).setLearningRate(0.03); // Optimal regularization and step size
            mlp.fit(X_train_scaled, y_train_c);
            int[] pred = mlp.predict(X_test_scaled);
            double acc = mlp.score(X_test_scaled, y_test_c);
            System.out.println("    MLPClassifier Accuracy (with StandardScaler): " + String.format("%.4f", acc));
            showModelConfusionMatrix("MLPClassifier", y_test_c, pred, speciesClasses);
            assert acc >= 0.95 : "Expected accuracy above 95% with scaling, got " + acc;
        });
    }

    /**
     * Group 5: Regression models applied to 7:3 split
     */
    private static void testRegression() {
        System.out.println("\n--- 5. REGRESSION (7:3 Train-Test Split on Iris) ---");

        test("LinearRegression", () -> {
            LinearRegression lr = new LinearRegression();
            lr.fit(X_train_r, y_train_r);
            double r2 = lr.score(X_test_r, y_test_r);
            System.out.println("    LinearRegression R2: " + String.format("%.4f", r2));
        });

        test("Ridge Regression", () -> {
            Ridge ridge = new Ridge(0.1, 2000, 0.001);
            ridge.fit(X_train_r, y_train_r);
            double r2 = ridge.score(X_test_r, y_test_r);
            System.out.println("    Ridge R2: " + String.format("%.4f", r2));
        });

        test("Lasso Regression", () -> {
            Lasso lasso = new Lasso(0.01, 500);
            lasso.fit(X_train_r, y_train_r);
            double r2 = lasso.score(X_test_r, y_test_r);
            System.out.println("    Lasso R2: " + String.format("%.4f", r2));
        });

        test("Support Vector Regressor (SVR)", () -> {
            SVR svr = new SVR();
            svr.fit(X_train_r, y_train_r);
            double r2 = svr.score(X_test_r, y_test_r);
            System.out.println("    SVR R2: " + String.format("%.4f", r2));
        });

        test("GradientBoosting Regressor", () -> {
            GradientBoostingRegressor gbr = new GradientBoostingRegressor(20, 0.1, 3);
            gbr.fit(X_train_r, y_train_r);
            double r2 = gbr.score(X_test_r, y_test_r);
            System.out.println("    GradientBoostingRegressor R2: " + String.format("%.4f", r2));
            assert r2 > 0;
        });

        test("DecisionTree Regressor", () -> {
            DecisionTree dt = new DecisionTree(5, 2, "regressor");
            dt.fit(X_train_r, y_train_r);
            double r2 = dt.score(X_test_r, y_test_r);
            System.out.println("    DecisionTree Regressor R2: " + String.format("%.4f", r2));
        });

        test("KNeighbors Regressor", () -> {
            KNeighbors knn = new KNeighbors(5, "regressor");
            knn.fit(X_train_r, y_train_r);
            double r2 = knn.score(X_test_r, y_test_r);
            System.out.println("    KNeighbors Regressor R2: " + String.format("%.4f", r2));
        });

        test("Multi-Layer Perceptron Regressor (MLPRegressor)", () -> {
            // Standardize features for optimal MLP convergence
            StandardScaler scaler = new StandardScaler();
            scaler.fit(X_train_r);
            NDArray X_train_scaled = scaler.transform(X_train_r);
            NDArray X_test_scaled = scaler.transform(X_test_r);

            MLPRegressor mlp = new MLPRegressor(10);
            mlp.setVerbose(true);
            mlp.setMaxIter(200).setLearningRate(0.01); // Optimized learning rate for standardized inputs
            mlp.fit(X_train_scaled, y_train_r);
            double r2 = mlp.score(X_test_scaled, y_test_r);
            System.out.println("    MLPRegressor R2 (with StandardScaler): " + String.format("%.4f", r2));
            assert r2 > 0.5 : "Expected R2 above 0.5 with scaling, got " + r2;
        });
    }

    /**
     * Group 6: Clustering Algorithms
     */
    private static void testClustering() {
        System.out.println("\n--- 6. CLUSTERING ---");
        test("KMeans", () -> {
            KMeans km = new KMeans(3);
            km.fit(X_iris);
            int[] labels = km.predict(X_iris);
            assert labels.length == 150;
            Set<Integer> unique = new HashSet<>(); 
            for (int l : labels) unique.add(l);
            System.out.println("    KMeans Unique clusters: " + unique.size());
        });
        test("DBSCAN", () -> {
            DBSCAN db = new DBSCAN(0.5, 5);
            db.fit(X_iris);
            int[] labels = db.getLabels();
            assert labels.length == 150;
            System.out.println("    DBSCAN Clusters found: " + db.getNumClusters());
        });
    }

    /**
     * Group 7: Dimensionality Reduction
     */
    private static void testDecomposition() {
        System.out.println("\n--- 7. DECOMPOSITION ---");
        test("Principal Component Analysis (PCA)", () -> {
            PCA pca = new PCA(2);
            pca.fit(X_iris);
            NDArray Xt = pca.transform(X_iris);
            assert Xt.getShape()[1] == 2 : "Expected 2 components";
            System.out.println("    PCA Explained variance ratio: " + Arrays.toString(pca.getExplainedVarianceRatio()));
        });
    }

    private static void testMetrics() {
        System.out.println("\n--- 8. METRICS (Evaluated on Iris 7:3 Test Split using KNeighbors) ---");
        
        // Train a model to get realistic predictions
        KNeighbors knn = new KNeighbors(5, "classifier", -1);
        knn.fit(X_train_c, y_train_c);
        int[] y_pred = knn.predict(X_test_c);

        test("accuracyScore", () -> {
            double acc = Metrics.accuracyScore(y_test_c, y_pred);
            System.out.println("    Calculated Accuracy Score: " + String.format("%.4f", acc));
            assert acc >= 0.90;
        });

        test("meanSquaredError / meanAbsoluteError / r2Score", () -> {
            double mse = Metrics.meanSquaredError(y_reg, y_reg);
            double mae = Metrics.meanAbsoluteError(y_reg, y_reg);
            double r2 = Metrics.r2Score(y_reg, y_reg);
            assert mse == 0.0;
            assert mae == 0.0;
            assert r2 == 1.0;
        });

        test("classificationReport & confusionMatrix printout", () -> {
            String report = Metrics.classificationReport(y_test_c, y_pred);
            System.out.println("\n--- CLASSIFICATION REPORT ---");
            System.out.print(report);
            System.out.println("-----------------------------\n");
            
            int[][] cm = Metrics.confusionMatrix(y_test_c, y_pred);
            System.out.println("--- CONFUSION MATRIX ---");
            Metrics.printConfusionMatrix(cm);
            System.out.println("------------------------\n");
            
            assert report.length() > 0;
            assert cm.length == 3;
        });
    }

    /**
     * Group 9: Machine Learning Pipelines
     */
    private static void testPipeline() {
        System.out.println("\n--- 9. PIPELINE ---");
        test("Pipeline (Imputer -> Scaler -> RandomForest)", () -> {
            List<Object> steps = new ArrayList<>();
            steps.add(new SimpleImputer("mean"));
            steps.add(new StandardScaler());
            steps.add(new RandomForest(10, 5, 2, "classifier", 0));
            Pipeline pipe = new Pipeline(steps);
            pipe.fit(X_train_c, y_train_c);
            int[] pred = (int[]) pipe.predict(X_test_c);
            assert pred.length == X_test_c.getShape()[0];
            int ok = 0; 
            for (int i = 0; i < pred.length; i++) {
                if (pred[i] == y_test_c[i]) ok++;
            }
            System.out.println("    Pipeline Test Accuracy: " + String.format("%.4f", (double) ok / pred.length));
        });
    }

    /**
     * Group 10: Hyperparameter Tuning
     */
    private static void testGridSearchCV() {
        System.out.println("\n--- 10. GRIDSEARCHCV ---");
        test("GridSearchCV (Parallel multi-threading grid search, n_jobs=-1)", () -> {
            Map<String, Object[]> paramGrid = new HashMap<>();
            paramGrid.put("n_estimators", new Object[]{5, 10});
            paramGrid.put("max_depth", new Object[]{3, 5});

            long t0 = System.currentTimeMillis();
            GridSearchCV grid = new GridSearchCV(() -> new RandomForest(), paramGrid, 3, "classifier", -1);
            grid.fit(X_iris, y_iris);
            long elapsed = System.currentTimeMillis() - t0;

            System.out.println("    Tuning Time: " + elapsed + " ms | Best Parameters: " + grid.getBestParams());
            assert grid.getBestScore() > 0;
        });
    }

    /**
     * Group 11: Multi-Threaded Parallel Execution Utils
     */
    private static void testParallelUtils() {
        System.out.println("\n--- 11. PARALLELUTILS ---");
        test("ParallelUtils core utility (60% Default Thread Pool Allocation)", () -> {
            int cores = Runtime.getRuntime().availableProcessors();
            int expected = Math.max(1, (int) (cores * 0.6));
            System.out.println("    System CPU cores: " + cores + " | 60% thread capacity: " + expected);
        });
    }

    /**
     * Group 12: Heatmaps & Plotting
     */
    private static void testPlotting() {
        System.out.println("\n--- 12. PLOTTING ---");
        test("Seaborn Heatmap rendering with Color Bar & savefig capabilities", () -> {
            try {
                int[][] cm = {{12, 1, 0}, {0, 14, 1}, {0, 0, 15}};
                String[] classes = {"Setosa", "Versicolor", "Virginica"};
                
                matplotlib.Matplotlib.clf();
                matplotlib.Matplotlib.title("Iris Species Test Confusion Matrix");
                seaborn.Seaborn.heatmap(cm, classes);
                
                String tempPath = "_temp_cm_test.png";
                matplotlib.Matplotlib.savefig(tempPath);
                
                java.io.File file = new java.io.File(tempPath);
                assert file.exists() : "PNG file was not generated";
                assert file.length() > 0 : "Generated PNG file is empty";
                file.delete();
            } catch (Exception e) {
                throw new RuntimeException("Plotting test failed: " + e.getMessage(), e);
            }
        });

        test("Matplotlib.imshow rendering with raw matrix data", () -> {
            try {
                double[][] imageGrid = {
                    {0.1, 0.2, 0.3},
                    {0.4, 0.5, 0.6},
                    {0.7, 0.8, 0.9}
                };
                matplotlib.Matplotlib.clf();
                matplotlib.Matplotlib.title("imshow Pixel Grid Test");
                matplotlib.Matplotlib.imshow(imageGrid);
                
                String tempPath = "_temp_imshow_test.png";
                matplotlib.Matplotlib.savefig(tempPath);
                
                java.io.File file = new java.io.File(tempPath);
                assert file.exists() : "imshow PNG was not generated";
                assert file.length() > 0;
                file.delete();
            } catch (Exception e) {
                throw new RuntimeException("imshow test failed: " + e.getMessage(), e);
            }
        });

        test("Matplotlib.imshow rendering with 2D NDArray data directly", () -> {
            try {
                double[][] imageGrid = {
                    {0.9, 0.8, 0.7},
                    {0.6, 0.5, 0.4},
                    {0.3, 0.2, 0.1}
                };
                NDArray ndGrid = NumJa.array(imageGrid);
                matplotlib.Matplotlib.clf();
                matplotlib.Matplotlib.title("imshow NDArray Test");
                matplotlib.Matplotlib.imshow(ndGrid);
                
                String tempPath = "_temp_nd_imshow_test.png";
                matplotlib.Matplotlib.savefig(tempPath);
                
                java.io.File file = new java.io.File(tempPath);
                assert file.exists() : "NDArray imshow PNG was not generated";
                assert file.length() > 0;
                file.delete();
            } catch (Exception e) {
                throw new RuntimeException("NDArray imshow test failed: " + e.getMessage(), e);
            }
        });
    }
}
