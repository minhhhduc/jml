package sklearn.model_selection;

import numja.core.NDArray;
import sklearn.utils.ParallelUtils;

import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.function.Supplier;

/**
 * Exhaustive search over specified parameter values for an estimator.
 */
public class GridSearchCV {
    private Supplier<Object> estimatorSupplier;
    private Map<String, Object[]> paramGrid;
    private int cv;
    private int n_jobs;
    private String mode; // "classifier" or "regressor"

    private Object bestEstimator;
    private Map<String, Object> bestParams;
    private double bestScore = -Double.MAX_VALUE;

    public GridSearchCV(Supplier<Object> estimatorSupplier, Map<String, Object[]> paramGrid, int cv, String mode, int n_jobs) {
        this.estimatorSupplier = estimatorSupplier;
        this.paramGrid = paramGrid;
        this.cv = cv;
        this.mode = mode;
        this.n_jobs = n_jobs;
    }

    public GridSearchCV fit(NDArray X, Object y) {
        List<Map<String, Object>> combinations = generateCombinations(paramGrid);

        List<Callable<GridResult>> tasks = new ArrayList<>();
        for (Map<String, Object> params : combinations) {
            tasks.add(() -> {
                double score = evaluateParams(params, X, y);
                return new GridResult(params, score);
            });
        }

        List<GridResult> results = ParallelUtils.execute(tasks, n_jobs);

        for (GridResult result : results) {
            if (result.score > bestScore) {
                bestScore = result.score;
                bestParams = result.params;
            }
        }

        // Fit the best estimator on the full dataset
        bestEstimator = estimatorSupplier.get();
        setParameters(bestEstimator, bestParams);
        try {
            if (y instanceof int[]) {
                bestEstimator.getClass().getMethod("fit", NDArray.class, int[].class).invoke(bestEstimator, X, y);
            } else {
                bestEstimator.getClass().getMethod("fit", NDArray.class, double[].class).invoke(bestEstimator, X, y);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to fit best estimator", e);
        }

        return this;
    }

    private double evaluateParams(Map<String, Object> params, NDArray X, Object y) {
        try {
            if (mode.equals("classifier")) {
                ModelSelection.Estimator est = new ModelSelection.Estimator() {
                    Object inst = estimatorSupplier.get();
                    { setParameters(inst, params); }
                    public void fit(NDArray X_train, int[] y_train) {
                        try { inst.getClass().getMethod("fit", NDArray.class, int[].class).invoke(inst, X_train, y_train); }
                        catch (Exception e) { throw new RuntimeException(e); }
                    }
                    public double score(NDArray X_test, int[] y_test) {
                        try { return (double) inst.getClass().getMethod("score", NDArray.class, int[].class).invoke(inst, X_test, y_test); }
                        catch (Exception e) { throw new RuntimeException(e); }
                    }
                };
                double[] scores = ModelSelection.crossValScore(est, X, (int[]) y, cv);
                double mean = 0; for (double s : scores) mean += s; return mean / cv;
            } else {
                ModelSelection.EstimatorRegression est = new ModelSelection.EstimatorRegression() {
                    Object inst = estimatorSupplier.get();
                    { setParameters(inst, params); }
                    public void fit(NDArray X_train, double[] y_train) {
                        try { inst.getClass().getMethod("fit", NDArray.class, double[].class).invoke(inst, X_train, y_train); }
                        catch (Exception e) { throw new RuntimeException(e); }
                    }
                    public double score(NDArray X_test, double[] y_test) {
                        try { return (double) inst.getClass().getMethod("score", NDArray.class, double[].class).invoke(inst, X_test, y_test); }
                        catch (Exception e) { throw new RuntimeException(e); }
                    }
                };
                double[] scores = ModelSelection.crossValScore(est, X, (double[]) y, cv);
                double mean = 0; for (double s : scores) mean += s; return mean / cv;
            }
        } catch (Exception e) {
            throw new RuntimeException("Error evaluating parameters", e);
        }
    }

    private void setParameters(Object estimator, Map<String, Object> params) {
        Class<?> clazz = estimator.getClass();
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            try {
                Field field = clazz.getDeclaredField(entry.getKey());
                field.setAccessible(true);
                field.set(estimator, entry.getValue());
            } catch (NoSuchFieldException e) {
                throw new RuntimeException("Parameter " + entry.getKey() + " not found in " + clazz.getSimpleName());
            } catch (IllegalAccessException e) {
                throw new RuntimeException("Cannot set parameter " + entry.getKey(), e);
            }
        }
    }

    private List<Map<String, Object>> generateCombinations(Map<String, Object[]> grid) {
        List<Map<String, Object>> result = new ArrayList<>();
        result.add(new HashMap<>());

        for (Map.Entry<String, Object[]> entry : grid.entrySet()) {
            String key = entry.getKey();
            Object[] values = entry.getValue();
            List<Map<String, Object>> newResult = new ArrayList<>();
            for (Map<String, Object> map : result) {
                for (Object value : values) {
                    Map<String, Object> newMap = new HashMap<>(map);
                    newMap.put(key, value);
                    newResult.add(newMap);
                }
            }
            result = newResult;
        }
        return result;
    }

    public Object getBestEstimator() { return bestEstimator; }
    public Map<String, Object> getBestParams() { return bestParams; }
    public double getBestScore() { return bestScore; }

    private static class GridResult {
        Map<String, Object> params;
        double score;
        GridResult(Map<String, Object> params, double score) {
            this.params = params;
            this.score = score;
        }
    }
}
