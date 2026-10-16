package sklearn.pipeline;

import numja.core.NDArray;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
 * Pipeline of transforms with a final estimator.
 */
public class Pipeline {
    private List<Object> steps;

    public Pipeline(List<Object> steps) {
        this.steps = new ArrayList<>(steps);
    }

    public Pipeline fit(NDArray X, Object y) {
        NDArray Xt = X;
        for (int i = 0; i < steps.size() - 1; i++) {
            Object step = steps.get(i);
            try {
                // Duck-typing fit_transform
                Method fitTransform = step.getClass().getMethod("fit_transform", NDArray.class);
                Xt = (NDArray) fitTransform.invoke(step, Xt);
            } catch (NoSuchMethodException e) {
                try {
                    // Fallback to fit() then transform()
                    Method fit = step.getClass().getMethod("fit", NDArray.class);
                    fit.invoke(step, Xt);
                    Method transform = step.getClass().getMethod("transform", NDArray.class);
                    Xt = (NDArray) transform.invoke(step, Xt);
                } catch (Exception ex) {
                    throw new RuntimeException("Step " + i + " must implement fit_transform or fit+transform", ex);
                }
            } catch (Exception e) {
                throw new RuntimeException("Error executing step " + i, e);
            }
        }

        Object finalEstimator = steps.get(steps.size() - 1);
        try {
            if (y instanceof int[]) {
                Method fit = finalEstimator.getClass().getMethod("fit", NDArray.class, int[].class);
                fit.invoke(finalEstimator, Xt, y);
            } else if (y instanceof double[]) {
                Method fit = finalEstimator.getClass().getMethod("fit", NDArray.class, double[].class);
                fit.invoke(finalEstimator, Xt, y);
            } else {
                throw new IllegalArgumentException("Unsupported label type: " + y.getClass());
            }
        } catch (Exception e) {
            throw new RuntimeException("Final estimator failed to fit", e);
        }

        return this;
    }

    public Object predict(NDArray X) {
        NDArray Xt = X;
        for (int i = 0; i < steps.size() - 1; i++) {
            Object step = steps.get(i);
            try {
                Method transform = step.getClass().getMethod("transform", NDArray.class);
                Xt = (NDArray) transform.invoke(step, Xt);
            } catch (Exception e) {
                throw new RuntimeException("Error executing transform on step " + i, e);
            }
        }

        Object finalEstimator = steps.get(steps.size() - 1);
        try {
            try {
                Method predict = finalEstimator.getClass().getMethod("predict", NDArray.class);
                return predict.invoke(finalEstimator, Xt);
            } catch (NoSuchMethodException e) {
                Method predictReg = finalEstimator.getClass().getMethod("predictRegression", NDArray.class);
                return predictReg.invoke(finalEstimator, Xt);
            }
        } catch (Exception e) {
            throw new RuntimeException("Final estimator failed to predict", e);
        }
    }
}
