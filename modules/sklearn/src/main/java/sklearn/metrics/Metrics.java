package sklearn.metrics;

/**
 * Model evaluation metrics for classification and regression
 * Supports both binary and multi-class evaluation
 */
public class Metrics {

    // ===== CLASSIFICATION METRICS =====

    public static double accuracyScore(int[] y_true, int[] y_pred) {
        checkLength(y_true, y_pred);
        int correct = 0;
        for (int i = 0; i < y_true.length; i++) if (y_true[i] == y_pred[i]) correct++;
        return (double) correct / y_true.length;
    }

    /** Precision - binary (positive class = 1) */
    public static double precisionScore(int[] y_true, int[] y_pred) {
        int tp = 0, fp = 0;
        for (int i = 0; i < y_true.length; i++) {
            if (y_pred[i] == 1) { if (y_true[i] == 1) tp++; else fp++; }
        }
        return (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
    }

    /** Recall - binary */
    public static double recallScore(int[] y_true, int[] y_pred) {
        int tp = 0, fn = 0;
        for (int i = 0; i < y_true.length; i++) {
            if (y_true[i] == 1) { if (y_pred[i] == 1) tp++; else fn++; }
        }
        return (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
    }

    /** F1 score - binary */
    public static double f1Score(int[] y_true, int[] y_pred) {
        double p = precisionScore(y_true, y_pred);
        double r = recallScore(y_true, y_pred);
        return (p + r) == 0 ? 0 : 2 * p * r / (p + r);
    }

    /** Multi-class confusion matrix (n_classes × n_classes) */
    public static int[][] confusionMatrix(int[] y_true, int[] y_pred) {
        int maxLabel = 0;
        for (int l : y_true) if (l > maxLabel) maxLabel = l;
        for (int l : y_pred) if (l > maxLabel) maxLabel = l;
        int n = maxLabel + 1;

        int[][] cm = new int[n][n];
        for (int i = 0; i < y_true.length; i++) cm[y_true[i]][y_pred[i]]++;
        return cm;
    }

    /** Precision per class */
    public static double[] precisionPerClass(int[] y_true, int[] y_pred) {
        int[][] cm = confusionMatrix(y_true, y_pred);
        int n = cm.length;
        double[] prec = new double[n];
        for (int c = 0; c < n; c++) {
            int tp = cm[c][c]; int colSum = 0;
            for (int r = 0; r < n; r++) colSum += cm[r][c];
            prec[c] = colSum == 0 ? 0 : (double) tp / colSum;
        }
        return prec;
    }

    /** Recall per class */
    public static double[] recallPerClass(int[] y_true, int[] y_pred) {
        int[][] cm = confusionMatrix(y_true, y_pred);
        int n = cm.length;
        double[] rec = new double[n];
        for (int c = 0; c < n; c++) {
            int tp = cm[c][c]; int rowSum = 0;
            for (int j = 0; j < n; j++) rowSum += cm[c][j];
            rec[c] = rowSum == 0 ? 0 : (double) tp / rowSum;
        }
        return rec;
    }

    /** F1 per class */
    public static double[] f1PerClass(int[] y_true, int[] y_pred) {
        double[] p = precisionPerClass(y_true, y_pred);
        double[] r = recallPerClass(y_true, y_pred);
        double[] f1 = new double[p.length];
        for (int c = 0; c < p.length; c++) {
            f1[c] = (p[c] + r[c]) == 0 ? 0 : 2 * p[c] * r[c] / (p[c] + r[c]);
        }
        return f1;
    }

    /** Classification report as formatted string */
    public static String classificationReport(int[] y_true, int[] y_pred) {
        double[] prec = precisionPerClass(y_true, y_pred);
        double[] rec = recallPerClass(y_true, y_pred);
        double[] f1 = f1PerClass(y_true, y_pred);
        int[][] cm = confusionMatrix(y_true, y_pred);
        int n = cm.length;

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-12s %10s %10s %10s %10s\n", "", "precision", "recall", "f1-score", "support"));
        sb.append("\n");

        int totalSupport = 0;
        double weightedP = 0, weightedR = 0, weightedF = 0;

        for (int c = 0; c < n; c++) {
            int support = 0;
            for (int j = 0; j < n; j++) support += cm[c][j];
            totalSupport += support;
            weightedP += prec[c] * support;
            weightedR += rec[c] * support;
            weightedF += f1[c] * support;

            sb.append(String.format("%-12d %10.4f %10.4f %10.4f %10d\n",
                c, prec[c], rec[c], f1[c], support));
        }

        sb.append("\n");
        sb.append(String.format("%-12s %10.4f %10.4f %10.4f %10d\n", "accuracy",
            accuracyScore(y_true, y_pred), accuracyScore(y_true, y_pred),
            accuracyScore(y_true, y_pred), totalSupport));

        if (totalSupport > 0) {
            sb.append(String.format("%-12s %10.4f %10.4f %10.4f %10d\n", "weighted avg",
                weightedP / totalSupport, weightedR / totalSupport, weightedF / totalSupport, totalSupport));
        }

        return sb.toString();
    }

    /** Print confusion matrix (multi-class) */
    public static void printConfusionMatrix(int[][] cm) {
        System.out.println("Confusion Matrix:");
        System.out.print("        ");
        for (int j = 0; j < cm.length; j++) System.out.printf("%-6d", j);
        System.out.println();
        for (int i = 0; i < cm.length; i++) {
            System.out.printf("%-8d", i);
            for (int j = 0; j < cm[i].length; j++) System.out.printf("%-6d", cm[i][j]);
            System.out.println();
        }
    }

    // ===== REGRESSION METRICS =====

    public static double meanSquaredError(double[] y_true, double[] y_pred) {
        checkLength(y_true, y_pred);
        double sum = 0;
        for (int i = 0; i < y_true.length; i++) {
            double e = y_true[i] - y_pred[i]; sum += e * e;
        }
        return sum / y_true.length;
    }

    public static double rmse(double[] y_true, double[] y_pred) {
        return Math.sqrt(meanSquaredError(y_true, y_pred));
    }

    public static double meanAbsoluteError(double[] y_true, double[] y_pred) {
        checkLength(y_true, y_pred);
        double sum = 0;
        for (int i = 0; i < y_true.length; i++) sum += Math.abs(y_true[i] - y_pred[i]);
        return sum / y_true.length;
    }

    public static double r2Score(double[] y_true, double[] y_pred) {
        double my = 0; for (double v : y_true) my += v; my /= y_true.length;
        double sst = 0, ssr = 0;
        for (int i = 0; i < y_true.length; i++) {
            sst += (y_true[i] - my) * (y_true[i] - my);
            ssr += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        }
        return 1.0 - ssr / sst;
    }

    private static void checkLength(int[] a, int[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Length mismatch");
    }
    private static void checkLength(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Length mismatch");
    }
}
