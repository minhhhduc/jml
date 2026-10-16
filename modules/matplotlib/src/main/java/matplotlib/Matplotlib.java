package matplotlib;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ChartUtils;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.awt.*;
import java.awt.geom.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Matplotlib - 2D plotting library (JFreeChart wrapper)
 * Provides MATLAB-like plotting interface for Java
 */
public final class Matplotlib {

    private Matplotlib() {}

    private static JFreeChart currentChart;
    private static XYSeriesCollection currentDataset = new XYSeriesCollection();
    private static String chartTitle = "Figure 1";
    private static String xLabel = "X";
    private static String yLabel = "Y";
    private static boolean showLegend = false;
    private static List<Boolean> isScatterList = new ArrayList<>();

    // Heatmap data
    private static double[][] heatmapData;
    private static String[] heatmapLabels;
    private static String heatmapTitle = "Confusion Matrix";

    // Imshow data
    private static double[][] imshowData;
    private static String imshowTitle;

    /**
     * Clear the current figure (reset all plot state)
     */
    public static void clf() {
        currentDataset = new XYSeriesCollection();
        currentChart = null;
        isScatterList.clear();
        chartTitle = "Figure 1";
        xLabel = "X";
        yLabel = "Y";
        showLegend = false;
        heatmapData = null;
        heatmapLabels = null;
        heatmapTitle = "Confusion Matrix";
        imshowData = null;
        imshowTitle = null;
    }

    /**
     * Set the title of the plot
     * @param t title string
     */
    public static void title(String t) { chartTitle = t; }
    
    /**
     * Set the label of the X axis
     * @param xl label string
     */
    public static void xlabel(String xl) { xLabel = xl; }
    
    /**
     * Set the label of the Y axis
     * @param yl label string
     */
    public static void ylabel(String yl) { yLabel = yl; }
    
    /**
     * Enable legend display on the plot
     */
    public static void legend() { showLegend = true; }

    /**
     * Plot a line graph with X and Y data
     * @param x X-axis values
     * @param y Y-axis values
     * @param label legend label for this series
     */
    public static void plot(double[] x, double[] y, String label) {
        XYSeries series = new XYSeries(label == null ? "Series " + (currentDataset.getSeriesCount()+1) : label);
        int n = Math.min(x.length, y.length);
        for (int i = 0; i < n; i++) {
            series.add(x[i], y[i]);
        }
        currentDataset.addSeries(series);
        isScatterList.add(false);
    }

    /**
     * Plot a line graph with X and Y data (no label)
     * @param x X-axis values
     * @param y Y-axis values
     */
    public static void plot(double[] x, double[] y) {
        plot(x, y, null);
    }

    /**
     * Plot a scatter plot with X and Y data
     * @param x X-axis values
     * @param y Y-axis values
     * @param label legend label for this series
     */
    public static void scatter(double[] x, double[] y, String label) {
        XYSeries series = new XYSeries(label == null ? "Series " + (currentDataset.getSeriesCount()+1) : label);
        int n = Math.min(x.length, y.length);
        for (int i = 0; i < n; i++) {
            series.add(x[i], y[i]);
        }
        currentDataset.addSeries(series);
        isScatterList.add(true);
    }

    /**
     * Plot a scatter plot with X and Y data (no label)
     * @param x X-axis values
     * @param y Y-axis values
     */
    public static void scatter(double[] x, double[] y) {
        scatter(x, y, null);
    }

    /**
     * Plot a line graph with X and Y NDArrays
     */
    public static void plot(numja.core.NDArray x, numja.core.NDArray y, String label) {
        plot(x.toDoubleArray(), y.toDoubleArray(), label);
    }

    /**
     * Plot a line graph with X and Y NDArrays (no label)
     */
    public static void plot(numja.core.NDArray x, numja.core.NDArray y) {
        plot(x.toDoubleArray(), y.toDoubleArray(), null);
    }

    /**
     * Plot a scatter plot with X and Y NDArrays
     */
    public static void scatter(numja.core.NDArray x, numja.core.NDArray y, String label) {
        scatter(x.toDoubleArray(), y.toDoubleArray(), label);
    }

    /**
     * Plot a scatter plot with X and Y NDArrays (no label)
     */
    public static void scatter(numja.core.NDArray x, numja.core.NDArray y) {
        scatter(x.toDoubleArray(), y.toDoubleArray(), null);
    }

    private static void buildChart() {
        currentChart = ChartFactory.createXYLineChart(
                chartTitle,
                xLabel,
                yLabel,
                currentDataset,
                PlotOrientation.VERTICAL,
                showLegend,
                true,
                false
        );

        XYPlot plot = currentChart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        for (int i = 0; i < currentDataset.getSeriesCount(); i++) {
            boolean isScatter = isScatterList.get(i);
            renderer.setSeriesLinesVisible(i, !isScatter);
            renderer.setSeriesShapesVisible(i, true);
        }
        plot.setRenderer(renderer);
    }

    /**
     * Define custom Heatmap data and labels
     */
    public static void heatmap(double[][] data, String[] labels) {
        heatmapData = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            heatmapData[i] = data[i].clone();
        }
        heatmapLabels = labels.clone();
        heatmapTitle = chartTitle != null && !chartTitle.equals("Figure 1") ? chartTitle : "Confusion Matrix";
    }

    /**
     * Display a 2D matrix as an image grid (imshow)
     * @param data 2D array of intensity/color values
     */
    public static void imshow(double[][] data) {
        imshowData = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            imshowData[i] = data[i].clone();
        }
        imshowTitle = chartTitle != null && !chartTitle.equals("Figure 1") ? chartTitle : "Image View (imshow)";
    }

    /**
     * Display a 2D NDArray as an image grid (imshow)
     * @param data 2D NDArray of intensity/color values
     */
    public static void imshow(numja.core.NDArray data) {
        imshow(data.toArray());
    }

    /**
     * Display the plot window and show the current figure
     */
    public static void show() {
        if (imshowData != null) {
            try {
                java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
                javax.swing.SwingUtilities.invokeLater(() -> {
                    javax.swing.JFrame frame = new javax.swing.JFrame(imshowTitle);
                    frame.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
                    frame.addWindowListener(new java.awt.event.WindowAdapter() {
                        @Override
                        public void windowClosed(java.awt.event.WindowEvent e) {
                            latch.countDown();
                        }
                    });
                    
                    ImshowPanel panel = new ImshowPanel(imshowData, imshowTitle);
                    panel.setPreferredSize(new java.awt.Dimension(800, 600));
                    frame.setContentPane(panel);
                    frame.pack();
                    frame.setLocationRelativeTo(null);
                    frame.setVisible(true);
                });
                latch.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return;
        }

        if (heatmapData != null) {
            try {
                java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
                javax.swing.SwingUtilities.invokeLater(() -> {
                    javax.swing.JFrame frame = new javax.swing.JFrame(heatmapTitle);
                    frame.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
                    frame.addWindowListener(new java.awt.event.WindowAdapter() {
                        @Override
                        public void windowClosed(java.awt.event.WindowEvent e) {
                            latch.countDown();
                        }
                    });
                    
                    HeatmapPanel panel = new HeatmapPanel(heatmapData, heatmapLabels, heatmapTitle);
                    panel.setPreferredSize(new java.awt.Dimension(800, 600));
                    frame.setContentPane(panel);
                    frame.pack();
                    frame.setLocationRelativeTo(null);
                    frame.setVisible(true);
                });
                latch.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return;
        }

        buildChart();
        if (currentChart == null) {
            System.out.println("Nothing to show.");
            return;
        }
        
        // Wait for the window to close before continuing execution to mimic python's blocking plt.show()
        try {
            java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
            
            javax.swing.SwingUtilities.invokeLater(() -> {
                javax.swing.JFrame frame = new javax.swing.JFrame("Figure 1");
                // Dispose on close, and count down the latch
                frame.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
                frame.addWindowListener(new java.awt.event.WindowAdapter() {
                    @Override
                    public void windowClosed(java.awt.event.WindowEvent e) {
                        latch.countDown();
                    }
                });
                
                org.jfree.chart.ChartPanel chartPanel = new org.jfree.chart.ChartPanel(currentChart);
                chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));
                frame.setContentPane(chartPanel);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            });
            
            latch.await(); // Block the main thread until the window is closed
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void savefig(String outPath) throws IOException {
        if (imshowData != null) {
            BufferedImage image = new BufferedImage(800, 600, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2 = image.createGraphics();
            ImshowPanel panel = new ImshowPanel(imshowData, imshowTitle);
            panel.setSize(800, 600);
            panel.paint(g2);
            g2.dispose();
            javax.imageio.ImageIO.write(image, "PNG", new File(outPath));
            return;
        }
        if (heatmapData != null) {
            BufferedImage image = new BufferedImage(800, 600, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2 = image.createGraphics();
            HeatmapPanel panel = new HeatmapPanel(heatmapData, heatmapLabels, heatmapTitle);
            panel.setSize(800, 600);
            panel.paint(g2);
            g2.dispose();
            javax.imageio.ImageIO.write(image, "PNG", new File(outPath));
            return;
        }
        buildChart();
        ChartUtils.saveChartAsPNG(new File(outPath), currentChart, 800, 600);
    }

    /**
     * Swing-based Heatmap panel for confusion matrices and correlation heatmaps
     */
    public static class HeatmapPanel extends javax.swing.JPanel {
        private final double[][] data;
        private final String[] labels;
        private final String title;

        public HeatmapPanel(double[][] data, String[] labels, String title) {
            this.data = data;
            this.labels = labels;
            this.title = title;
            setBackground(Color.WHITE);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

            int w = getWidth();
            int h = getHeight();

            int leftMargin = 100;
            int rightMargin = 160; // Make room for the vertical color bar
            int topMargin = 80;
            int bottomMargin = 100;

            int gridW = w - leftMargin - rightMargin;
            int gridH = h - topMargin - bottomMargin;

            if (gridW <= 0 || gridH <= 0) return;

            int rows = data.length;
            int cols = data[0].length;

            int cellW = gridW / cols;
            int cellH = gridH / rows;

            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            for (double[] row : data) {
                for (double val : row) {
                    if (val < min) min = val;
                    if (val > max) max = val;
                }
            }
            if (min == max) { min = 0; max = 1; }

            // High-end elegant colors
            Color minColor = new Color(243, 244, 246); // Cool Gray
            Color maxColor = new Color(37, 99, 235);   // Premium Royal Blue

            Font labelFont = new Font("SansSerif", Font.PLAIN, 12);
            Font valueFont = new Font("SansSerif", Font.BOLD, 14);
            Font headerFont = new Font("SansSerif", Font.BOLD, 18);
            Font titleFont = new Font("SansSerif", Font.BOLD, 14);

            // Cells
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    double val = data[r][c];
                    double frac = (val - min) / (max - min);
                    frac = Math.max(0.0, Math.min(1.0, frac));

                    int red = (int) (minColor.getRed() + frac * (maxColor.getRed() - minColor.getRed()));
                    int green = (int) (minColor.getGreen() + frac * (maxColor.getGreen() - minColor.getGreen()));
                    int blue = (int) (minColor.getBlue() + frac * (maxColor.getBlue() - minColor.getBlue()));
                    Color cellColor = new Color(red, green, blue);

                    int cx = leftMargin + c * cellW;
                    int cy = topMargin + r * cellH;

                    g2d.setColor(cellColor);
                    g2d.fillRect(cx, cy, cellW, cellH);

                    // Cell border
                    g2d.setColor(Color.WHITE);
                    g2d.setStroke(new BasicStroke(1.5f));
                    g2d.drawRect(cx, cy, cellW, cellH);

                    // Cell values
                    String text = (val == (long) val) ? String.format("%d", (long) val) : String.format("%.2f", val);
                    g2d.setFont(valueFont);
                    FontMetrics fm = g2d.getFontMetrics();
                    int tx = cx + (cellW - fm.stringWidth(text)) / 2;
                    int ty = cy + (cellH - fm.getHeight()) / 2 + fm.getAscent();

                    g2d.setColor(frac > 0.45 ? Color.WHITE : new Color(31, 41, 55));
                    g2d.drawString(text, tx, ty);
                }
            }

            // Draw Seaborn-style Vertical Color Bar on the Right
            int cbX = leftMargin + gridW + 40;
            int cbW = 22;
            int cbY = topMargin;
            int cbH = gridH;

            // Draw color bar gradient
            GradientPaint gp = new GradientPaint(cbX, cbY + cbH, minColor, cbX, cbY, maxColor);
            g2d.setPaint(gp);
            g2d.fillRect(cbX, cbY, cbW, cbH);

            // Draw color bar border
            g2d.setColor(new Color(209, 213, 219));
            g2d.setStroke(new BasicStroke(1.0f));
            g2d.drawRect(cbX, cbY, cbW, cbH);

            // Draw color bar ticks and labels
            g2d.setColor(new Color(75, 85, 99));
            g2d.setFont(labelFont);
            FontMetrics fmLayers = g2d.getFontMetrics();
            int numTicks = 5;
            for (int i = 0; i < numTicks; i++) {
                double frac = (double) i / (numTicks - 1);
                int ty = cbY + cbH - (int) (frac * cbH);
                
                // Draw small tick line
                g2d.drawLine(cbX + cbW, ty, cbX + cbW + 5, ty);

                // Draw value string next to tick line
                double tickVal = min + frac * (max - min);
                String tickText = (tickVal == (long) tickVal) ? String.format("%d", (long) tickVal) : String.format("%.2f", tickVal);
                g2d.drawString(tickText, cbX + cbW + 10, ty + fmLayers.getAscent() / 2 - 2);
            }

            // Headers & labels
            g2d.setColor(new Color(17, 24, 39));
            g2d.setFont(headerFont);
            FontMetrics fmHeader = g2d.getFontMetrics();
            g2d.drawString(title, (w - fmHeader.stringWidth(title)) / 2, 45);

            g2d.setFont(titleFont);
            FontMetrics fmTitle = g2d.getFontMetrics();
            String xTitle = "Predicted Class";
            g2d.drawString(xTitle, leftMargin + (gridW - fmTitle.stringWidth(xTitle)) / 2, h - 30);

            String yTitle = "Actual Class";
            AffineTransform orig = g2d.getTransform();
            g2d.translate(30, topMargin + gridH / 2);
            g2d.rotate(-Math.PI / 2);
            g2d.drawString(yTitle, -fmTitle.stringWidth(yTitle) / 2, 0);
            g2d.setTransform(orig);

            // Ticks
            g2d.setFont(labelFont);
            for (int i = 0; i < labels.length; i++) {
                String label = labels[i];

                int yTickY = topMargin + i * cellH + cellH / 2 + fmLayers.getAscent() / 2 - 2;
                g2d.drawString(label, leftMargin - 15 - fmLayers.stringWidth(label), yTickY);

                int xTickX = leftMargin + i * cellW + cellW / 2 - fmLayers.stringWidth(label) / 2;
                g2d.drawString(label, xTickX, topMargin + gridH + 20);
            }
        }
    }

    /**
     * Swing-based Imshow panel for actual pixel grid rendering (no numbers or cell lines)
     */
    public static class ImshowPanel extends javax.swing.JPanel {
        private final double[][] data;
        private final String title;

        public ImshowPanel(double[][] data, String title) {
            this.data = data;
            this.title = title;
            setBackground(Color.WHITE);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth();
            int h = getHeight();

            int leftMargin = 80;
            int rightMargin = 80;
            int topMargin = 80;
            int bottomMargin = 80;

            int gridW = w - leftMargin - rightMargin;
            int gridH = h - topMargin - bottomMargin;

            if (gridW <= 0 || gridH <= 0) return;

            int rows = data.length;
            int cols = data[0].length;

            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            for (double[] row : data) {
                for (double val : row) {
                    if (val < min) min = val;
                    if (val > max) max = val;
                }
            }
            if (min == max) { min = 0; max = 1; }

            // Create a BufferedImage of the exact matrix shape
            BufferedImage img = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    double val = data[y][x];
                    double frac = (val - min) / (max - min);
                    frac = Math.max(0.0, Math.min(1.0, frac));
                    // Smooth grayscale pixel mapping
                    int gray = (int) (frac * 255);
                    int rgb = (gray << 16) | (gray << 8) | gray;
                    img.setRGB(x, y, rgb);
                }
            }

            // Draw the BufferedImage stretched smoothly to fill the grid area
            g2d.drawImage(img, leftMargin, topMargin, gridW, gridH, null);

            // Draw grid outline border
            g2d.setColor(Color.DARK_GRAY);
            g2d.setStroke(new BasicStroke(2.0f));
            g2d.drawRect(leftMargin, topMargin, gridW, gridH);

            // Draw title
            Font headerFont = new Font("SansSerif", Font.BOLD, 18);
            g2d.setColor(new Color(17, 24, 39));
            g2d.setFont(headerFont);
            FontMetrics fmHeader = g2d.getFontMetrics();
            g2d.drawString(title, (w - fmHeader.stringWidth(title)) / 2, 45);

            // Draw axis ticks and boundaries (0 to rows/cols)
            Font labelFont = new Font("SansSerif", Font.PLAIN, 12);
            g2d.setFont(labelFont);
            g2d.setColor(new Color(75, 85, 99));
            FontMetrics fmLayers = g2d.getFontMetrics();

            // X ticks (top and bottom)
            g2d.drawString("0", leftMargin - fmLayers.stringWidth("0") / 2, topMargin + gridH + 20);
            g2d.drawString(String.valueOf(cols), leftMargin + gridW - fmLayers.stringWidth(String.valueOf(cols)) / 2, topMargin + gridH + 20);

            // Y ticks (left)
            g2d.drawString("0", leftMargin - 15 - fmLayers.stringWidth("0"), topMargin + fmLayers.getAscent() / 2);
            g2d.drawString(String.valueOf(rows), leftMargin - 15 - fmLayers.stringWidth(String.valueOf(rows)), topMargin + gridH + fmLayers.getAscent() / 2 - 2);
        }
    }
}
