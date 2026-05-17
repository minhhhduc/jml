import matplotlib.Matplotlib;
import numja.core.NDArray;
import numja.NumJa;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Dedicated test script for Matplotlib.imshow() proving high-resolution pixel image rendering.
 * 1. Generates and renders a smooth 200x200 radial gradient wave image.
 * 2. Loads a real image file (confusion_matrix.png) as a 2D grayscale NDArray and displays it.
 */
public class TestImshow {

    public static void main(String[] args) {
        System.out.println("=================================================");
        System.out.println("  Testing Matplotlib.imshow() Smooth Pixel Mode  ");
        System.out.println("=================================================");

        // --- PART 1: Smooth 200x200 Radial Wave Grid ---
        System.out.println("\n[1/2] Generating a high-resolution smooth 200x200 gradient wave...");
        int size = 200;
        double[][] grid = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double dx = i - (size / 2.0);
                double dy = j - (size / 2.0);
                grid[i][j] = Math.sin(Math.sqrt(dx * dx + dy * dy) / 10.0);
            }
        }

        NDArray data = NumJa.array(grid);
        System.out.println("Generated NDArray shape: " + data.getShape()[0] + "x" + data.getShape()[1]);

        Matplotlib.clf();
        Matplotlib.title("Matplotlib.imshow() Smooth 200x200 Radial Wave");
        Matplotlib.imshow(data);

        System.out.println("--> Showing 200x200 pattern window. Close it to proceed to image loading...");
        Matplotlib.show();


        // --- PART 2: Load and Render Real Image File (confusion_matrix.png) ---
        System.out.println("\n[2/2] Loading real image file 'confusion_matrix.png' as 2D Grayscale NDArray...");
        File imgFile = new File("confusion_matrix.png");
        if (!imgFile.exists()) {
            imgFile = new File("../confusion_matrix.png");
        }

        if (imgFile.exists()) {
            try {
                NDArray grayscaleImg = loadImage(imgFile.getAbsolutePath());
                System.out.println("Successfully loaded image file!");
                System.out.println("Image shape: " + grayscaleImg.getShape()[0] + "x" + grayscaleImg.getShape()[1]);

                Matplotlib.clf();
                Matplotlib.title("Matplotlib.imshow() Real Image File: " + imgFile.getName());
                Matplotlib.imshow(grayscaleImg);

                System.out.println("--> Showing loaded image file window. Close it to finish...");
                Matplotlib.show();

            } catch (IOException e) {
                System.out.println("Could not read image file: " + e.getMessage());
            }
        } else {
            System.out.println("Image file 'confusion_matrix.png' not found in workspace root.");
        }

        System.out.println("\nTest completed successfully!");
    }

    /**
     * Helper method to load a real image file, convert it to Grayscale, and return as 2D NDArray
     */
    private static NDArray loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        int w = img.getWidth();
        int h = img.getHeight();
        double[][] data = new double[h][w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                // Standard Luma grayscale conversion formula
                data[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        }
        return NumJa.array(data);
    }
}
