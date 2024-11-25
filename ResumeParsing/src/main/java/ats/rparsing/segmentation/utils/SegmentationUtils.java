package ats.rparsing.segmentation.utils;

import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;

public class SegmentationUtils {

    private static final float[] SOBEL_X = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    private static final float[] SOBEL_Y = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };

    public static BufferedImage applyEdgeDetection(BufferedImage image) {
        BufferedImage result = new BufferedImage(
            image.getWidth(), 
            image.getHeight(),
            BufferedImage.TYPE_BYTE_GRAY
        );

        // Apply Sobel operator
        BufferedImage gradientX = applySobelOperator(image, SOBEL_X);
        BufferedImage gradientY = applySobelOperator(image, SOBEL_Y);

        // Combine gradients
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int gx = gradientX.getRGB(x, y) & 0xFF;
                int gy = gradientY.getRGB(x, y) & 0xFF;
                int magnitude = (int) Math.sqrt(gx * gx + gy * gy);
                magnitude = Math.min(magnitude, 255);
                result.setRGB(x, y, (magnitude << 16) | (magnitude << 8) | magnitude);
            }
        }

        return result;
    }

    private static BufferedImage applySobelOperator(BufferedImage image, float[] kernel) {
        Kernel k = new Kernel(3, 3, kernel);
        ConvolveOp op = new ConvolveOp(k, ConvolveOp.EDGE_NO_OP, null);
        return op.filter(image, null);
    }

    public static boolean isLineDetected(BufferedImage image, int yStart, int yEnd) {
        int threshold = 200; // Adjust based on image characteristics
        int minLineLength = image.getWidth() / 3; // Minimum line length
        
        for (int y = yStart; y <= yEnd; y++) {
            int consecutiveBlackPixels = 0;
            for (int x = 0; x < image.getWidth(); x++) {
                if ((image.getRGB(x, y) & 0xFF) < threshold) {
                    consecutiveBlackPixels++;
                    if (consecutiveBlackPixels >= minLineLength) {
                        return true;
                    }
                } else {
                    consecutiveBlackPixels = 0;
                }
            }
        }
        return false;
    }
} 