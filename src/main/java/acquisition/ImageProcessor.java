package acquisition;

import java.awt.image.BufferedImage;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

public class ImageProcessor {
    // Constants for image processing
    private static final int TARGET_WIDTH = 2400;
    private static final int THRESHOLD_VALUE = 210;  // Increased to catch lighter text
    private static final float SHARPEN_WEIGHT = 0.4f;  // Increased slightly for better edge definition
    private static final int WINDOW_SIZE = 15;  // For adaptive thresholding

    public BufferedImage processImage(BufferedImage image) {
        // Apply preprocessing
        BufferedImage preprocessed = preprocessImage(image);

        // Apply thresholding
        BufferedImage thresholded = applyThresholding(preprocessed);
        
        // Enhance quality
        return enhanceImageQuality(thresholded);
    }

    public BufferedImage preprocessImage(BufferedImage image) {
        if (image == null) return null;

        // Convert to grayscale
        BufferedImage grayscale = new BufferedImage(
                image.getWidth(),
                image.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY
        );
        Graphics2D g = grayscale.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();
        // Resize image while maintaining aspect ratio
        double aspectRatio = (double) image.getHeight() / image.getWidth();
        int targetHeight = (int) (TARGET_WIDTH * aspectRatio);
        BufferedImage resized = new BufferedImage(TARGET_WIDTH, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.drawImage(grayscale, 0, 0, TARGET_WIDTH, targetHeight, null);
        g.dispose();

        // Remove blur step and only apply sharpening
        return enhanceImageQuality(resized);
    }

    public BufferedImage applyThresholding(BufferedImage image) {
        if (image == null) return null;

        BufferedImage result = new BufferedImage(
                image.getWidth(),
                image.getHeight(),
                BufferedImage.TYPE_BYTE_BINARY
        );

        // Apply adaptive thresholding with local window
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                // Calculate local average
                int sum = 0;
                int count = 0;
                for (int wy = Math.max(0, y - WINDOW_SIZE/2); 
                     wy < Math.min(image.getHeight(), y + WINDOW_SIZE/2); wy++) {
                    for (int wx = Math.max(0, x - WINDOW_SIZE/2); 
                         wx < Math.min(image.getWidth(), x + WINDOW_SIZE/2); wx++) {
                        int pixel = image.getRGB(wx, wy);
                        sum += (pixel >> 16) & 0xff;
                        count++;
                    }
                }
                int average = sum / count;
                
                // Get current pixel brightness
                int pixel = image.getRGB(x, y);
                int brightness = (pixel >> 16) & 0xff;
                
                // Apply local threshold with offset
                int threshold = Math.max(average - 10, THRESHOLD_VALUE);
                result.setRGB(x, y, brightness < threshold ? Color.BLACK.getRGB() : Color.WHITE.getRGB());
            }
        }
        return result;
    }

    public BufferedImage enhanceImageQuality(BufferedImage image) {
        if (image == null) return null;

        // Create sharpening kernel
        float[] sharpenKernel = {
                0.0f, -SHARPEN_WEIGHT, 0.0f,
                -SHARPEN_WEIGHT, 1 + (4 * SHARPEN_WEIGHT), -SHARPEN_WEIGHT,
                0.0f, -SHARPEN_WEIGHT, 0.0f
        };

        // Apply kernel convolution for sharpening
        BufferedImage sharpened = new BufferedImage(
                image.getWidth(),
                image.getHeight(),
                image.getType()
        );

        for (int y = 1; y < image.getHeight() - 1; y++) {
            for (int x = 1; x < image.getWidth() - 1; x++) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int pixel = image.getRGB(x + kx, y + ky);
                        int brightness = (pixel >> 16) & 0xff;
                        sum += brightness * sharpenKernel[(ky + 1) * 3 + (kx + 1)];
                    }
                }
                int newBrightness = Math.min(Math.max((int) sum, 0), 255);
                sharpened.setRGB(x, y, new Color(newBrightness, newBrightness, newBrightness).getRGB());
            }
        }
        return sharpened;
    }
}
