package imageproc;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

public class ImageUtils {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // Display an image from a file path or a Mat object
    public static void display(Mat img, String imgPath) {
        if (img == null && imgPath != null) {
            img = Imgcodecs.imread(imgPath);
        } else if (img == null) {
            throw new IllegalArgumentException("No image provided. Please provide either imgPath or img.");
        }

        // Convert Mat to BufferedImage
        BufferedImage bufferedImage = (BufferedImage) HighGui.toBufferedImage(img);

        // Display with JFrame
        JFrame frame = new JFrame();
        frame.getContentPane().add(new JLabel(new ImageIcon(bufferedImage)));
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    // Save an image and display it
    public static void writeDisplay(Mat img, String imgPath) {
        Imgcodecs.imwrite(imgPath, img);
        display(null, imgPath);
    }

    // Convert an image to grayscale
    public static Mat greyscale(Mat img) {
        Mat greyImg = new Mat();
        Imgproc.cvtColor(img, greyImg, Imgproc.COLOR_BGR2GRAY);
        return greyImg;
    }

    // Perform image thinning using erosion
    public static Mat thinning(Mat img) {
        Mat erodedImg = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Imgproc.erode(img, erodedImg, kernel);
        return erodedImg;
    }

    // Remove noise using morphological transformations
    public static Mat noiseRemoval(Mat img) {
        Mat denoisedImg = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1));
        Imgproc.dilate(img, denoisedImg, kernel, new Point(-1, -1), 3);
        Imgproc.morphologyEx(denoisedImg, denoisedImg, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.medianBlur(denoisedImg, denoisedImg, 3);
        return denoisedImg;
    }

    // Apply Canny edge detection with median-based thresholds
    public static Mat medianCanny(Mat img, double thresh1, double thresh2) {
        Mat edgeImg = new Mat();
        Mat gray = greyscale(img);
        double median = Core.mean(gray).val[0];
        Imgproc.Canny(gray, edgeImg, thresh1 * median, thresh2 * median);
        return edgeImg;
    }

    // Helper function to check if two rectangles overlap
    public static boolean overlap(Rect rect1, Rect rect2) {
        return rect1.x < rect2.x + rect2.width && rect1.x + rect1.width > rect2.x &&
                rect1.y < rect2.y + rect2.height && rect1.y + rect1.height > rect2.y;
    }

    // Additional methods would follow to translate functions like `display_`, `overlap_1`, `overlap_2`, `iter_boxes`, etc.
}
