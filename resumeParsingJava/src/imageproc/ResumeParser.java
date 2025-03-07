package imageproc;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import com.resumeparser.imageproc.utils.*;
import java.util.ArrayList;
import java.util.List;

public class ResumeParser {
    private boolean save;
    private boolean display;

    public ResumeParser(boolean save, boolean display) {
        this.save = save;
        this.display = display;
    }

    public void processImage() {
        // Method for image processing pipeline (to be implemented)
    }

    private void main() {
        // Entry point for processing logic (to be implemented)
    }

    public Tuple<Mat, List<Rect>> scanFrames(Mat img, int margin, int maxArea, boolean displayFlag) {
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels);

        Mat blueEdges = Utils.medianCanny(channels.get(0), 0, 1);
        Mat greenEdges = Utils.medianCanny(channels.get(1), 0, 1);
        Mat redEdges = Utils.medianCanny(channels.get(2), 0, 1);
        Mat edges = new Mat();
        Core.bitwise_or(blueEdges, greenEdges, edges);
        Core.bitwise_or(edges, redEdges, edges);

        List<Rect> boxes = Utils.scanImg(edges, margin, maxArea);

        if (displayFlag) {
            Utils.display(img, boxes);
        }

        return new Tuple<>(img, boxes);
    }

    public Tuple<Mat, List<Rect>> cropImage(Mat image, List<Rect> boxes, int index, int kernelMaxSize, int padding, boolean displayFlag) {
        Mat croppedImg = Utils.retrieveCropped(image, boxes.get(index));
        Mat gray = new Mat();
        Imgproc.cvtColor(croppedImg, gray, Imgproc.COLOR_BGR2GRAY);
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(3, 3), 0);
        Mat bw = new Mat();
        Imgproc.threshold(blurred, bw, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        List<Rect> allLines = new ArrayList<>();
        for (int k = 1; k <= kernelMaxSize; k++) {
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(k, 1));
            Mat bwClosed = new Mat();
            Imgproc.morphologyEx(bw, bwClosed, Imgproc.MORPH_CLOSE, kernel);
            List<Rect> lines = Utils.getLines(bwClosed, padding);
            if (lines != null) {
                allLines.addAll(lines);
            }
        }

        if (allLines.isEmpty()) {
            System.out.println("Second FLAG");
            return clusterBoxes(croppedImg);
        }

        List<Rect> lines = new ArrayList<>();
        for (Rect line : allLines) {
            if (!lines.contains(line) && Utils.checkSize(line, 52)) {
                lines.add(line);
            }
        }

        if (lines.isEmpty()) {
            System.out.println("Second FLAG");
            return clusterBoxes(croppedImg);
        }

        lines.sort((a, b) -> (a.y != b.y) ? a.y - b.y : a.x - b.x);
        System.out.println("First FLAG");

        if (displayFlag) {
            Utils.display(croppedImg, lines);
        }

        return new Tuple<>(croppedImg, lines);
    }

    public Tuple<Mat, List<Rect>> clusterBoxes(Mat croppedImg, boolean displayFlag) {
        Tuple<Mat, List<Rect>> scanResult = scanFrames(croppedImg, 5, 3000, false);
        croppedImg = scanResult.getFirst();
        List<Rect> boxes = scanResult.getSecond();

        List<Point> xCenters = new ArrayList<>();
        for (Rect box : boxes) {
            xCenters.add(new Point(box.x + box.width / 2, box.y + box.height / 2));
        }

        double epsilon = xCenters.stream().mapToDouble(p -> p.y).min().orElse(1.0);
        HDBSCAN clustering = new HDBSCAN(epsilon, 2);  // Placeholder for HDBSCAN clustering
        int[] labels = clustering.fitPredict(xCenters);

        List<Rect> mergedBoxes = Utils.mergeClusters(boxes, labels);

        if (displayFlag) {
            Utils.display(croppedImg, mergedBoxes);
        }

        return new Tuple<>(croppedImg, mergedBoxes);
    }
}
