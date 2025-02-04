package segmentation.models;

import java.awt.Point;
import java.util.List;

public class Region {
    private final BoundingBox boundingBox;
    private final double area;
    private final List<Point> contourPoints;

    public Region(BoundingBox boundingBox, double area, List<Point> contourPoints) {
        this.boundingBox = boundingBox;
        this.area = area;
        this.contourPoints = contourPoints;
    }

    // Getters
    public BoundingBox getBoundingBox() { return boundingBox; }
    public double getArea() { return area; }
    public List<Point> getContourPoints() { return contourPoints; }
} 