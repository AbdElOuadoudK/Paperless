package segmentation.config;

public class SegmentationConfig {
    private final int minBlockSize;
    private final int thresholdValue;
    private final double minRegionArea;
    private final int edgeDetectionThreshold;

    public SegmentationConfig(int minBlockSize, int thresholdValue, 
                            double minRegionArea, int edgeDetectionThreshold) {
        this.minBlockSize = minBlockSize;
        this.thresholdValue = thresholdValue;
        this.minRegionArea = minRegionArea;
        this.edgeDetectionThreshold = edgeDetectionThreshold;
    }

    // Getters
    public int getMinBlockSize() { return minBlockSize; }
    public int getThresholdValue() { return thresholdValue; }
    public double getMinRegionArea() { return minRegionArea; }
    public int getEdgeDetectionThreshold() { return edgeDetectionThreshold; }
} 