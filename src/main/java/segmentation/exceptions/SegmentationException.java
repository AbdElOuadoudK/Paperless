package segmentation.exceptions;

public class SegmentationException extends Exception {
    public SegmentationException(String message) {
        super(message);
    }

    public SegmentationException(String message, Throwable cause) {
        super(message, cause);
    }
} 