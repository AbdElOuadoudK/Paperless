package utils;

public class ImageReadException extends Exception {
    public ImageReadException(String message) {
        super(message);
    }

    public ImageReadException(String message, Throwable cause) {
        super(message, cause);
    }
} 