package com.ats.langdetect.util;

/**
 * Custom exception for handling language detection errors.
 */
public class LanguageDetectionException extends Exception {

    /**
     * Constructs a new LanguageDetectionException with a specified detail message.
     * @param message The detail message.
     */
    public LanguageDetectionException(String message) {
        super(message);
    }
}
