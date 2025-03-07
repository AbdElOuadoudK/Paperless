package ats.rparsing.acquisition;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class FileHandler {
    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    static final Set<String> SUPPORTED_EXTENSIONS = new HashSet<>(
            Arrays.asList("pdf", "png", "jpg", "jpeg", "tiff")
    );

    public File loadFile(String filePath, boolean validateOnLoad, long maxSizeOverride) throws IOException {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new IOException("File on path " + filePath + " does not exist.");
        }
        if (!file.canRead()) {
            throw new IOException("Cannot read file, check permissions.");
        }

        if (validateOnLoad) {
            long effectiveMaxSize = maxSizeOverride > 0 ? maxSizeOverride : MAX_FILE_SIZE;
            validateFileWithSize(file, effectiveMaxSize);
        }

        return file;
    }

    // Helper method for validation with custom size
    private void validateFileWithSize(File file, long maxSize) throws IOException {
        if (!file.isFile()) {
            throw new IOException("Invalid file on path: " + file.getPath());
        }
        if (file.length() > maxSize) {
            throw new IOException("File size exceeds maximum limit of " + (maxSize / (1024 * 1024)) + " MB");
        }
        if (file.length() == 0) {
            throw new IOException("File is empty.");
        }
        checkFileType(file);
    }

    public String checkFileType(File file) throws IOException {
        String fileName = file.getName();
        String extension = fileName.substring(fileName.lastIndexOf(".") + 1).toLowerCase();

        if (!SUPPORTED_EXTENSIONS.contains(extension)) {
            throw new IOException("Unsupported file of type: " + extension);
        }
        return extension;
    }
}