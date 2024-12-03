package acquisition;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.List;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.nio.file.Files;
import utils.FileProcessingException;

public class FileHandler {
    private static final long DEFAULT_MAX_SIZE = 10 * 1024 * 1024; // 10MB
    private static final Logger logger = LoggerFactory.getLogger(FileHandler.class);
    static final Set<String> SUPPORTED_EXTENSIONS = new HashSet<>(
            Arrays.asList("pdf", "png", "jpg", "jpeg", "tiff")
    );

    // Use ThreadLocal to store fileName, fileNameNoX, and extension for thread safety
    static final ThreadLocal<String> fileName = ThreadLocal.withInitial(() -> null);
    static final ThreadLocal<String> fileNameNoX = ThreadLocal.withInitial(() -> null);
    static final ThreadLocal<String> extension = ThreadLocal.withInitial(() -> null);

    public File loadFile(String filePath, boolean validateOnLoad, long maxSizeOverride) throws FileProcessingException {
        File file = new File(filePath);
        if (!file.exists()) {
            logger.error("File on path {} does not exist.", filePath);
            throw new FileProcessingException("File on path " + filePath + " does not exist.");
        }
        if (!file.canRead()) {
            logger.error("Cannot read file, check permissions for path: {}", filePath);
            throw new FileProcessingException("Cannot read file, check permissions.");
        }

        if (validateOnLoad) {
            long effectiveMaxSize = maxSizeOverride > 0 ? maxSizeOverride : DEFAULT_MAX_SIZE;
            validateFileWithSize(file, effectiveMaxSize);
        }

        return file;
    }

    private void validateFileWithSize(File file, long maxSize) throws FileProcessingException {
        if (!file.isFile()) {
            logger.error("Invalid file on path: {}", file.getPath());
            throw new FileProcessingException("Invalid file on path: " + file.getPath());
        }
        if (file.length() > maxSize) {
            logger.error("File size exceeds maximum limit of {} MB", (maxSize / (1024 * 1024)));
            throw new FileProcessingException("File size exceeds maximum limit of " + (maxSize / (1024 * 1024)) + " MB");
        }
        if (file.length() == 0) {
            logger.error("File is empty: {}", file.getPath());
            throw new FileProcessingException("File is empty.");
        }
        checkFileType(file);
    }

    public String checkFileType(File file) throws FileProcessingException {
        fileName.set(file.getName());
        extension.set(fileName.get().substring(fileName.get().lastIndexOf(".") + 1).toLowerCase());
        fileNameNoX.set(fileName.get().substring(0, fileName.get().lastIndexOf('.')));
        if (!SUPPORTED_EXTENSIONS.contains(extension.get())) {
            logger.error("Unsupported file of type: {}", extension.get());
            throw new FileProcessingException("Unsupported file of type: " + extension.get());
        }

        String mimeType;
        try {
            mimeType = Files.probeContentType(file.toPath());
        } catch (IOException e) {
            logger.error("Error probing content type: {}", e.getMessage());
            throw new FileProcessingException("Error probing content type for file: " + file.getPath(), e);
        }
        if (!isSupportedMimeType(mimeType)) {
            logger.error("Unsupported MIME type: {}", mimeType);
            throw new FileProcessingException("Unsupported MIME type: " + mimeType);
        }

        return extension.get();
    }

    private boolean isSupportedMimeType(String mimeType) {
        return mimeType != null && (mimeType.equals("image/png") || 
                                    mimeType.equals("image/jpeg") || 
                                    mimeType.equals("image/tiff") || 
                                    mimeType.equals("application/pdf"));
    }

    public void saveImages(List<BufferedImage> images, String basePath) {
        int index = 0;
        for (BufferedImage image : images) {
            index++;
            try {
                File outputFile = new File(basePath + "-out-" + index + ".png");
                ImageIO.write(image, "png", outputFile);
                logger.info("Saved image: {}", outputFile.getName());
            } catch (IOException e) {
                logger.error("Error saving image: {}", e.getMessage());
            }
        }
    }
}