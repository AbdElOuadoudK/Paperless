package acquisition;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import utils.FileProcessingException;
import utils.ImageProcessingException;

import java.io.File;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

public class MainAcquisition {
    private static final Logger logger = LoggerFactory.getLogger(MainAcquisition.class);
    private final FileHandler fileHandler;
    private final FileConverter fileConverter;
    private final ImageProcessor imageProcessor;
    
    public MainAcquisition() {
        this.fileHandler = new FileHandler();
        this.fileConverter = new FileConverter();
        this.imageProcessor = new ImageProcessor();
    }

    public CompletableFuture<List<BufferedImage>> processDocument(String filePath, Boolean verbose, long MAX_FILE_SIZE, Boolean clutter, Boolean async) throws FileProcessingException, ImageProcessingException {
        if (verbose) {
            logger.info("Acquiring Document...");
        }

        // If async is true, run the processing in a separate thread
        if (async) {
            return CompletableFuture.supplyAsync(() -> {
                try {
                    return processDocumentInternal(filePath, verbose, MAX_FILE_SIZE, clutter);
                } catch (FileProcessingException | ImageProcessingException e) {
                    logger.error("Error processing document: {}", e.getMessage(), e);
                    throw new RuntimeException(e);
                }
            });
        } else {
            // Synchronous processing
            return CompletableFuture.completedFuture(processDocumentInternal(filePath, verbose, MAX_FILE_SIZE, clutter));
        }
    }

    private List<BufferedImage> processDocumentInternal(String filePath, Boolean verbose, long MAX_FILE_SIZE, Boolean clutter) throws FileProcessingException, ImageProcessingException {
        File documentFile = null;
        try {
            // Step 1: Load and validate file
            documentFile = fileHandler.loadFile(filePath, true, MAX_FILE_SIZE);
            if (documentFile == null) {
                throw new FileProcessingException("Failed to load the document file.");
            }

            // Step 2: Convert document to images
            List<BufferedImage> documentImages = fileConverter.convertToImage(documentFile);
            if (documentImages == null || documentImages.isEmpty()) {
                throw new ImageProcessingException("No images were generated from the document.");
            }

            // Step 3: Process each image
            List<BufferedImage> processedImages = new ArrayList<>();
            for (BufferedImage image : documentImages) {
                BufferedImage processed = imageProcessor.processImage(image);
                processedImages.add(processed);
            }

            if (verbose) {
                logger.info("File size: {} bytes.", documentFile.length());
                logger.info("{} image(s) acquired.", processedImages.size());
            }
            return processedImages;

        } catch (FileProcessingException | ImageProcessingException e) {
            logger.error("Error processing document: {}", e.getMessage(), e);
            throw e; // Rethrow the custom exception
        } catch (Exception e) {
            logger.error("Unexpected error: {}", e.getMessage(), e);
            throw new FileProcessingException("An unexpected error occurred while processing the document.", e);
        } finally {
            // Step 4: Cleanup temporary files if clutter is true
            if (clutter && documentFile != null && documentFile.exists()) {
                boolean deleted = documentFile.delete();
                if (deleted) {
                    logger.info("File deleted: {}", FileHandler.fileName.get());
                } else {
                    logger.warn("Failed to delete temporary file: {}", FileHandler.fileName.get());
                }
            }
        }
    }

    public static void main(String[] args) {
        MainAcquisition mainAcquisition = new MainAcquisition();

        try {
            String DOCS_DIR = "src/main/resources/resumes/";
            String basePath = "src/main/resources/out/acquisition-";
            Boolean verbose = true;
            Boolean clutter = false;
            long file_max_size = 10 * 1024 * 1024;
            Boolean async = false;


            String fileName = "resume-x.pdf";
            CompletableFuture<List<BufferedImage>> futureImages = mainAcquisition.processDocument(DOCS_DIR + fileName, verbose, file_max_size, clutter, async);
            futureImages.thenAccept(processedImages -> {
                mainAcquisition.fileHandler.saveImages(processedImages, basePath + FileHandler.fileNameNoX.get());
            }).exceptionally(e -> {
                logger.error("Error: {}", e.getMessage(), e);
                return null;
            });

        } catch (FileProcessingException | ImageProcessingException e) {
            logger.error("Error: {}", e.getMessage(), e);
        } catch (Exception e) {
            logger.error("Unexpected error: {}", e.getMessage(), e);
        }
    }
}