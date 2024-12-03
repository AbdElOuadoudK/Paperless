package acquisition;

import utils.FileConversionException;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class FileConverter {
    private static final Logger logger = LoggerFactory.getLogger(FileConverter.class);

    public CompletableFuture<List<BufferedImage>> convertToImage(File file) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                String fileName = file.getName().toLowerCase();

                if (fileName.endsWith(".pdf")) {
                    logger.info("PDF file detected, attempting conversion...");
                    try (PDDocument document = PDDocument.load(file)) {
                        return splitPagesIntoImages(document);
                    } catch (Exception e) {
                        throw new FileConversionException("Failed to convert PDF: " + e.getMessage(), e);
                    }
                } else if (FileHandler.SUPPORTED_EXTENSIONS.stream()
                        .anyMatch(ext -> fileName.toLowerCase().endsWith("." + ext))) {
                    logger.info("Image file detected, loading directly...");
                    try {
                        List<BufferedImage> images = new ArrayList<>();
                        BufferedImage image = null;
                        try {
                            image = ImageIO.read(new FileInputStream(file));
                        } catch (Exception e) {
                            logger.warn("First attempt failed, trying alternate method...");
                            try (InputStream is = new FileInputStream(file)) {
                                byte[] bytes = is.readAllBytes();
                                try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes)) {
                                    image = ImageIO.read(bis);
                                }
                            }
                        }

                        if (image == null) {
                            throw new FileConversionException("Failed to read image file: " + fileName + 
                                ". Supported formats are: " + String.join(", ", ImageIO.getReaderFormatNames()));
                        }
                        images.add(image);
                        return images;
                    } catch (Exception e) {
                        throw new FileConversionException("Failed to convert image: " + e.getMessage() + 
                            "\nFile path: " + file.getAbsolutePath(), e);
                    }
                } else {
                    logger.error("Unsupported file format detected: {}", fileName);
                    throw new UnsupportedOperationException("Unsupported file format: " + fileName);
                }
            } catch (FileConversionException e) {
                logger.error("Conversion error: {}", e.getMessage());
                throw new RuntimeException(e);
            } catch (Exception e) {
                logger.error("Unexpected error: {}", e.getMessage());
                throw new RuntimeException(e);
            }
        });
    }

    private List<BufferedImage> splitPagesIntoImages(PDDocument document) throws Exception {
        logger.info("Starting PDF page splitting process...");
        List<BufferedImage> images = new ArrayList<>();
        PDFRenderer pdfRenderer = new PDFRenderer(document);

        for (int page = 0; page < document.getNumberOfPages(); page++) {
            BufferedImage image = pdfRenderer.renderImageWithDPI(page, 300);
            images.add(image);
        }
        return images;
    }
}