package ats.rparsing.acquisition;

import java.io.File;
import java.awt.image.BufferedImage;
import java.util.List;

import java.util.ArrayList;
import java.net.URL;
import javax.imageio.ImageIO;
import java.io.IOException;

public class MainAcquisition {
    private final FileHandler fileHandler;
    private final FileConverter fileConverter;
    private final ImageProcessor imageProcessor;
    private static final long DEFAULT_MAX_SIZE = 10 * 1024 * 1024; // 10MB

    public MainAcquisition() {
        this.fileHandler = new FileHandler();
        this.fileConverter = new FileConverter();
        this.imageProcessor = new ImageProcessor();
    }

    public List<BufferedImage> processDocument(String filePath) {
        try {
            // Step 1: Load and validate file
            File documentFile = fileHandler.loadFile(filePath, true, DEFAULT_MAX_SIZE);

            // Step 2: Convert document to images
            List<BufferedImage> documentImages = fileConverter.convertToImage(documentFile);

            // Step 3: Process each image
            List<BufferedImage> processedImages = new ArrayList<>();
            for (BufferedImage image : documentImages) {
                // Apply preprocessing
                BufferedImage preprocessed = imageProcessor.preprocessImage(image);

                // Apply thresholding
                BufferedImage thresholded = imageProcessor.applyThresholding(preprocessed);

                // Enhance quality
                BufferedImage enhanced = imageProcessor.enhanceImageQuality(thresholded);

                processedImages.add(enhanced);
            }

            System.out.println("Document processed successfully");
            return processedImages;

        } catch (Exception e) {
            System.err.println("Error processing document: " + e.getMessage());
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    public static void main(String[] args) {
        MainAcquisition mainAcquisition = new MainAcquisition();
        
        try {
            URL resourceUrl = MainAcquisition.class.getClassLoader().getResource("cv.pdf");  //args[0]
            if (resourceUrl == null) {
                System.err.println("Could not find file in resources");
                return;
            }
            
            String filePath = java.net.URLDecoder.decode(resourceUrl.getPath(), "UTF-8");
            System.out.println("Processing file from path: " + filePath);
            List<BufferedImage> processedImages = mainAcquisition.processDocument(filePath);
            System.out.println("Processed " + processedImages.size() + " images");
            int index = 0;
            for (BufferedImage image : processedImages) {
                index++;
                // Save each image with an index to disk
                try {
                    File outputFile = new File("src/main/resources/output_cv_" + index + ".png");
                    ImageIO.write(image, "png", outputFile);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
