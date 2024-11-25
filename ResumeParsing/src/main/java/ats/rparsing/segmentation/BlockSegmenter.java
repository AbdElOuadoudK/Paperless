package ats.rparsing.segmentation;

import ats.rparsing.segmentation.config.SegmentationConfig;
import ats.rparsing.segmentation.models.Block;
import ats.rparsing.segmentation.models.Region;
import ats.rparsing.segmentation.utils.BlockDetectionUtils;
import ats.rparsing.segmentation.utils.ContourUtils;

import java.awt.image.BufferedImage;
import java.util.List;
import java.net.URL;
import javax.imageio.ImageIO;
import java.io.File;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.BasicStroke;

public class BlockSegmenter {
    private final SegmentationConfig config;

    public BlockSegmenter(SegmentationConfig config) {
        this.config = config;
    }

    public List<Block> segmentImage(BufferedImage processedImage, BufferedImage image) {
        // Step 2: Detect regions using contours
        List<Region> regions = ContourUtils.detectRegions(processedImage, config);

        // Step 3: Convert regions to blocks
        return BlockDetectionUtils.convertRegionsToBlocks(regions, image);
    }

    private BufferedImage drawBlocksOnImage(BufferedImage originalImage, List<Block> blocks) {
        // Create a copy of the original image to draw on
        BufferedImage outputImage = new BufferedImage(
            originalImage.getWidth(),
            originalImage.getHeight(),
            BufferedImage.TYPE_INT_RGB
        );
        Graphics2D g2d = outputImage.createGraphics();
        g2d.drawImage(originalImage, 0, 0, null);

        // Set line properties
        g2d.setStroke(new BasicStroke(5.0f));
        
        System.out.println("Drawing " + blocks.size() + " blocks:");
        // Draw each block with a different color based on type
        for (Block block : blocks) {
            System.out.println("Block type: " + block.getBlockType());
            System.out.println("Bounds: x=" + block.getBoundingBox().getX() + 
                             ", y=" + block.getBoundingBox().getY() +
                             ", width=" + block.getBoundingBox().getWidth() +
                             ", height=" + block.getBoundingBox().getHeight());
            
            Color blockColor = switch (block.getBlockType().toUpperCase()) {
                case "HEADER" -> new Color(255, 0, 0, 128);
                case "PARAGRAPH" -> new Color(0, 255, 0, 128);
                default -> new Color(0, 0, 255, 128);
            };
            
            g2d.setColor(blockColor);
            // Draw filled rectangle with some transparency
            g2d.fillRect(
                block.getBoundingBox().getX(),
                block.getBoundingBox().getY(),
                block.getBoundingBox().getWidth(),
                block.getBoundingBox().getHeight()
            );
            
            // Draw border in solid color
            g2d.setColor(blockColor.brighter());
            g2d.drawRect(
                block.getBoundingBox().getX(),
                block.getBoundingBox().getY(),
                block.getBoundingBox().getWidth(),
                block.getBoundingBox().getHeight()
            );
        }

        g2d.dispose();
        return outputImage;
    }

    public static void main(String[] args) {
        try {
            // Adjust parameters for better line detection
            SegmentationConfig config = new SegmentationConfig(
                2,     // minBlockSize (adjust this to control spacing thresholds)
                128,    // thresholdValue
                0.01,   // minRegionArea
                15      // edgeDetectionThreshold
            );
            BlockSegmenter blockSegmenter = new BlockSegmenter(config);

            // Load preprocessed test image from resources
            URL resourceUrl = BlockSegmenter.class.getClassLoader().getResource("output_cv_1.png");
            if (resourceUrl == null) {
                System.err.println("Could not find preprocessed image in resources");
                return;
            }
            
            BufferedImage preprocessedImage = ImageIO.read(resourceUrl);
            System.out.println("Loaded preprocessed image: " + 
                             preprocessedImage.getWidth() + "x" + 
                             preprocessedImage.getHeight());

            List<Block> blocks = blockSegmenter.segmentImage(preprocessedImage, preprocessedImage);
            System.out.println("Found " + blocks.size() + " blocks in image");
            
            BufferedImage outputImage = blockSegmenter.drawBlocksOnImage(preprocessedImage, blocks);
            File outputFile = new File("src/main/resources/segmented_cv.png");
            ImageIO.write(outputImage, "png", outputFile);
            System.out.println("Saved output image to: " + outputFile.getAbsolutePath());

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
} 