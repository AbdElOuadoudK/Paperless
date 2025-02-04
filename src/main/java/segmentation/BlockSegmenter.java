package segmentation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import segmentation.config.SegmentationConfig;
import segmentation.models.Block;
import segmentation.models.Region;
import segmentation.utils.BlockDetectionUtils;
import segmentation.utils.ContourUtils;
import segmentation.exceptions.SegmentationException;

import java.awt.image.BufferedImage;
import java.util.List;
import java.net.URL;
import javax.imageio.ImageIO;
import java.io.File;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.BasicStroke;

public class BlockSegmenter {
    private static final Logger logger = LoggerFactory.getLogger(BlockSegmenter.class);
    private final SegmentationConfig config;

    public BlockSegmenter(SegmentationConfig config) {
        this.config = config;
    }

    public List<Block> segmentImage(BufferedImage processedImage, BufferedImage image) throws SegmentationException {
        try {
            // Step 2: Detect regions using contours
            List<Region> regions = ContourUtils.detectRegions(processedImage, config);
            // Step 3: Convert regions to blocks
            return BlockDetectionUtils.convertRegionsToBlocks(regions, image);
        } catch (Exception e) {
            logger.error("Error during image segmentation: {}", e.getMessage(), e);
            throw new SegmentationException("Failed to segment image", e);
        }
    }

    public BufferedImage drawBlocksOnImage(BufferedImage originalImage, List<Block> blocks) {
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
        
        // Draw each block with a different color based on type
        for (Block block : blocks) {
            //logger.info("Block type: {}", block.getBlockType());
            //logger.info("Bounds: x={}, y={}, width={}, height={}", 
            //            block.getBoundingBox().getX(), 
            //            block.getBoundingBox().getY(), 
            //            block.getBoundingBox().getWidth(), 
             //           block.getBoundingBox().getHeight());
            
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
                1,     // minBlockSize
                128,   // thresholdValue
                0.01,  // minRegionArea
                15     // edgeDetectionThreshold
            );
            BlockSegmenter blockSegmenter = new BlockSegmenter(config);

            // Load preprocessed test image from resources
            URL resourceUrl = BlockSegmenter.class.getClassLoader().getResource("debug/acq-DEBUG-3.png"); //////////////////////////
            if (resourceUrl == null) {
                logger.error("Could not find preprocessed image.");
                return;
            }
            
            BufferedImage preprocessedImage = ImageIO.read(resourceUrl);
            logger.info("Loaded preprocessed image: {}x{}", 
                        preprocessedImage.getWidth(), 
                        preprocessedImage.getHeight());

            List<Block> blocks = blockSegmenter.segmentImage(preprocessedImage, preprocessedImage);
            logger.info("Found {} blocks in image", blocks.size());
            
            BufferedImage outputImage = blockSegmenter.drawBlocksOnImage(preprocessedImage, blocks);
            File outputFile = new File("src/main/resources/debug/seg-DEBUG-3.png");//////////////////////////
            ImageIO.write(outputImage, "png", outputFile);
            logger.info("Saved output image to: {}", outputFile.getAbsolutePath());

        } catch (SegmentationException e) {
            logger.error("Segmentation error: {}", e.getMessage(), e);
        } catch (Exception e) {
            logger.error("Unexpected error: {}", e.getMessage(), e);
        }
    }
} 