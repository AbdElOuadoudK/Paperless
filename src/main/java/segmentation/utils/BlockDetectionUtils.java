package segmentation.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import segmentation.models.Block;
import segmentation.models.BoundingBox;
import segmentation.models.Region;
import segmentation.exceptions.SegmentationException;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.stream.Collectors;

public class BlockDetectionUtils {
    private static final Logger logger = LoggerFactory.getLogger(BlockDetectionUtils.class);
    
    public static List<Block> convertRegionsToBlocks(List<Region> regions, BufferedImage image) throws SegmentationException {
        try {
            return regions.stream()
                .map(region -> {
                    BoundingBox box = region.getBoundingBox();
                    String content = extractContent(image, box);
                    String blockType = determineBlockType(region);
                    logger.debug("Converted region to block: {}", blockType);
                    return new Block(box, blockType, content);
                })
                .collect(Collectors.toList());
        } catch (Exception e) {
            logger.error("Error converting regions to blocks: {}", e.getMessage(), e);
            throw new SegmentationException("Failed to convert regions to blocks", e);
        }
    }

    private static String determineBlockType(Region region) {
        BoundingBox box = region.getBoundingBox();
        double aspectRatio = (double) box.getWidth() / box.getHeight();
        
        if (aspectRatio > 2.0) {
            return "LINE"; // Single line of text
        } else if (box.getHeight() < 10) {
            return "HEADER";
        } else {
            return "PARAGRAPH";
        }
    }

    private static String extractContent(BufferedImage image, BoundingBox box) {
        // This would integrate with OCR in the future
        // For now, return placeholder
        return "Content from region: " + box.toString();
    }
} 