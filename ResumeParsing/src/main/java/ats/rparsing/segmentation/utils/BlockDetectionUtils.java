package ats.rparsing.segmentation.utils;

import ats.rparsing.segmentation.models.Block;
import ats.rparsing.segmentation.models.BoundingBox;
import ats.rparsing.segmentation.models.Region;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.stream.Collectors;

public class BlockDetectionUtils {
    
    public static List<Block> convertRegionsToBlocks(List<Region> regions, BufferedImage image) {
        return regions.stream()
            .map(region -> {
                BoundingBox box = region.getBoundingBox();
                String content = extractContent(image, box);
                String blockType = determineBlockType(region);
                return new Block(box, blockType, content);
            })
            .collect(Collectors.toList());
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