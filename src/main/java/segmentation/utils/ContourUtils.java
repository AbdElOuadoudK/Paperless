package segmentation.utils;

import segmentation.models.Region;
import segmentation.models.Block;
import segmentation.models.BoundingBox;
import segmentation.config.SegmentationConfig;
import segmentation.exceptions.SegmentationException;
import segmentation.BlockSegmenter;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import java.awt.Point;

public class ContourUtils {

    public static List<Region> detectRegions(BufferedImage image, SegmentationConfig config) {
        List<Region> regions = new ArrayList<>();
        boolean[][] visited = new boolean[image.getHeight()][image.getWidth()];
        
        // First pass: detect characters
        List<Region> characterRegions = detectCharacterRegions(image, visited, config);
        
        // Second pass: group characters into lines
        List<Region> lineRegions = groupIntoLines(characterRegions, config);
        
        // Third pass: group lines into blocks
        return lineRegions; //groupIntoBlocks(lineRegions, config);
    }

    private static List<Region> detectCharacterRegions(BufferedImage image, boolean[][] visited, SegmentationConfig config) {
        List<Region> regions = new ArrayList<>();
        int borderMargin = 2;
        
        for (int y = borderMargin; y < image.getHeight() - borderMargin; y++) {
            for (int x = borderMargin; x < image.getWidth() - borderMargin; x++) {
                if (!visited[y][x] && isBlackPixel(image, x, y)) {
                    Region region = traceRegion(image, visited, x, y, config);
                    if (isValidCharacterRegion(region, image.getWidth(), image.getHeight())) {
                        regions.add(region);
                    }
                }
            }
        }
        
        return regions;
    }

    private static List<Region> groupIntoLines(List<Region> characterRegions, SegmentationConfig config) {
        List<Region> lines = new ArrayList<>();
        List<Region> sortedRegions = new ArrayList<>(characterRegions);
        
        // Sort regions by y-coordinate first, then by x-coordinate
        sortedRegions.sort((r1, r2) -> {
            int yDiff = r1.getBoundingBox().getY() - r2.getBoundingBox().getY();
            return yDiff != 0 ? yDiff : r1.getBoundingBox().getX() - r2.getBoundingBox().getX();
        });
        
        List<Region> currentLine = new ArrayList<>();
        int lastY = -1;
        int lastX = -1;
        int horizontalSpacing = config.getMinBlockSize() * 72;  // Base horizontal threshold
        int idx=0;
        for (Region region : sortedRegions) {
            int currentY = region.getBoundingBox().getY();
            int currentX = region.getBoundingBox().getX();
            
            // Calculate dynamic vertical spacing based on the height of the current region
            int verticalSpacing = (int) (region.getBoundingBox().getHeight() * 0.4); // 1.5 times the height

            boolean shouldStartNewLine = 
                lastY == -1 || // First region
                Math.abs(currentX - lastX) > horizontalSpacing || // Too far horizontally
                Math.abs(currentY - lastY) > verticalSpacing;  // Too far vertically
                
            System.out.println("idx: " + String.valueOf(idx) + " | " + String.valueOf(Math.abs(currentX - lastX) > horizontalSpacing) + " | " + String.valueOf(Math.abs(currentY - lastY) > verticalSpacing) );
            
            if (shouldStartNewLine) {
                if (!currentLine.isEmpty()) {
                    lines.add(mergeLine(currentLine));
                    currentLine.clear();
                }
                currentLine.add(region);
                try{
                    if (false){
                        debugginDraw(lines, idx, "lines-");
                        debugginDraw(currentLine, idx, "currentLine-");
                    }
                }
                catch(IOException e){
                    e.printStackTrace();
                }
                idx++;
            } else {
                currentLine.add(region);
                try{
                    if (false){
                        debugginDraw(lines, idx, "lines-");
                        debugginDraw(currentLine, idx, "currentLine-");
                    }
                }
                catch(IOException e){
                    e.printStackTrace();
                }
                idx++;
            }
            
            lastY = currentY;
            lastX = currentX + region.getBoundingBox().getWidth();
        }
        
        if (!currentLine.isEmpty()) {
            lines.add(mergeLine(currentLine));
        }
        
        return groupOverlapped(lines);
    }

    private static List<Region> groupIntoBlocks(List<Region> lineRegions, SegmentationConfig config) {
        List<Region> blocks = new ArrayList<>();
        List<Region> currentBlock = new ArrayList<>();
        int lastY = -1;
        int blockSpacing = config.getMinBlockSize() * 4;  // Larger spacing for blocks
        
        for (Region line : lineRegions) {
            int currentY = line.getBoundingBox().getY();
            
            if (lastY == -1 || Math.abs(currentY - lastY) <= blockSpacing) {
                currentBlock.add(line);
            } else {
                if (!currentBlock.isEmpty()) {
                    blocks.add(mergeLine(currentBlock));  // Reusing mergeLine since the logic is similar
                    currentBlock.clear();
                }
                currentBlock.add(line);
            }
            lastY = currentY;
        }
        
        if (!currentBlock.isEmpty()) {
            blocks.add(mergeLine(currentBlock));
        }
        
        return blocks;
    }

    private static Region mergeLine(List<Region> lineRegions) {
        int minX = Integer.MAX_VALUE;
        int minY = Integer.MAX_VALUE;
        int maxX = 0;
        int maxY = 0;
        List<Point> allPoints = new ArrayList<>();
        
        for (Region region : lineRegions) {
            BoundingBox box = region.getBoundingBox();
            minX = Math.min(minX, box.getX());
            minY = Math.min(minY, box.getY());
            maxX = Math.max(maxX, box.getX() + box.getWidth());
            maxY = Math.max(maxY, box.getY() + box.getHeight());
            allPoints.addAll(region.getContourPoints());
        }
        
        BoundingBox lineBox = new BoundingBox(minX, minY, maxX - minX, maxY - minY);
        return new Region(lineBox, allPoints.size(), allPoints);
    }

    private static boolean isValidCharacterRegion(Region region, int imageWidth, int imageHeight) {
        BoundingBox box = region.getBoundingBox();
        double aspectRatio = (double) box.getWidth() / box.getHeight();
        double relativeArea = (double) (box.getWidth() * box.getHeight()) / (imageWidth * imageHeight);
        
        return aspectRatio < 7 && // Not too wide
               aspectRatio > 0.121 && // Not too tall
               relativeArea < 0.01 && // Not too large
               relativeArea > 3e-6; // Not too small
    }

    private static boolean isBlackPixel(BufferedImage image, int x, int y) {
        return (image.getRGB(x, y) & 0xFF) < 200;
    }

    private static Region traceRegion(BufferedImage image, boolean[][] visited, 
                                    int startX, int startY, SegmentationConfig config) {
        List<Point> points = new ArrayList<>();
        int[] bounds = new int[]{startX, startX, startY, startY}; // minX, maxX, minY, maxY
        
        // Implement flood fill algorithm to trace the region
        floodFill(image, visited, startX, startY, points, bounds);
        
        // Create bounding box using the updated bounds
        int width = bounds[1] - bounds[0] + 1;  // Add 1 to include both edges
        int height = bounds[3] - bounds[2] + 1;  // Add 1 to include both edges
        
        BoundingBox box = new BoundingBox(bounds[0], bounds[2], width, height);
        return new Region(box, points.size(), points);
    }

    private static void floodFill(BufferedImage image, boolean[][] visited,
                            int x, int y, List<Point> points, int[] bounds) {
        List<Point> stack = new ArrayList<>();
        stack.add(new Point(x, y));
        
        int[][] directions = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        
        while (!stack.isEmpty()) {
            Point current = stack.remove(stack.size() - 1);
            x = current.x;
            y = current.y;
            
            if (x < 0 || x >= image.getWidth() || y < 0 || y >= image.getHeight() 
                || visited[y][x] || !isBlackPixel(image, x, y)) {
                continue;
            }
            
            // Mark as visited and add to region points
            visited[y][x] = true;
            points.add(current);
            
            // Update bounds
            bounds[0] = Math.min(bounds[0], x); // minX
            bounds[1] = Math.max(bounds[1], x); // maxX
            bounds[2] = Math.min(bounds[2], y); // minY
            bounds[3] = Math.max(bounds[3], y); // maxY
            
            // Add unvisited neighbors to stack
            for (int[] dir : directions) {
                int newX = x + dir[0];
                int newY = y + dir[1];
                if (newX >= 0 && newX < image.getWidth() && 
                    newY >= 0 && newY < image.getHeight()) {
                    stack.add(new Point(newX, newY));
                }
            }
        }
    }

    private static List<Region> mergeCloseRegions(List<Region> regions) {
        List<Region> mergedRegions = new ArrayList<>();
        boolean merged;

        do {
            merged = false;
            List<Region> tempRegions = new ArrayList<>(regions); // Copy the current regions

            for (int i = 0; i < tempRegions.size(); i++) {
                Region regionA = tempRegions.get(i);
                for (int j = i + 1; j < tempRegions.size(); j++) {
                    Region regionB = tempRegions.get(j);
                    Region mergedRegion = mergeIfClose(regionA, regionB, 100);
                    
                    if (mergedRegion != null) {
                        // If merged, update the list and set the flag
                        mergedRegions.add(mergedRegion);
                        merged = true;
                        // Remove the merged regions from the temp list
                        tempRegions.remove(j); // Remove regionB
                        tempRegions.remove(i); // Remove regionA
                        tempRegions.add(mergedRegion); // Add the new merged region
                        break; // Break to restart the outer loop
                    }
                }
                if (merged) {
                    break; // Break the outer loop if a merge occurred
                }
            }

            // If no merges occurred, add remaining regions to the merged list
            if (!merged) {
                mergedRegions.addAll(tempRegions);
            }

            // Update regions for the next iteration
            regions = new ArrayList<>(mergedRegions);
        } while (merged); // Continue until no more merges occur

        return mergedRegions;
    }

    private static Region mergeIfClose(Region regionA, Region regionB, int threshold) {
        BoundingBox boxA = regionA.getBoundingBox();
        BoundingBox boxB = regionB.getBoundingBox();

        // Calculate the coordinates
        int beginX_A = boxA.getX();
        int endX_A = boxA.getX() + boxA.getWidth();
        int beginY_A = boxA.getY();
        int endY_A = boxA.getY() + boxA.getHeight();

        int beginX_B = boxB.getX();
        int endX_B = boxB.getX() + boxB.getWidth();
        int beginY_B = boxB.getY();
        int endY_B = boxB.getY() + boxB.getHeight();

        // Check if the regions are on the same line and if they are close enough
        boolean areOnSameLine = Math.abs(beginY_A - beginY_B) < 10; // Allow some tolerance in the y-axis
        boolean areClose = Math.abs(beginX_B - endX_A) <= threshold; // Check if the right edge of A is close to the left edge of B

        if (areOnSameLine && areClose) {
            // Merge the two regions
            List<Point> combinedPoints = new ArrayList<>(regionA.getContourPoints());
            combinedPoints.addAll(regionB.getContourPoints());

            // Create a new bounding box that encompasses both regions
            BoundingBox newBoundingBox = new BoundingBox(
                Math.min(boxA.getX(), boxB.getX()),
                Math.min(boxA.getY(), boxB.getY()),
                Math.max(endX_A, endX_B) - Math.min(boxA.getX(), boxB.getX()),
                Math.max(endY_A, endY_B) - Math.min(boxA.getY(), boxB.getY())
            );

            return new Region(newBoundingBox, combinedPoints.size(), combinedPoints);
        }

        return null; // Return null if regions are not merged
    }

    private static void debugginDraw(List<Region> regions, int idx, String nature) throws IOException {//////////////////////////
        // Adjust parameters for better line detection
        SegmentationConfig config = new SegmentationConfig(
            1,     // minBlockSize
            128,   // thresholdValue
            0.01,  // minRegionArea
            15     // edgeDetectionThreshold
        );
        BlockSegmenter blockSegmenter = new BlockSegmenter(config);
        URL resourceUrl = BlockSegmenter.class.getClassLoader().getResource("debug/acq-DEBUG-5.png");
        if (resourceUrl == null) {
            return;
        }
        
        try {
            BufferedImage image = ImageIO.read(resourceUrl);
            List<Block> blocks = BlockDetectionUtils.convertRegionsToBlocks(regions, image);
            BufferedImage outputImage = blockSegmenter.drawBlocksOnImage(image, blocks);
            File outputFile = new File("src/main/resources/debug/seg-DEBUG-" + nature + String.valueOf(idx) + ".png");
            ImageIO.write(outputImage, "png", outputFile);
        } catch (SegmentationException e) {
            // Handle the exception (e.g., log it or rethrow it)
            e.printStackTrace(); // or use a logger
        } catch (IOException e) {
            // Handle IOException from ImageIO.write
            e.printStackTrace(); // or use a logger
        }
    }           

    private static List<Region> groupOverlapped(List<Region> lineRegions) {
        List<Region> mergedLines = new ArrayList<>();
        boolean merged;

        do {
            merged = false;
            List<Region> newMergedLines = new ArrayList<>();

            for (Region currentLine : lineRegions) {
                boolean isMerged = false;
                BoundingBox currentBox = currentLine.getBoundingBox();

                for (int i = 0; i < newMergedLines.size(); i++) {
                    Region mergedLine = newMergedLines.get(i);
                    BoundingBox mergedBox = mergedLine.getBoundingBox();

                    // Check if the current line overlaps with the merged line
                    if (currentBox.intersects(mergedBox)) {
                        // Merge the two lines
                        List<Point> combinedPoints = new ArrayList<>(mergedLine.getContourPoints());
                        combinedPoints.addAll(currentLine.getContourPoints());

                        // Create a new merged region
                        BoundingBox newMergedBox = new BoundingBox(
                            Math.min(currentBox.getX(), mergedBox.getX()),
                            Math.min(currentBox.getY(), mergedBox.getY()),
                            Math.max(currentBox.getX() + currentBox.getWidth(), mergedBox.getX() + mergedBox.getWidth()) - Math.min(currentBox.getX(), mergedBox.getX()),
                            Math.max(currentBox.getY() + currentBox.getHeight(), mergedBox.getY() + mergedBox.getHeight()) - Math.min(currentBox.getY(), mergedBox.getY())
                        );

                        Region newMergedRegion = new Region(newMergedBox, combinedPoints.size(), combinedPoints);
                        newMergedLines.set(i, newMergedRegion); // Replace the old merged line with the new one
                        merged = true; // Indicate that a merge has occurred
                        isMerged = true;
                        break;
                    }
                }

                // If no merge happened, add the current line as a new entry
                if (!isMerged) {
                    newMergedLines.add(currentLine);
                }
            }

            lineRegions = newMergedLines; // Update lineRegions for the next iteration
        } while (merged); // Continue until no more merges occur

        return lineRegions;
    }
} 