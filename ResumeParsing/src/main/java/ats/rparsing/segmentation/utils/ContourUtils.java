package ats.rparsing.segmentation.utils;

import ats.rparsing.segmentation.models.Region;
import ats.rparsing.segmentation.models.BoundingBox;
import ats.rparsing.segmentation.config.SegmentationConfig;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
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
        return groupIntoBlocks(lineRegions, config);
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
        int verticalSpacing = config.getMinBlockSize() * 4;    // Base vertical threshold
        int horizontalSpacing = config.getMinBlockSize() * 1;  // Base horizontal threshold
        int minGap = 2; // Minimum gap in pixels to consider lines separate

        for (Region region : sortedRegions) {
            int currentY = region.getBoundingBox().getY();
            int currentX = region.getBoundingBox().getX();
            
            // Calculate dynamic vertical spacing based on the height of the current region
            int dynamicVerticalSpacing = (int) (region.getBoundingBox().getHeight() * 1.5); // 1.5 times the height

            boolean shouldStartNewLine = 
                lastY == -1 || // First region
                Math.abs(currentY - lastY) > Math.max(verticalSpacing, dynamicVerticalSpacing) || // Too far vertically
                (currentX - lastX) > horizontalSpacing || // Too far horizontally
                (currentY - lastY) > minGap; // Minimum gap check
            
            if (shouldStartNewLine) {
                if (!currentLine.isEmpty()) {
                    lines.add(mergeLine(currentLine));
                    currentLine.clear();
                }
                currentLine.add(region);
            } else {
                currentLine.add(region);
            }
            
            lastY = currentY;
            lastX = currentX + region.getBoundingBox().getWidth();
        }
        
        if (!currentLine.isEmpty()) {
            lines.add(mergeLine(currentLine));
        }
        
        return lines;
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
        
        return aspectRatio < 3.0 && // Not too wide
               aspectRatio > 0.2 && // Not too tall
               relativeArea < 0.01 && // Not too large
               relativeArea > 0.00001; // Not too small
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
        boolean[] merged = new boolean[regions.size()];
        
        for (int i = 0; i < regions.size(); i++) {
            if (merged[i]) continue;
            
            Region currentRegion = regions.get(i);
            BoundingBox currentBox = currentRegion.getBoundingBox();
            List<Point> mergedPoints = new ArrayList<>(currentRegion.getContourPoints());
            
            // Look for regions to merge
            for (int j = i + 1; j < regions.size(); j++) {
                if (merged[j]) continue;
                
                Region otherRegion = regions.get(j);
                BoundingBox otherBox = otherRegion.getBoundingBox();
                
                // Check if regions are close or overlapping
                if (areRegionsClose(currentBox, otherBox, 20)) { // 20 pixels threshold
                    mergedPoints.addAll(otherRegion.getContourPoints());
                    merged[j] = true;
                }
            }
            
            // Create new merged region
            if (!merged[i]) {
                mergedRegions.add(currentRegion);
            }
        }
        
        return mergedRegions;
    }

    private static boolean areRegionsClose(BoundingBox box1, BoundingBox box2, int threshold) {
        return Math.abs(box1.getX() - box2.getX()) < threshold ||
               Math.abs(box1.getY() - box2.getY()) < threshold ||
               box1.intersects(box2);
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
} 