package segmentation.models;

public class BoundingBox {
    private final int x;
    private final int y;
    private final int width;
    private final int height;

    public BoundingBox(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    // Getters and utility methods
    public int getX() { return x; }
    public int getY() { return y; }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    
    public boolean intersects(BoundingBox other) {
        return !(other.x > (x + width) || 
                (other.x + other.width) < x || 
                other.y > (y + height) ||
                (other.y + other.height) < y);
    }

    public boolean contains(BoundingBox other) {
        return (this.x <= other.x) &&
               (this.y <= other.y) &&
               (this.x + this.width >= other.x + other.width) &&
               (this.y + this.height >= other.y + other.height);
    }
} 