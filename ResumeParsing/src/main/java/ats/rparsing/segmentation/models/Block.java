package ats.rparsing.segmentation.models;

public class Block {
    private final BoundingBox boundingBox;
    private final String blockType;
    private final String content;

    public Block(BoundingBox boundingBox, String blockType, String content) {
        this.boundingBox = boundingBox;
        this.blockType = blockType;
        this.content = content;
    }

    // Getters
    public BoundingBox getBoundingBox() { return boundingBox; }
    public String getBlockType() { return blockType; }
    public String getContent() { return content; }
} 