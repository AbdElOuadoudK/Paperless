package extraction;

public class TextBlock {
    private final String content;
    private TextBlockType type;
    
    public TextBlock(String content) {
        this.content = content;
        this.type = determineType(content);
    }
    
    private TextBlockType determineType(String content) {
        // Basic type determination logic - you can enhance this based on your needs
        if (content.length() < 50 && content.toUpperCase().equals(content)) {
            return TextBlockType.HEADING;
        } else if (content.matches(".*\\d+.*")) {
            return TextBlockType.NUMERIC_CONTENT;
        } else {
            return TextBlockType.PARAGRAPH;
        }
    }
    
    public String getContent() {
        return content;
    }
    
    public TextBlockType getType() {
        return type;
    }
    
    public void setType(TextBlockType type) {
        this.type = type;
    }
} 