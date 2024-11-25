package ats.rparsing.extraction;

import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class TextExtractor {
    private final Tesseract tesseract;
    
    public TextExtractor() {
        this.tesseract = new Tesseract();
        initializeTesseract();
    }
    
    private void initializeTesseract() {
        // Configure Tesseract settings
        tesseract.setLanguage("eng + fra + ara"); // Set language - you can add more languages if needed
        tesseract.setPageSegMode(1); // Automatic page segmentation with OSD
        tesseract.setOcrEngineMode(1); // Neural net LSTM engine
    }
    
    public String extractText(BufferedImage image) throws TesseractException {
        if (image == null) {
            throw new IllegalArgumentException("Input image cannot be null");
        }
        return tesseract.doOCR(image);
    }
    
    public List<TextBlock> extractTextBlocks(BufferedImage image) throws TesseractException {
        String extractedText = extractText(image);
        return parseTextBlocks(extractedText);
    }
    
    private List<TextBlock> parseTextBlocks(String text) {
        List<TextBlock> blocks = new ArrayList<>();
        String[] paragraphs = text.split("\n\n");
        
        for (String paragraph : paragraphs) {
            if (!paragraph.trim().isEmpty()) {
                blocks.add(new TextBlock(paragraph.trim()));
            }
        }
        return blocks;
    }
} 