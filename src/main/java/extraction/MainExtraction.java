package extraction;

import net.sourceforge.tess4j.TesseractException;
import java.awt.image.BufferedImage;
import java.util.List;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;

public class MainExtraction {

    public MainExtraction() {
        try {
            BufferedImage image = ImageIO.read(new File("src/main/resources/test.png"));
            TextExtractor extractor = new TextExtractor();
            
            // Set Tesseract data path before extraction
            System.setProperty("TESSDATA_PREFIX", "C:/Program Files/Tesseract-OCR/tessdata");
            
            try {
                List<TextBlock> blocks = extractor.extractTextBlocks(image);
                for (TextBlock block : blocks) {
                    System.out.println(block.toString());
                    System.out.println("Type: " + block.getType());
                    System.out.println("Content: " + block.getContent());
                }
            } catch (TesseractException e) {
                System.err.println("Error extracting text: " + e.getMessage());
                e.printStackTrace();
            }
        } catch (IOException e) {
            System.err.println("Error reading image file: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new MainExtraction();
    }
}
