package ats.rparsing.acquisition;

import ats.rparsing.utils.FileConversionException;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
//import org.apache.poi.xwpf.usermodel.XWPFDocument;
//import org.apache.poi.hwpf.HWPFDocument;
//import org.apache.poi.xslf.usermodel.XMLSlideShow;
//import org.apache.poi.sl.draw.Drawable;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;


public class FileConverter {
    public List<BufferedImage> convertToImage(File file) throws Exception {
        String fileName = file.getName().toLowerCase();
        System.out.println("Converting " + fileName + " to images...");
        System.out.println("File size: " + file.length() + " bytes");

        if (fileName.endsWith(".pdf")) {
            System.out.println("PDF file detected, attempting conversion...");
            try (PDDocument document = PDDocument.load(file)) {
                return splitPagesIntoImages(document);
            } catch (Exception e) {
                throw new FileConversionException("Failed to convert PDF: " + e.getMessage(), e);
            }
        } else if (FileHandler.SUPPORTED_EXTENSIONS.stream()
                .anyMatch(ext -> fileName.toLowerCase().endsWith("." + ext))) {
            System.out.println("Image file detected, loading directly...");
            try {
                List<BufferedImage> images = new ArrayList<>();
                BufferedImage image = null;
                try {
                    image = ImageIO.read(new FileInputStream(file));
                } catch (Exception e) {
                    System.out.println("First attempt failed, trying alternate method...");
                    try (InputStream is = new FileInputStream(file)) {
                        byte[] bytes = is.readAllBytes();
                        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes)) {
                            image = ImageIO.read(bis);
                        }
                    }
                }
                
                if (image == null) {
                    throw new FileConversionException("Failed to read image file: " + fileName + 
                        ". Supported formats are: " + String.join(", ", ImageIO.getReaderFormatNames()));
                }
                images.add(image);
                return images;
            } catch (Exception e) {
                throw new FileConversionException("Failed to convert image: " + e.getMessage() + 
                    "\nFile path: " + file.getAbsolutePath(), e);
            }
        } else {
            System.out.println("Unsupported file format detected: " + fileName);
            throw new UnsupportedOperationException("Unsupported file format: " + fileName);
        }
    }

    private List<BufferedImage> splitPagesIntoImages(PDDocument document) throws Exception {
        System.out.println("Starting PDF page splitting process...");
        List<BufferedImage> images = new ArrayList<>();
        PDFRenderer pdfRenderer = new PDFRenderer(document);

        for (int page = 0; page < document.getNumberOfPages(); page++) {
            BufferedImage image = pdfRenderer.renderImageWithDPI(page, 300); // 300 DPI for good quality
            images.add(image);
        }
        System.out.println("PDF splitting completed!");
        return images;
    }
}

