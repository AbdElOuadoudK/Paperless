package ats.rparsing.extraction;

import net.sourceforge.tess4j.*;
import net.sourceforge.tess4j.TesseractException;

import java.io.File;

public class tts {

    public static void main(String[] args) {
        ITesseract reader = new Tesseract();

        //reader.setTessVariable("C:\\Program Files\\Tesseract-OCR\\tesseract.exe");

        reader.setDatapath("C:\\Users\\windows 10\\Downloads\\Tess4J-3.4.8-src\\Tess4J\\tessdata");
        reader.setLanguage("eng");
        try {
            // Perform OCR on an image file
            File imageFile = new File("C:/Users/windows 10/Documents/GitHub/Paperless/sources/data/resumes/test.png");
            String result = reader.doOCR(imageFile);
            System.out.println(result);
        } catch (TesseractException e) {
            System.err.println("Error occurred: " + e.getMessage());
            /* e.printStackTrace() */
        }
    }
}