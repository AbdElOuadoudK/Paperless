package com.ats;

import com.ats.langdetect.detector.LanguageDetector;
import com.ats.langdetect.detector.Language;
import com.ats.langdetect.util.LanguageDetectionException;
import com.ats.langdetect.util.TextUtils;
import com.ats.langdetect.detector.LinguaDetector;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.io.File;
import java.util.Objects;

public class LangDetectMain {

    // New method to handle single line detection
    private static Language detectSingleLine(String text, LanguageDetector rule_detector, 
            LinguaDetector lingua_detector, boolean verbose, boolean display_tokens) {
        try {
            // Check if text is empty or only whitespace
            if (text == null || text.trim().isEmpty()) {
                return null;
            }

            com.github.pemistahl.lingua.api.Language lingua_detectedLanguage = lingua_detector.detectLanguage(text);
            
            // Add null check for detected language
            if (lingua_detectedLanguage == null) {
                return rule_detector.detectLanguage(TextUtils.preprocessTokens(TextUtils.tokenize(text), display_tokens), verbose);
            }

            double lingua_confidence = lingua_detector.getConfidence(text, lingua_detectedLanguage);
            
            // Preprocess the input text
            List<String> tokens = TextUtils.tokenize(text);
            List<String> processedTokens = TextUtils.preprocessTokens(tokens, display_tokens);

            // Detect language
            Language rule_detectedLanguage = rule_detector.detectLanguage(processedTokens, verbose);

            if (verbose) {
                System.out.printf("LINGUA: Detected Language: %s%n", lingua_detectedLanguage);
                System.out.printf("LINGUA: Confidence: %.2f%%%n%n", lingua_confidence * 100);
            }

            if (lingua_confidence > 0.33) {
                return Language.fromLingua(lingua_detectedLanguage);
            } else {
                return rule_detectedLanguage;
            }
        } catch (LanguageDetectionException e) {

            System.err.println(e.getMessage());
            return null;
        }
    }

    public static Language[] GetLanguage(String[] args, boolean verbose, boolean display_tokens) {


        List<String> textSamples;
        if (args.length > 0) {
            // print reading from file if path provided
            System.out.println("Reading from given file... ");
            try {
                String content = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(args[0])));
                textSamples = Arrays.asList(content.split("\n"));
            } catch (java.io.IOException e) {
                System.err.println("Error reading file: " + e.getMessage());
                return null;
            }
        } else {
            // print reading from test file
            System.out.println("No input file provided! Reading from default file... ");
            File file = new File("src/test/input_text.txt");
            try {
                String content = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(file.getPath())));
                textSamples = Arrays.asList(content.split("\n"));
            } catch (java.io.IOException e) {
                System.err.println("Error reading test file: " + e.getMessage());
                textSamples = null;
            }
            if (!file.exists()) {
                // Prompt user for input if no file provided
                System.out.println("No default file found! Reading from user input... ");
                java.util.Scanner scanner = new java.util.Scanner(System.in);
                String userInput = scanner.nextLine();
                textSamples = Collections.singletonList(userInput);
                scanner.close();
            }
        }

        // Initialize the language detectors
        LanguageDetector rule_detector = new LanguageDetector();
        LinguaDetector lingua_detector = new LinguaDetector();

        // Process each sample and detect its language
        assert textSamples != null;
        Language[] results = new Language[textSamples.size()];
        for (int i = 0; i < textSamples.size(); i++) {
            results[i] = detectSingleLine(textSamples.get(i), rule_detector, 
                    lingua_detector, verbose, display_tokens);
        }
        return results;
    }

    public static void main(String[] args) {
        Language[] detectedLanguages = GetLanguage(args, true, true);
        
        for (int i = 0; i < Objects.requireNonNull(detectedLanguages).length; i++) {
            System.out.printf("Line %d - Detected Language: %s%n", 
                    i + 1, detectedLanguages[i] != null ? detectedLanguages[i].toString() : "Unknown");
        }
    }
}
