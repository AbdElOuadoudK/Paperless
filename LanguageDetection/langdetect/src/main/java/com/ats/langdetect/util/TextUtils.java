package com.ats.langdetect.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import com.ibm.icu.text.BreakIterator;
import java.util.Locale;

public class TextUtils {

    /**
     * Removes punctuation from a given text string.
     * @param text The input string to process.
     * @return The text string without punctuation.
     */
    public static String removePunctuation(String text) {
        return text.replaceAll("\\p{Punct}", "");
    }

    /**
     * Converts a given text string to lowercase.
     * @param text The input string to convert.
     * @return The text string in lowercase.
     */
    public static String toLowerCase(String text) {
        return text.toLowerCase();
    }

    /**
     * Splits a given text into tokens (words) using Lingui/ICU
     * @param text The input string to split.
     * @return A list of tokens (words).
     */
    public static List<String> tokenize(String text) {
        List<String> tokenList = new ArrayList<>();
        BreakIterator wordIterator = BreakIterator.getWordInstance(Locale.ROOT);
        wordIterator.setText(text);
        
        int start = wordIterator.first();
        int end = wordIterator.next();
        
        while (end != BreakIterator.DONE) {
            String token = text.substring(start, end).trim();
            if (!token.isEmpty() && Character.isLetterOrDigit(token.charAt(0))) {
                tokenList.add(token);
            }
            start = end;
            end = wordIterator.next();
        }
        return tokenList;
    }

    /**
     * Preprocesses a list of text tokens by removing punctuation and converting to lowercase.
     * @param tokens A list of text tokens to preprocess.
     * @return A list of cleaned tokens.
     */
    public static List<String> preprocessTokens(List<String> tokens, boolean display_tokens) {
        List<String> processedTokens = new ArrayList<>();
        for (String token : tokens) {
            String cleanedToken = removePunctuation(token);
            cleanedToken = toLowerCase(cleanedToken);
            if (!cleanedToken.isEmpty()) {
                processedTokens.add(cleanedToken);
            }
        }
        if (display_tokens) {
            System.out.println("TOKENS: " + processedTokens);
        }
        return processedTokens;
    }

    // Method to parse stopwords from file and return as a list
    public static List<String> parseStopwords(String filename) {
        List<String> stopwords = new ArrayList<>();
        
        try (InputStream is = TextUtils.class.getClassLoader().getResourceAsStream(filename)) {
            assert is != null;
            try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {

                    // Remove trailing commas and whitespace, then add to list
                    String token = line.trim().replace(",", "");
                    if (!token.isEmpty()) {
                        stopwords.add(token);
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());

        } catch (NullPointerException e) {
            System.err.println("Could not find file in classpath: " + filename);
        }

        return stopwords;
    }

    // Method to build regex pattern from stopwords list
    public static String buildRegex(List<String> stopwords) {
        // Join all stopwords with "|" and add word boundaries
        return "\\b(" + String.join("|", stopwords) + ")\\b";
    }

}
