package com.ats.langdetect.rules;

import java.util.List;
import java.util.regex.Pattern;

import com.ats.langdetect.util.TextUtils;

public class ArabicRule {
    private static final String filename = "arabic_common_words.txt";
    private static final String regex;
    static {
        List<String> stopwords = TextUtils.parseStopwords(filename);

        if (!stopwords.isEmpty()) {
            regex = TextUtils.buildRegex(stopwords);
        } else {
            System.err.println("No tokens found in arabic_common_words");
            regex = null;
        }
    }
    // Regular expressions for common Arabic patterns
    private static final Pattern COMMON_ARABIC_WORDS;

    static {
        assert regex != null;
        COMMON_ARABIC_WORDS = Pattern.compile(regex);
    }

    private static final Pattern ARABIC_CHARACTERS = Pattern.compile("[\\u0600-\\u06FF]");
    private static final Pattern ARABIC_SENTENCE_STRUCTURE = Pattern.compile("^[\\u0600-\\u06FF\\s]+[.،؟]?$");

    /**
     * Checks if a given token matches Arabic language patterns.
     * @param token The text token to evaluate.
     * @return true if the token matches Arabic patterns; false otherwise.
     */
    public boolean matches(String token) {
        // Check if the token contains common Arabic words
        if (COMMON_ARABIC_WORDS.matcher(token).find()) {
            return true;
        }

        // Check if the token contains Arabic script characters
        if (ARABIC_CHARACTERS.matcher(token).find()) {
            return true;
        }

        // Check if the token follows a simple Arabic sentence structure
        return ARABIC_SENTENCE_STRUCTURE.matcher(token).matches();

        // If no patterns matched, return false
    }
}
