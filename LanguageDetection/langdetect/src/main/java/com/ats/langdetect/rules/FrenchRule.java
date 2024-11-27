package com.ats.langdetect.rules;

import java.util.List;
import java.util.regex.Pattern;

import com.ats.langdetect.util.TextUtils;

public class FrenchRule {
    private static final String filename = "french_common_words.txt";
    private static final String regex;
    static {
        List<String> stopwords = TextUtils.parseStopwords(filename);

        if (!stopwords.isEmpty()) {
            regex = TextUtils.buildRegex(stopwords);
        } else {
            System.err.println("No tokens found in french_common_words");
            regex = null;
        }
    }
    // Regular expressions for common French patterns
    private static final Pattern COMMON_FRENCH_WORDS;

    static {
        assert regex != null;
        COMMON_FRENCH_WORDS = Pattern.compile(regex, Pattern.CASE_INSENSITIVE);
    }

    private static final Pattern FRENCH_ACCENTS = Pattern.compile(".*[éèêëàâîïçôûù].*");
    private static final Pattern FRENCH_SENTENCE_STRUCTURE = Pattern.compile("^[A-Z][a-zàâçéèêëîïôûùüÿñæœ]*(\\s[a-zàâçéèêëîïôûùüÿñæœ]+)*[.!?]?$", Pattern.CASE_INSENSITIVE);

    /**
     * Checks if a given token matches French language patterns.
     * @param token The text token to evaluate.
     * @return true if the token matches French patterns; false otherwise.
     */
    public boolean matches(String token) {
        // Check if the token contains common French words
        if (COMMON_FRENCH_WORDS.matcher(token).find()) {
            return true;
        }

        // Check if the token contains French accent characters
        if (FRENCH_ACCENTS.matcher(token).matches()) {
            return true;
        }

        // Check if the token follows a typical French sentence structure
        return FRENCH_SENTENCE_STRUCTURE.matcher(token).matches();

        // If no patterns matched, return false
    }
}
