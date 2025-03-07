package com.ats.langdetect.rules;

import com.ats.langdetect.util.TextUtils;

import java.util.List;
import java.util.regex.Pattern;

public class EnglishRule {
    private static final String filename = "english_common_words.txt";
    private static final String regex;
    static {
        List<String> stopwords = TextUtils.parseStopwords(filename);

        if (!stopwords.isEmpty()) {
            regex = TextUtils.buildRegex(stopwords);
        } else {
            System.err.println("No tokens found in english_common_words");
            regex = null;
        }
    }
    // Regular expressions for common English patterns
    private static final Pattern COMMON_ENGLISH_WORDS;

    static {
        assert regex != null;
        COMMON_ENGLISH_WORDS = Pattern.compile(regex, Pattern.CASE_INSENSITIVE);
    }

    private static final Pattern ENGLISH_CHARACTERS = Pattern.compile("^[a-zA-Z]+$");
    private static final Pattern ENGLISH_SENTENCE_STRUCTURE = Pattern.compile("^[A-Z][a-z]*(\\s[a-z]+)*[.!?]?$");

    /**
     * Checks if a given token matches English language patterns.
     * @param token The text token to evaluate.
     * @return true if the token matches English patterns; false otherwise.
     */
    public boolean matches(String token) {
        // Check if the token contains common English words
        if (COMMON_ENGLISH_WORDS.matcher(token).find()) {
            return true;
        }

        // Check if the token consists primarily of English alphabet characters
        if (ENGLISH_CHARACTERS.matcher(token).matches()) {
            return true;
        }

        // Check if the token follows a simple English sentence structure (capitalized start, lowercase words, and punctuation at the end)
        return ENGLISH_SENTENCE_STRUCTURE.matcher(token).matches();

        // If no patterns matched, return false
    }
}
