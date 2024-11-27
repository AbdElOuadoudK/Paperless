package com.ats.langdetect.detector;

import com.ats.langdetect.rules.EnglishRule;
import com.ats.langdetect.rules.FrenchRule;
import com.ats.langdetect.rules.ArabicRule;
import com.ats.langdetect.util.LanguageDetectionException;

import java.util.List;
import java.util.HashMap;
import java.util.Map;

public class LanguageDetector {

    // Instances of the rule classes for each language
    private final EnglishRule englishRule;
    private final FrenchRule frenchRule;
    private final ArabicRule arabicRule;

    public LanguageDetector() {
        this.englishRule = new EnglishRule();
        this.frenchRule = new FrenchRule();
        this.arabicRule = new ArabicRule();
    }

    /**
     * Detects the language of a list of tokens.
     * @param tokens A list of text tokens to classify.
     * @return The detected language as a Language enum.
     * @throws LanguageDetectionException if detection fails.
     */
    public Language detectLanguage(List<String> tokens, boolean verbose) throws LanguageDetectionException {
        // Map to store match counts for each language
        Map<Language, Integer> languageScores = new HashMap<>();
        languageScores.put(Language.ENGLISH, 0);
        languageScores.put(Language.FRENCH, 0);
        languageScores.put(Language.ARABIC, 0);

        // Analyze each token with the rules
        for (String token : tokens) {
            if (englishRule.matches(token)) {
                languageScores.put(Language.ENGLISH, languageScores.get(Language.ENGLISH) + 1);
            }
            if (frenchRule.matches(token)) {
                languageScores.put(Language.FRENCH, languageScores.get(Language.FRENCH) + 1);
            }
            if (arabicRule.matches(token)) {
                languageScores.put(Language.ARABIC, languageScores.get(Language.ARABIC) + 1);
            }
        }
        
        double eng_prob = tokens.isEmpty() ? 0 : (double) languageScores.get(Language.ENGLISH) / tokens.size();
        double fr_prob = tokens.isEmpty() ? 0 : (double) languageScores.get(Language.FRENCH) / tokens.size();
        double ar_prob = tokens.isEmpty() ? 0 : (double) languageScores.get(Language.ARABIC) / tokens.size();

        double confidenceThreshold = 0.2; // Adjust this value based on your needs
        if (Math.max(Math.max(eng_prob, fr_prob), ar_prob) < confidenceThreshold) {
            throw new LanguageDetectionException("Confidence too low to determine language.");
        }



        // Determine the language with the highest score
        Language detectedLanguage = null;
        int maxScore = 0;
        for (Map.Entry<Language, Integer> entry : languageScores.entrySet()) {
            if (entry.getValue() > maxScore) {
                maxScore = entry.getValue();
                detectedLanguage = entry.getKey();
            }
        }

        if (detectedLanguage == null || maxScore == 0) {
            throw new LanguageDetectionException("Unable to determine the language from the provided tokens.");
        }

        if (verbose) {
            System.out.println("RULE-BASED: Detected Language: " + detectedLanguage);
            System.out.printf("RULE-BASED: Probabilities: \tENGLISH: %.2f%%\tFRENCH: %.2f%%\tARABIC: %.2f%%%n", eng_prob * 100, fr_prob * 100, ar_prob * 100);
        }

        return detectedLanguage;
    }
}
