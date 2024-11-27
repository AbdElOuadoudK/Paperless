package com.ats.langdetect.detector;

import com.github.pemistahl.lingua.api.Language;
import com.github.pemistahl.lingua.api.LanguageDetector;
import com.github.pemistahl.lingua.api.LanguageDetectorBuilder;

public class LinguaDetector {
    private final LanguageDetector detector;

    public LinguaDetector() {
        // Initialize Lingua detector with desired languages
        detector = LanguageDetectorBuilder.fromLanguages(
            Language.ENGLISH,
            Language.FRENCH,
            Language.ARABIC
        ).build();
    }

    public Language detectLanguage(String text) {
        return detector.detectLanguageOf(text);
    }

    public double getConfidence(String text, Language language) {
        return detector.computeLanguageConfidenceValues(text)
                      .get(language);
    }
} 