package com.ats.langdetect.detector;

public enum Language {
    ENGLISH,
    FRENCH,
    ARABIC;

    /// Provides a user-friendly string representation of the language.
    /// @return The name of the language in title case.
    @Override
    public String toString() {
        return switch (this) {
            case ENGLISH -> "ENGLISH";
            case FRENCH -> "FRENCH";
            case ARABIC -> "ARABIC";
        };
    }

    /**
     * Converts a Lingua Language to our Language enum
     * @param linguaLang The Lingua Language to convert
     * @return The corresponding Language enum value, or null if no match is found
     */
    public static Language fromLingua(com.github.pemistahl.lingua.api.Language linguaLang) {
        if (linguaLang == null) return null;
        
        return switch (linguaLang) {
            case ENGLISH -> ENGLISH;
            case FRENCH -> FRENCH;
            case ARABIC -> ARABIC;
            default -> null;
        };
    }
}
