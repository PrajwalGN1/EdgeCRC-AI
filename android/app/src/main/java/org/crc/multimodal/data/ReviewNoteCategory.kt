package org.crc.multimodal.data

/**
 * Categorical classifications for pathology review observations and notes.
 */
enum class ReviewNoteCategory(val displayName: String) {
    HISTOLOGICAL_OBSERVATION("Histological Observation"),
    MODEL_UNCERTAINTY("Model Uncertainty"),
    INPUT_QUALITY_ISSUE("Input Quality Issue"),
    TECHNICAL_ISSUE("Technical Issue"),
    GENERAL_RESEARCH_NOTE("General Research Note");

    companion object {
        fun fromString(value: String): ReviewNoteCategory {
            return entries.firstOrNull { it.name.equals(value, ignoreCase = true) }
                ?: GENERAL_RESEARCH_NOTE
        }
    }
}
