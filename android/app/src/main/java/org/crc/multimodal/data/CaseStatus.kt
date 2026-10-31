package org.crc.multimodal.data

/**
 * Standard research case statuses for the pathology workstation workflow.
 */
enum class CaseStatus(val displayName: String) {
    DRAFT("Draft"),
    READY_FOR_ANALYSIS("Ready for Analysis"),
    ANALYZED("Analyzed"),
    UNDER_REVIEW("Under Review"),
    REVIEWED("Reviewed");

    companion object {
        fun fromString(value: String): CaseStatus {
            return entries.firstOrNull { it.name.equals(value, ignoreCase = true) } ?: DRAFT
        }
    }
}
