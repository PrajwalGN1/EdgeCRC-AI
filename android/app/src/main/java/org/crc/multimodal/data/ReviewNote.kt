package org.crc.multimodal.data

/**
 * Domain entity representing a pathologist or investigator review observation on a case.
 */
data class ReviewNote(
    val id: Long = 0L,
    val caseId: Long,
    val category: ReviewNoteCategory = ReviewNoteCategory.GENERAL_RESEARCH_NOTE,
    val authorLabel: String = "Research Investigator",
    val content: String,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis()
) {
    init {
        require(content.isNotBlank()) { "Review note content cannot be blank." }
    }
}
