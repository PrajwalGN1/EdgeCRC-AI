package org.crc.multimodal.data

import org.crc.multimodal.inference.InferenceProtocol

/**
 * Domain entity representing a research case in the pathology workstation.
 */
data class ResearchCase(
    val id: Long = 0L,
    val caseIdentifier: String,
    val specimenDescription: String = "",
    val researchContext: String = "",
    val status: CaseStatus = CaseStatus.DRAFT,
    val patchCount: Int = 10,
    val protocol: InferenceProtocol? = InferenceProtocol.PROTOCOL_1_CAPPED,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis()
) {
    init {
        require(caseIdentifier.isNotBlank()) { "Case identifier cannot be blank." }
        require(patchCount > 0) { "Patch count must be strictly positive." }
        require(researchContext.length <= 50000) { "Research context cannot exceed 50,000 characters." }
    }
}
