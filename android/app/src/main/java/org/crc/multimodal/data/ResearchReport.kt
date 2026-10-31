package org.crc.multimodal.data

/**
 * Domain entity representing a formal, structured research report compiled from case data and inference.
 */
data class ResearchReport(
    val id: Long = 0L,
    val caseId: Long,
    val inferenceRunId: Long? = null,
    val generatedAt: Long = System.currentTimeMillis(),
    val reportContent: String,
    val findingsSummary: String
)
