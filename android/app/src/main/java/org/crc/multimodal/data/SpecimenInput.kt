package org.crc.multimodal.data

/**
 * Domain entity representing an ingested specimen input file or tensor reference.
 */
data class SpecimenInput(
    val id: Long = 0L,
    val caseId: Long,
    val inputType: String,
    val localUriOrAsset: String,
    val filename: String,
    val validationStatus: String = "VALID",
    val createdAt: Long = System.currentTimeMillis()
)
