package org.crc.multimodal.data

/**
 * Lifecycle status of a persisted research report.
 */
enum class ReportExportStatus {
    GENERATED,
    EXPORTED,
    FAILED,
    MISSING_FILE;

    companion object {
        fun fromString(value: String): ReportExportStatus {
            return try {
                valueOf(value.uppercase())
            } catch (e: Exception) {
                GENERATED
            }
        }
    }
}

/**
 * Persistent domain record tracking the lifecycle, provenance, cryptographic digest,
 * and storage locations of generated research reports.
 */
data class PersistentReportRecord(
    val id: Long = 0L,
    val reportId: String,
    val caseId: Long,
    val inferenceRunId: Long? = null,
    val templateVersion: String = "1.0.0",
    val createdAt: Long = System.currentTimeMillis(),
    val exportStatus: ReportExportStatus = ReportExportStatus.GENERATED,
    val fileUri: String? = null,
    val filePath: String? = null,
    val modelVersion: String? = "fusion_model_fp32.onnx",
    val sha256Hash: String? = null,
    val errorMessage: String? = null
)
