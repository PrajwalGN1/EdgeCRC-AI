

package org.crc.multimodal.batch

import org.crc.multimodal.inference.InputProvenance

/**
 * Status lifecycle for a batch inference session.
 */
enum class BatchStatus {
    DRAFT,
    VALIDATING,
    READY,
    PROCESSING,
    COMPLETED,
    PARTIALLY_COMPLETED,
    CANCELLED,
    FAILED
}

/**
 * Status lifecycle for an individual specimen within a batch.
 */
enum class BatchSpecimenStatus {
    PENDING,
    VALIDATED,
    REJECTED,
    INFERRING,
    COMPLETED,
    FAILED,
    CANCELLED
}

/**
 * Detailed traceability record for a single specimen in a batch.
 */
data class BatchSpecimenItem(
    val id: Long = 0L,
    val batchId: String,
    val specimenId: String,
    val sourceUri: String = "",
    val fileName: String = "",
    val fileSizeBytes: Long = 0L,
    val bundleSha256: String = "",
    val visionSha256: String = "",
    val textSha256: String = "",
    val schemaVersion: Int = 2,
    val patchCount: Int = 0,
    val status: BatchSpecimenStatus = BatchSpecimenStatus.PENDING,
    val validationError: String? = null,
    val logit: Float? = null,
    val probability: Float? = null,
    val predictedClass: Int? = null, // 0 for MSS, 1 for MSI
    val predictedLabel: String? = null, // "MSS", "MSI"
    val threshold: Float = 0.83f,
    val parseTimeMs: Long = 0L,
    val validationTimeMs: Long = 0L,
    val inferenceLatencyMs: Long = 0L,
    val provenance: InputProvenance = InputProvenance.AUTHENTIC_IMPORTED,
    val errorDetails: String? = null,
    val completedTimestamp: Long? = null,
    val protocol: String = "PROTOCOL_1_CAPPED",
    val protocolReason: String? = null,
    val isArchived: Boolean = false
)

/**
 * Summary aggregate statistics computed strictly over successfully processed specimens.
 */
data class BatchStatistics(
    val totalSelected: Int,
    val totalAccepted: Int,
    val totalRejected: Int,
    val totalProcessed: Int,
    val totalSuccessful: Int,
    val totalFailed: Int,
    val totalCancelled: Int,
    val msiCount: Int,
    val mssCount: Int,
    val msiProportion: Float, // msiCount / totalSuccessful
    val mssProportion: Float, // mssCount / totalSuccessful
    val meanLatencyMs: Double,
    val medianLatencyMs: Double,
    val minLatencyMs: Long,
    val maxLatencyMs: Long,
    val totalDurationMs: Long,
    val patchCountMin: Int,
    val patchCountMax: Int,
    val patchCountMean: Double,
    val exclusionCount: Int,
    val exclusionReasons: Map<String, Int>,
    val totalArchived: Int = 0
) {
    companion object {
        fun compute(items: List<BatchSpecimenItem>, totalDurationMs: Long = 0L): BatchStatistics {
            val totalSelected = items.size
            val archivedItems = items.filter { it.isArchived }
            val totalArchived = archivedItems.size
            val activeItems = items.filter { !it.isArchived }

            val accepted = activeItems.filter { it.status != BatchSpecimenStatus.REJECTED }
            val rejected = activeItems.filter { it.status == BatchSpecimenStatus.REJECTED }
            val successful = activeItems.filter { it.status == BatchSpecimenStatus.COMPLETED && it.probability != null }
            val failed = activeItems.filter { it.status == BatchSpecimenStatus.FAILED }
            val cancelled = activeItems.filter { it.status == BatchSpecimenStatus.CANCELLED }

            val msiCount = successful.count { it.predictedClass == 1 }
            val mssCount = successful.count { it.predictedClass == 0 }
            val successCount = successful.size

            val msiProp = if (successCount > 0) msiCount.toFloat() / successCount else 0.0f
            val mssProp = if (successCount > 0) mssCount.toFloat() / successCount else 0.0f

            val latencies = successful.map { it.inferenceLatencyMs }.sorted()
            val meanLat = if (latencies.isNotEmpty()) latencies.average() else 0.0
            val medianLat = if (latencies.isNotEmpty()) {
                val mid = latencies.size / 2
                if (latencies.size % 2 == 1) latencies[mid].toDouble()
                else (latencies[mid - 1] + latencies[mid]) / 2.0
            } else 0.0
            val minLat = latencies.firstOrNull() ?: 0L
            val maxLat = latencies.lastOrNull() ?: 0L

            val patchCounts = successful.map { it.patchCount }.sorted()
            val pMin = patchCounts.firstOrNull() ?: 0
            val pMax = patchCounts.lastOrNull() ?: 0
            val pMean = if (patchCounts.isNotEmpty()) patchCounts.average() else 0.0

            val exclusions = mutableMapOf<String, Int>()
            rejected.forEach {
                val reason = it.validationError ?: "Validation rejected"
                exclusions[reason] = (exclusions[reason] ?: 0) + 1
            }
            failed.forEach {
                val reason = it.errorDetails ?: "Inference failure"
                exclusions[reason] = (exclusions[reason] ?: 0) + 1
            }
            cancelled.forEach {
                exclusions["Cancelled by user"] = (exclusions["Cancelled by user"] ?: 0) + 1
            }

            return BatchStatistics(
                totalSelected = totalSelected,
                totalAccepted = accepted.size,
                totalRejected = rejected.size,
                totalProcessed = successful.size + failed.size,
                totalSuccessful = successCount,
                totalFailed = failed.size,
                totalCancelled = cancelled.size,
                msiCount = msiCount,
                mssCount = mssCount,
                msiProportion = msiProp,
                mssProportion = mssProp,
                meanLatencyMs = meanLat,
                medianLatencyMs = medianLat,
                minLatencyMs = minLat,
                maxLatencyMs = maxLat,
                totalDurationMs = totalDurationMs,
                patchCountMin = pMin,
                patchCountMax = pMax,
                patchCountMean = pMean,
                exclusionCount = rejected.size + failed.size + cancelled.size,
                exclusionReasons = exclusions,
                totalArchived = totalArchived
            )
        }
    }
}

/**
 * Top-level entity representing an entire batch run.
 */
data class BatchRun(
    val batchId: String,
    val title: String = "CRC Multimodal Batch Evaluation",
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
    val status: BatchStatus = BatchStatus.DRAFT,
    val modelSha256: String = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73",
    val appVersion: String = "1.0.0-phase6",
    val executionProvider: String = "CPUExecutionProvider",
    val decisionThreshold: Float = 0.83f,
    val totalDurationMs: Long = 0L,
    val items: List<BatchSpecimenItem> = emptyList()
) {
    val statistics: BatchStatistics
        get() = BatchStatistics.compute(items, totalDurationMs)
}
