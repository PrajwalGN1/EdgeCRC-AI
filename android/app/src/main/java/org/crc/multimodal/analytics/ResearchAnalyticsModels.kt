package org.crc.multimodal.analytics

/**
 * Latency metrics for research inference runs.
 */
data class LatencyMetrics(
    val meanMs: Double = 0.0,
    val medianMs: Double = 0.0,
    val minMs: Double = 0.0,
    val maxMs: Double = 0.0
)

/**
 * Patch count distribution metrics.
 */
data class PatchMetrics(
    val meanPatches: Double = 0.0,
    val minPatches: Int = 0,
    val maxPatches: Int = 0,
    val distribution: Map<String, Int> = emptyMap() // e.g. "1-100", "101-500", "501-1000", ">1000"
)

/**
 * Decision-margin distribution metrics.
 *
 * Margin formula: abs(probability - 0.83)
 * Distance to fixed decision boundary tau = 0.83.
 * RESEARCH USE ONLY: Does not represent calibrated uncertainty or clinical diagnostic certainty.
 */
data class MarginMetrics(
    val minMargin: Float = 0.0f,
    val maxMargin: Float = 0.0f,
    val meanMargin: Float = 0.0f,
    val nearThresholdCount: Int = 0, // Specimens within |P - 0.83| <= 0.05
    val distribution: Map<String, Int> = emptyMap() // "[0.00 - 0.05]", "(0.05 - 0.15]", "(0.15 - 0.30]", "> 0.30"
)

/**
 * Batch-level summary statistics.
 */
data class BatchCohortSummary(
    val totalBatches: Int = 0,
    val completedBatches: Int = 0,
    val partiallyCompletedBatches: Int = 0,
    val failedBatches: Int = 0,
    val cancelledBatches: Int = 0,
    val totalBatchSpecimens: Int = 0,
    val successfulBatchSpecimens: Int = 0
)

/**
 * Comprehensive summary of research analytics across single-specimen cases and batch cohorts.
 */
data class ResearchAnalyticsSummary(
    val totalSpecimens: Int = 0,
    val totalSuccessful: Int = 0,
    val totalFailed: Int = 0,
    val totalCancelled: Int = 0,
    val totalPendingOrDraft: Int = 0,
    val msiCount: Int = 0,
    val mssCount: Int = 0,
    val msiProportion: Float = 0.0f,
    val mssProportion: Float = 0.0f,
    val probabilityBuckets: Map<String, Int> = emptyMap(),
    val marginMetrics: MarginMetrics = MarginMetrics(),
    val latencyMetrics: LatencyMetrics = LatencyMetrics(),
    val patchMetrics: PatchMetrics = PatchMetrics(),
    val batchSummary: BatchCohortSummary = BatchCohortSummary()
)

/**
 * Unified individual specimen prediction record for explorer list, search, and sorting.
 */
data class SpecimenPredictionRecord(
    val specimenId: String,
    val origin: String, // "SINGLE_CASE" or "BATCH"
    val batchId: String? = null,
    val caseId: Long? = null,
    val runId: Long? = null,
    val status: String, // "SUCCESS", "FAILED", "CANCELLED", "PENDING"
    val probability: Float? = null,
    val predictedClass: Int? = null, // 0 = MSS, 1 = MSI
    val predictedLabel: String? = null, // "MSS" or "MSI"
    val threshold: Float = 0.83f,
    val signedMargin: Float? = null, // P - 0.83
    val absMargin: Float? = null, // abs(P - 0.83)
    val patchCount: Int = 0,
    val latencyMs: Double = 0.0,
    val executionTimestamp: Long = 0L,
    val errorMessage: String? = null
)

/**
 * Filter and sort options for the specimen explorer.
 */
enum class AnalyticsFilterStatus {
    ALL,
    SUCCESS_ONLY,
    MSI_ONLY,
    MSS_ONLY,
    NEAR_THRESHOLD_ONLY, // |P - 0.83| <= 0.05
    FAILED_OR_CANCELLED
}

enum class AnalyticsSortOption {
    DATE_NEWEST,
    DATE_OLDEST,
    MARGIN_SMALLEST, // Closest to decision boundary
    MARGIN_LARGEST,
    PROBABILITY_HIGHEST,
    PROBABILITY_LOWEST,
    PATCH_COUNT_HIGHEST,
    LATENCY_HIGHEST
}
