package org.crc.multimodal.analytics

import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import kotlin.math.abs

/**
 * Pure, deterministic computation engine for research analytics and decision-margin aggregations.
 *
 * Implements:
 * 1. Exact SQLite reconciliation across single-specimen cases and batch runs.
 * 2. Decision-margin calculations: margin = abs(P_MSI - 0.83).
 * 3. Division-by-zero resilience for empty and incomplete datasets.
 * 4. Specimen-level filtering and multi-criteria sorting.
 */
object ResearchAnalyticsEngine {

    const val OPERATING_THRESHOLD = 0.83f
    const val NEAR_THRESHOLD_TOLERANCE = 0.05f

    /**
     * Converts single-case inference runs and batch specimen items into a unified record list.
     */
    fun buildUnifiedRecords(
        cases: List<ResearchCase>,
        runs: List<InferenceRun>,
        batchRuns: List<BatchRun> = emptyList()
    ): List<SpecimenPredictionRecord> {
        val records = mutableListOf<SpecimenPredictionRecord>()

        // 1. Single-specimen cases with their latest/persisted runs
        val caseMap = cases.associateBy { it.id }
        for (run in runs) {
            val cCase = caseMap[run.caseId]
            val specimenId = cCase?.caseIdentifier ?: "CASE-${run.caseId}"
            val p = run.probability
            val absM = abs(p - OPERATING_THRESHOLD)
            val signedM = p - OPERATING_THRESHOLD

            records.add(
                SpecimenPredictionRecord(
                    specimenId = specimenId,
                    origin = "SINGLE_CASE",
                    batchId = null,
                    caseId = run.caseId,
                    runId = run.id,
                    status = run.status,
                    probability = p,
                    predictedClass = run.predictedClass,
                    predictedLabel = run.predictedLabel,
                    threshold = OPERATING_THRESHOLD,
                    signedMargin = signedM,
                    absMargin = absM,
                    patchCount = run.numPatches,
                    latencyMs = run.latencyMs,
                    executionTimestamp = run.executionTimestamp,
                    errorMessage = if (run.status != "SUCCESS") "Inference failure" else null
                )
            )
        }

        // Add any cases that are in DRAFT or READY_FOR_ANALYSIS without runs
        val casesWithRuns = runs.map { it.caseId }.toSet()
        for (cCase in cases) {
            if (cCase.id !in casesWithRuns) {
                records.add(
                    SpecimenPredictionRecord(
                        specimenId = cCase.caseIdentifier,
                        origin = "SINGLE_CASE",
                        batchId = null,
                        caseId = cCase.id,
                        runId = null,
                        status = if (cCase.status == CaseStatus.READY_FOR_ANALYSIS) "PENDING" else "DRAFT",
                        probability = null,
                        predictedClass = null,
                        predictedLabel = null,
                        threshold = OPERATING_THRESHOLD,
                        signedMargin = null,
                        absMargin = null,
                        patchCount = cCase.patchCount,
                        latencyMs = 0.0,
                        executionTimestamp = cCase.createdAt,
                        errorMessage = null
                    )
                )
            }
        }

        // 2. Batch specimen items
        for (batch in batchRuns) {
            for (item in batch.items) {
                val statusStr = when (item.status) {
                    BatchSpecimenStatus.COMPLETED -> "SUCCESS"
                    BatchSpecimenStatus.FAILED, BatchSpecimenStatus.REJECTED -> "FAILED"
                    BatchSpecimenStatus.CANCELLED -> "CANCELLED"
                    else -> "PENDING"
                }

                val p = item.probability
                val absM = if (p != null) abs(p - OPERATING_THRESHOLD) else null
                val signedM = if (p != null) p - OPERATING_THRESHOLD else null

                records.add(
                    SpecimenPredictionRecord(
                        specimenId = item.specimenId,
                        origin = "BATCH",
                        batchId = item.batchId,
                        caseId = null,
                        runId = null,
                        status = statusStr,
                        probability = p,
                        predictedClass = item.predictedClass,
                        predictedLabel = item.predictedLabel,
                        threshold = OPERATING_THRESHOLD,
                        signedMargin = signedM,
                        absMargin = absM,
                        patchCount = item.patchCount,
                        latencyMs = item.inferenceLatencyMs.toDouble(),
                        executionTimestamp = item.completedTimestamp ?: batch.createdAt,
                        errorMessage = item.validationError ?: item.errorDetails
                    )
                )
            }
        }

        return records
    }

    /**
     * Computes the aggregated summary statistics over all unified records and batch runs.
     */
    fun computeSummary(
        records: List<SpecimenPredictionRecord>,
        batchRuns: List<BatchRun> = emptyList()
    ): ResearchAnalyticsSummary {
        val totalSpecimens = records.size
        val successful = records.filter { it.status == "SUCCESS" && it.probability != null }
        val failed = records.filter { it.status == "FAILED" }
        val cancelled = records.filter { it.status == "CANCELLED" }
        val pendingOrDraft = records.filter { it.status in listOf("PENDING", "DRAFT") }

        val totalSuccessful = successful.size
        val totalFailed = failed.size
        val totalCancelled = cancelled.size
        val totalPending = pendingOrDraft.size

        val msiCount = successful.count { it.predictedClass == 1 }
        val mssCount = successful.count { it.predictedClass == 0 }

        val msiProportion = if (totalSuccessful > 0) msiCount.toFloat() / totalSuccessful else 0.0f
        val mssProportion = if (totalSuccessful > 0) mssCount.toFloat() / totalSuccessful else 0.0f

        // Probability Buckets
        val pBuckets = linkedMapOf(
            "[0.00 - 0.20)" to 0,
            "[0.20 - 0.40)" to 0,
            "[0.40 - 0.60)" to 0,
            "[0.60 - 0.80)" to 0,
            "[0.80 - 0.83)" to 0,
            "[0.83 - 1.00]" to 0
        )
        for (rec in successful) {
            val p = rec.probability ?: continue
            when {
                p < 0.20f -> pBuckets["[0.00 - 0.20)"] = (pBuckets["[0.00 - 0.20)"] ?: 0) + 1
                p < 0.40f -> pBuckets["[0.20 - 0.40)"] = (pBuckets["[0.20 - 0.40)"] ?: 0) + 1
                p < 0.60f -> pBuckets["[0.40 - 0.60)"] = (pBuckets["[0.40 - 0.60)"] ?: 0) + 1
                p < 0.80f -> pBuckets["[0.60 - 0.80)"] = (pBuckets["[0.60 - 0.80)"] ?: 0) + 1
                p < 0.83f -> pBuckets["[0.80 - 0.83)"] = (pBuckets["[0.80 - 0.83)"] ?: 0) + 1
                else -> pBuckets["[0.83 - 1.00]"] = (pBuckets["[0.83 - 1.00]"] ?: 0) + 1
            }
        }

        // Decision Margin Metrics
        val margins = successful.mapNotNull { it.absMargin }
        val marginMetrics = if (margins.isNotEmpty()) {
            val minM = margins.minOrNull() ?: 0.0f
            val maxM = margins.maxOrNull() ?: 0.0f
            val meanM = (margins.sum() / margins.size)
            val nearCount = margins.count { it <= NEAR_THRESHOLD_TOLERANCE }

            val mDist = linkedMapOf(
                "[0.00 - 0.05]" to 0,
                "(0.05 - 0.15]" to 0,
                "(0.15 - 0.30]" to 0,
                "> 0.30" to 0
            )
            for (m in margins) {
                when {
                    m <= 0.05f -> mDist["[0.00 - 0.05]"] = (mDist["[0.00 - 0.05]"] ?: 0) + 1
                    m <= 0.15f -> mDist["(0.05 - 0.15]"] = (mDist["(0.05 - 0.15]"] ?: 0) + 1
                    m <= 0.30f -> mDist["(0.15 - 0.30]"] = (mDist["(0.15 - 0.30]"] ?: 0) + 1
                    else -> mDist["> 0.30"] = (mDist["> 0.30"] ?: 0) + 1
                }
            }
            MarginMetrics(minM, maxM, meanM, nearCount, mDist)
        } else {
            MarginMetrics()
        }

        // Latency Metrics
        val latencies = successful.map { it.latencyMs }.filter { it > 0.0 }.sorted()
        val latencyMetrics = if (latencies.isNotEmpty()) {
            val meanLat = latencies.sum() / latencies.size
            val medianLat = if (latencies.size % 2 == 1) {
                latencies[latencies.size / 2]
            } else {
                (latencies[latencies.size / 2 - 1] + latencies[latencies.size / 2]) / 2.0
            }
            val minLat = latencies.first()
            val maxLat = latencies.last()
            LatencyMetrics(meanLat, medianLat, minLat, maxLat)
        } else {
            LatencyMetrics()
        }

        // Patch Metrics
        val patches = successful.map { it.patchCount }.filter { it > 0 }
        val patchMetrics = if (patches.isNotEmpty()) {
            val meanP = patches.sum().toDouble() / patches.size
            val minP = patches.minOrNull() ?: 0
            val maxP = patches.maxOrNull() ?: 0

            val pDist = linkedMapOf(
                "1 - 100" to 0,
                "101 - 500" to 0,
                "501 - 1000" to 0,
                "> 1000" to 0
            )
            for (p in patches) {
                when {
                    p <= 100 -> pDist["1 - 100"] = (pDist["1 - 100"] ?: 0) + 1
                    p <= 500 -> pDist["101 - 500"] = (pDist["101 - 500"] ?: 0) + 1
                    p <= 1000 -> pDist["501 - 1000"] = (pDist["501 - 1000"] ?: 0) + 1
                    else -> pDist["> 1000"] = (pDist["> 1000"] ?: 0) + 1
                }
            }
            PatchMetrics(meanP, minP, maxP, pDist)
        } else {
            PatchMetrics()
        }

        // Batch Cohort Summary
        val totalBatches = batchRuns.size
        val completedBatches = batchRuns.count { it.status == BatchStatus.COMPLETED }
        val partialBatches = batchRuns.count { it.status == BatchStatus.PARTIALLY_COMPLETED }
        val failedBatches = batchRuns.count { it.status == BatchStatus.FAILED }
        val cancelledBatches = batchRuns.count { it.status == BatchStatus.CANCELLED }
        val totalBatchItems = batchRuns.sumOf { it.items.size }
        val successfulBatchItems = batchRuns.sumOf { b -> b.items.count { it.status == BatchSpecimenStatus.COMPLETED && it.probability != null } }

        val batchCohortSummary = BatchCohortSummary(
            totalBatches = totalBatches,
            completedBatches = completedBatches,
            partiallyCompletedBatches = partialBatches,
            failedBatches = failedBatches,
            cancelledBatches = cancelledBatches,
            totalBatchSpecimens = totalBatchItems,
            successfulBatchSpecimens = successfulBatchItems
        )

        return ResearchAnalyticsSummary(
            totalSpecimens = totalSpecimens,
            totalSuccessful = totalSuccessful,
            totalFailed = totalFailed,
            totalCancelled = totalCancelled,
            totalPendingOrDraft = totalPending,
            msiCount = msiCount,
            mssCount = mssCount,
            msiProportion = msiProportion,
            mssProportion = mssProportion,
            probabilityBuckets = pBuckets,
            marginMetrics = marginMetrics,
            latencyMetrics = latencyMetrics,
            patchMetrics = patchMetrics,
            batchSummary = batchCohortSummary
        )
    }

    /**
     * Filters and sorts specimen prediction records according to user criteria.
     */
    fun filterAndSort(
        records: List<SpecimenPredictionRecord>,
        searchQuery: String = "",
        filterStatus: AnalyticsFilterStatus = AnalyticsFilterStatus.ALL,
        sortOption: AnalyticsSortOption = AnalyticsSortOption.DATE_NEWEST
    ): List<SpecimenPredictionRecord> {
        val query = searchQuery.trim().lowercase()

        // 1. Search Query Filter
        var filtered = if (query.isEmpty()) {
            records
        } else {
            records.filter {
                it.specimenId.lowercase().contains(query) || (it.batchId?.lowercase()?.contains(query) == true)
            }
        }

        // 2. Status Filter
        filtered = when (filterStatus) {
            AnalyticsFilterStatus.ALL -> filtered
            AnalyticsFilterStatus.SUCCESS_ONLY -> filtered.filter { it.status == "SUCCESS" }
            AnalyticsFilterStatus.MSI_ONLY -> filtered.filter { it.status == "SUCCESS" && it.predictedClass == 1 }
            AnalyticsFilterStatus.MSS_ONLY -> filtered.filter { it.status == "SUCCESS" && it.predictedClass == 0 }
            AnalyticsFilterStatus.NEAR_THRESHOLD_ONLY -> filtered.filter {
                it.status == "SUCCESS" && (it.absMargin ?: Float.MAX_VALUE) <= NEAR_THRESHOLD_TOLERANCE
            }
            AnalyticsFilterStatus.FAILED_OR_CANCELLED -> filtered.filter { it.status in listOf("FAILED", "CANCELLED") }
        }

        // 3. Sorting
        return when (sortOption) {
            AnalyticsSortOption.DATE_NEWEST -> filtered.sortedByDescending { it.executionTimestamp }
            AnalyticsSortOption.DATE_OLDEST -> filtered.sortedBy { it.executionTimestamp }
            AnalyticsSortOption.MARGIN_SMALLEST -> filtered.sortedBy { it.absMargin ?: Float.MAX_VALUE }
            AnalyticsSortOption.MARGIN_LARGEST -> filtered.sortedByDescending { it.absMargin ?: -1.0f }
            AnalyticsSortOption.PROBABILITY_HIGHEST -> filtered.sortedByDescending { it.probability ?: -1.0f }
            AnalyticsSortOption.PROBABILITY_LOWEST -> filtered.sortedBy { it.probability ?: Float.MAX_VALUE }
            AnalyticsSortOption.PATCH_COUNT_HIGHEST -> filtered.sortedByDescending { it.patchCount }
            AnalyticsSortOption.LATENCY_HIGHEST -> filtered.sortedByDescending { it.latencyMs }
        }
    }
}
