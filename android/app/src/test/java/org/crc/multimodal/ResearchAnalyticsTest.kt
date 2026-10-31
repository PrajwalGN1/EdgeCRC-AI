package org.crc.multimodal

import org.crc.multimodal.analytics.AnalyticsFilterStatus
import org.crc.multimodal.analytics.AnalyticsSortOption
import org.crc.multimodal.analytics.ResearchAnalyticsEngine
import org.crc.multimodal.analytics.SpecimenPredictionRecord
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.InferenceProtocol
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.abs

/**
 * Unit test suite for WP7.3: Research Analytics and Decision-Margin Dashboard.
 *
 * Verifies:
 * 1. Exact mathematical reconciliation with SQLite records.
 * 2. Decision-margin calculations: margin = abs(P_MSI - 0.8300).
 * 3. Near-threshold boundary identification (margin <= 0.05).
 * 4. Division-by-zero safety on empty / incomplete datasets.
 * 5. Probability bucket and patch distribution sum integrity.
 * 6. Specimen explorer search, multi-status filtering, and sorting.
 */
class ResearchAnalyticsTest {

    private lateinit var repository: InMemoryWorkstationRepository
    private lateinit var mockBatchRepository: InMemoryBatchRepository

    @Before
    fun setUp() {
        repository = InMemoryWorkstationRepository()
        mockBatchRepository = InMemoryBatchRepository()
    }

    @Test
    fun testEmptyDatasetResilience() {
        val summary = repository.getResearchAnalyticsSummary(mockBatchRepository)

        // Resilient zero values without arithmetic exceptions
        assertEquals(0, summary.totalSpecimens)
        assertEquals(0, summary.totalSuccessful)
        assertEquals(0, summary.totalFailed)
        assertEquals(0, summary.totalCancelled)
        assertEquals(0, summary.msiCount)
        assertEquals(0, summary.mssCount)
        assertEquals(0.0f, summary.msiProportion, 1e-6f)
        assertEquals(0.0f, summary.mssProportion, 1e-6f)
        assertEquals(0.0, summary.latencyMetrics.meanMs, 1e-6)
        assertEquals(0.0, summary.latencyMetrics.medianMs, 1e-6)
        assertEquals(0.0f, summary.marginMetrics.meanMargin, 1e-6f)
        assertEquals(0.0f, summary.marginMetrics.minMargin, 1e-6f)
        assertEquals(0, summary.marginMetrics.nearThresholdCount)
    }

    @Test
    fun testDecisionMarginFormulaAndBoundaries() {
        val threshold = 0.83f

        // Case 1: Close boundary MSI (e.g. Case 36 TCGA-CM-4752: P = 0.84074)
        val p1 = 0.84074f
        val absM1 = abs(p1 - threshold)
        val signedM1 = p1 - threshold
        assertEquals(0.01074f, absM1, 1e-5f)
        assertEquals(0.01074f, signedM1, 1e-5f)
        assertTrue("Must be classified as near-threshold", absM1 <= ResearchAnalyticsEngine.NEAR_THRESHOLD_TOLERANCE)

        // Case 2: Distant MSS (e.g. TCGA-CM-6675: P = 0.16225)
        val p2 = 0.16225f
        val absM2 = abs(p2 - threshold)
        val signedM2 = p2 - threshold
        assertEquals(0.66775f, absM2, 1e-5f)
        assertEquals(-0.66775f, signedM2, 1e-5f)
        assertTrue("Must NOT be near-threshold", absM2 > ResearchAnalyticsEngine.NEAR_THRESHOLD_TOLERANCE)

        // Case 3: Exact boundary (P = 0.8300)
        val p3 = 0.83000f
        val absM3 = abs(p3 - threshold)
        assertEquals(0.0f, absM3, 1e-6f)
    }

    @Test
    fun testExactSQLiteReconciliation() {
        // Seed 2 single cases
        val c1Id = repository.insertCase(
            ResearchCase(caseIdentifier = "TCGA-AA-0001", status = CaseStatus.READY_FOR_ANALYSIS, patchCount = 100)
        )
        val run1 = InferenceRun(
            caseId = c1Id,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.95f,
            logit = 3.0f,
            threshold = 0.83f,
            margin = 0.12f,
            latencyMs = 40.0,
            numPatches = 100,
            diagnosticsSummary = "Test 1",
            executionTimestamp = 1000L,
            status = "SUCCESS"
        )
        repository.recordInferenceResult(c1Id, run1, CaseStatus.ANALYZED)

        val c2Id = repository.insertCase(
            ResearchCase(caseIdentifier = "TCGA-AA-0002", status = CaseStatus.READY_FOR_ANALYSIS, patchCount = 200)
        )
        val run2 = InferenceRun(
            caseId = c2Id,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.81f, // near boundary MSS (|0.81 - 0.83| = 0.02)
            logit = 1.4f,
            threshold = 0.83f,
            margin = -0.02f,
            latencyMs = 60.0,
            numPatches = 200,
            diagnosticsSummary = "Test 2",
            executionTimestamp = 2000L,
            status = "SUCCESS"
        )
        repository.recordInferenceResult(c2Id, run2, CaseStatus.ANALYZED)

        // Seed 1 batch run with 2 specimens (1 success, 1 failed)
        val batchItem1 = BatchSpecimenItem(
            batchId = "BATCH-01",
            specimenId = "TCGA-BATCH-S1",
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.10f,
            predictedClass = 0,
            predictedLabel = "MSS",
            patchCount = 150,
            inferenceLatencyMs = 50L
        )
        val batchItem2 = BatchSpecimenItem(
            batchId = "BATCH-01",
            specimenId = "TCGA-BATCH-S2",
            status = BatchSpecimenStatus.FAILED,
            validationError = "Corrupt tensor"
        )
        val batch = BatchRun(
            batchId = "BATCH-01",
            title = "Test Batch Cohort",
            createdAt = 3000L,
            updatedAt = 3500L,
            status = BatchStatus.PARTIALLY_COMPLETED,
            items = listOf(batchItem1, batchItem2)
        )
        mockBatchRepository.insertBatchRun(batch)

        // Compute summary across repositories
        val summary = repository.getResearchAnalyticsSummary(mockBatchRepository)

        // Verifications:
        // Total specimens = 2 single cases + 2 batch specimens = 4
        assertEquals(4, summary.totalSpecimens)
        // Successful predictions = 2 single cases + 1 batch item = 3
        assertEquals(3, summary.totalSuccessful)
        // Failed specimens = 1
        assertEquals(1, summary.totalFailed)

        // MSI vs MSS:
        // MSI: TCGA-AA-0001 (P=0.95) -> 1
        // MSS: TCGA-AA-0002 (P=0.81), TCGA-BATCH-S1 (P=0.10) -> 2
        assertEquals(1, summary.msiCount)
        assertEquals(2, summary.mssCount)
        assertEquals(1.0f / 3.0f, summary.msiProportion, 1e-4f)
        assertEquals(2.0f / 3.0f, summary.mssProportion, 1e-4f)

        // Decision Margins:
        // M1: |0.95 - 0.83| = 0.12
        // M2: |0.81 - 0.83| = 0.02 (near boundary <= 0.05)
        // M3: |0.10 - 0.83| = 0.73
        // Min margin = 0.02
        assertEquals(0.02f, summary.marginMetrics.minMargin, 1e-4f)
        assertEquals(0.73f, summary.marginMetrics.maxMargin, 1e-4f)
        // Near boundary count = 1 (TCGA-AA-0002)
        assertEquals(1, summary.marginMetrics.nearThresholdCount)

        // Latencies: 40.0, 50.0, 60.0 -> Median = 50.0, Mean = 50.0
        assertEquals(50.0, summary.latencyMetrics.medianMs, 1e-4)
        assertEquals(50.0, summary.latencyMetrics.meanMs, 1e-4)
        assertEquals(40.0, summary.latencyMetrics.minMs, 1e-4)
        assertEquals(60.0, summary.latencyMetrics.maxMs, 1e-4)
    }

    @Test
    fun testProbabilityBucketsDistribution() {
        val cases = listOf(
            createPredictionRecord("S1", 0.10f), // [0.00 - 0.20)
            createPredictionRecord("S2", 0.35f), // [0.20 - 0.40)
            createPredictionRecord("S3", 0.55f), // [0.40 - 0.60)
            createPredictionRecord("S4", 0.75f), // [0.60 - 0.80)
            createPredictionRecord("S5", 0.81f), // [0.80 - 0.83)
            createPredictionRecord("S6", 0.829f), // [0.80 - 0.83)
            createPredictionRecord("S7", 0.830f), // [0.83 - 1.00]
            createPredictionRecord("S8", 0.99f)  // [0.83 - 1.00]
        )

        val summary = ResearchAnalyticsEngine.computeSummary(cases)

        assertEquals(8, summary.totalSuccessful)
        val pBuckets = summary.probabilityBuckets
        assertEquals(1, pBuckets["[0.00 - 0.20)"])
        assertEquals(1, pBuckets["[0.20 - 0.40)"])
        assertEquals(1, pBuckets["[0.40 - 0.60)"])
        assertEquals(1, pBuckets["[0.60 - 0.80)"])
        assertEquals(2, pBuckets["[0.80 - 0.83)"]) // 0.81 and 0.829
        assertEquals(2, pBuckets["[0.83 - 1.00]"]) // 0.830 and 0.99

        // Sum of all buckets must equal total successful
        val totalBucketCount = pBuckets.values.sum()
        assertEquals(summary.totalSuccessful, totalBucketCount)
    }

    @Test
    fun testSearchAndFilterFunctionality() {
        val records = listOf(
            createPredictionRecord("TCGA-A6-3807", 0.95f, "BATCH-ALPHA"),
            createPredictionRecord("TCGA-CM-4752", 0.84f, "BATCH-BETA"), // Near boundary (|0.84 - 0.83| = 0.01)
            createPredictionRecord("TCGA-AA-3877", 0.15f, "BATCH-ALPHA"),
            SpecimenPredictionRecord(
                specimenId = "TCGA-ERR-9999",
                origin = "BATCH",
                batchId = "BATCH-BETA",
                status = "FAILED",
                errorMessage = "Corrupt payload"
            )
        )

        // 1. Search Query: "4752"
        val searched = ResearchAnalyticsEngine.filterAndSort(records, searchQuery = "4752")
        assertEquals(1, searched.size)
        assertEquals("TCGA-CM-4752", searched.first().specimenId)

        // 2. Search Query by Batch ID: "alpha"
        val batchFiltered = ResearchAnalyticsEngine.filterAndSort(records, searchQuery = "alpha")
        assertEquals(2, batchFiltered.size)

        // 3. Filter Status: MSI_ONLY
        val msiFiltered = ResearchAnalyticsEngine.filterAndSort(records, filterStatus = AnalyticsFilterStatus.MSI_ONLY)
        assertEquals(2, msiFiltered.size)
        assertTrue(msiFiltered.all { it.predictedClass == 1 })

        // 4. Filter Status: NEAR_THRESHOLD_ONLY
        val nearFiltered = ResearchAnalyticsEngine.filterAndSort(records, filterStatus = AnalyticsFilterStatus.NEAR_THRESHOLD_ONLY)
        assertEquals(1, nearFiltered.size)
        assertEquals("TCGA-CM-4752", nearFiltered.first().specimenId)

        // 5. Filter Status: FAILED_OR_CANCELLED
        val failedFiltered = ResearchAnalyticsEngine.filterAndSort(records, filterStatus = AnalyticsFilterStatus.FAILED_OR_CANCELLED)
        assertEquals(1, failedFiltered.size)
        assertEquals("TCGA-ERR-9999", failedFiltered.first().specimenId)
    }

    @Test
    fun testMultiCriteriaSorting() {
        val records = listOf(
            createPredictionRecord("S1", 0.95f), // absMargin = 0.12
            createPredictionRecord("S2", 0.84f), // absMargin = 0.01
            createPredictionRecord("S3", 0.15f)  // absMargin = 0.68
        )

        // Sort by MARGIN_SMALLEST: S2 (0.01) -> S1 (0.12) -> S3 (0.68)
        val sortedSmallest = ResearchAnalyticsEngine.filterAndSort(
            records,
            sortOption = AnalyticsSortOption.MARGIN_SMALLEST
        )
        assertEquals("S2", sortedSmallest[0].specimenId)
        assertEquals("S1", sortedSmallest[1].specimenId)
        assertEquals("S3", sortedSmallest[2].specimenId)

        // Sort by MARGIN_LARGEST: S3 (0.68) -> S1 (0.12) -> S2 (0.01)
        val sortedLargest = ResearchAnalyticsEngine.filterAndSort(
            records,
            sortOption = AnalyticsSortOption.MARGIN_LARGEST
        )
        assertEquals("S3", sortedLargest[0].specimenId)
        assertEquals("S1", sortedLargest[1].specimenId)
        assertEquals("S2", sortedLargest[2].specimenId)

        // Sort by PROBABILITY_HIGHEST: S1 (0.95) -> S2 (0.84) -> S3 (0.15)
        val sortedProb = ResearchAnalyticsEngine.filterAndSort(
            records,
            sortOption = AnalyticsSortOption.PROBABILITY_HIGHEST
        )
        assertEquals("S1", sortedProb[0].specimenId)
        assertEquals("S2", sortedProb[1].specimenId)
        assertEquals("S3", sortedProb[2].specimenId)
    }

    private fun createPredictionRecord(
        id: String,
        prob: Float,
        batchId: String? = null
    ): SpecimenPredictionRecord {
        val isMsi = prob >= 0.83f
        val absM = abs(prob - 0.83f)
        val signedM = prob - 0.83f

        return SpecimenPredictionRecord(
            specimenId = id,
            origin = if (batchId != null) "BATCH" else "SINGLE_CASE",
            batchId = batchId,
            caseId = 1L,
            runId = 1L,
            status = "SUCCESS",
            probability = prob,
            predictedClass = if (isMsi) 1 else 0,
            predictedLabel = if (isMsi) "MSI" else "MSS",
            threshold = 0.83f,
            signedMargin = signedM,
            absMargin = absM,
            patchCount = 385,
            latencyMs = 45.0,
            executionTimestamp = System.currentTimeMillis(),
            errorMessage = null
        )
    }
}
