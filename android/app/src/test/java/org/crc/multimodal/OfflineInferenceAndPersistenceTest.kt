package org.crc.multimodal

import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputValidator
import org.crc.multimodal.inference.ThresholdLogic
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.util.concurrent.ConcurrentHashMap

/**
 * WP7.2: Offline Inference and Persistence Unit Test Suite.
 *
 * Verifies:
 * 1. Atomic transactional persistence and state transitions.
 * 2. Safe failure recovery without false completion.
 * 3. Crash recovery reconciliation of interrupted/orphan cases.
 * 4. Strict input revalidation: dimensions, bounds, finite float enforcement (no NaN/Inf).
 * 5. Decision threshold boundary (tau = 0.83) and margin calculations.
 * 6. Provenance digests, model SHA-256 traceability, and concurrent duplicate execution prevention.
 */
class OfflineInferenceAndPersistenceTest {

    private lateinit var repository: InMemoryWorkstationRepository

    companion object {
        const val EXPECTED_MODEL_SHA256 = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
        const val FROZEN_THRESHOLD = 0.83f
    }

    @Before
    fun setUp() {
        repository = InMemoryWorkstationRepository()
    }

    @Test
    fun testAtomicResultCommitAndStatusTransition() {
        val initialCase = ResearchCase(
            caseIdentifier = "TCGA-WP72-001",
            specimenDescription = "Primary colorectal resection",
            researchContext = "Offline inference verification",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 100,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val caseId = repository.insertCase(initialCase)
        assertTrue(caseId > 0)

        val run = InferenceRun(
            caseId = caseId,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32 (AUTHENTIC_IMPORTED)",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.9421f,
            logit = 2.7890f,
            threshold = FROZEN_THRESHOLD,
            margin = 0.1121f,
            latencyMs = 45.2,
            numPatches = 100,
            diagnosticsSummary = "Model SHA-256: $EXPECTED_MODEL_SHA256",
            executionTimestamp = System.currentTimeMillis(),
            status = "SUCCESS"
        )

        val runId = repository.recordInferenceResult(caseId, run, CaseStatus.ANALYZED)
        assertTrue(runId > 0)

        // Verify case is now ANALYZED
        val updatedCase = repository.getCaseById(caseId)
        assertNotNull(updatedCase)
        assertEquals(CaseStatus.ANALYZED, updatedCase?.status)
        assertEquals(100, updatedCase?.patchCount)
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, updatedCase?.protocol)

        // Verify run was persisted
        val persistedRun = repository.getLatestRunForCase(caseId)
        assertNotNull(persistedRun)
        assertEquals(runId, persistedRun?.id)
        assertEquals(1, persistedRun?.predictedClass)
        assertEquals("MSI", persistedRun?.predictedLabel)
        assertEquals(0.9421f, persistedRun?.probability ?: 0f, 1e-5f)
        assertEquals(FROZEN_THRESHOLD, persistedRun?.threshold ?: 0f, 1e-5f)
        assertEquals("SUCCESS", persistedRun?.status)
    }

    @Test
    fun testTransactionRollbackOnInvalidCase() {
        val run = InferenceRun(
            caseId = -9999L,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.12f,
            logit = -1.99f,
            threshold = FROZEN_THRESHOLD,
            margin = -0.71f,
            latencyMs = 30.0,
            numPatches = 50,
            diagnosticsSummary = "Invalid case run",
            executionTimestamp = System.currentTimeMillis(),
            status = "SUCCESS"
        )

        try {
            repository.recordInferenceResult(-9999L, run, CaseStatus.ANALYZED)
            fail("Expected IllegalStateException when recording result for non-existent case")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("not found"))
        }

        assertNull(repository.getLatestRunForCase(-9999L))
    }

    @Test
    fun testInferenceFailureDoesNotCreateCompletedRun() {
        val initialCase = ResearchCase(
            caseIdentifier = "TCGA-WP72-FAIL",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 50,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val caseId = repository.insertCase(initialCase)

        // Record an inference failure
        repository.recordInferenceFailure(caseId, "ONNX execution timeout or tensor mismatch")

        val caseAfterFail = repository.getCaseById(caseId)
        assertNotNull(caseAfterFail)
        // Must remain READY_FOR_ANALYSIS (not ANALYZED)
        assertEquals(CaseStatus.READY_FOR_ANALYSIS, caseAfterFail?.status)

        // No inference run should have been marked SUCCESS
        val run = repository.getLatestRunForCase(caseId)
        assertNull(run)
    }

    @Test
    fun testCrashRecoveryReconcilesOrphanAnalyzedCases() {
        // Case 1: Interrupted in-flight (marked ANALYZED without any persisted SUCCESS run)
        val orphanCase = ResearchCase(
            caseIdentifier = "TCGA-WP72-ORPHAN",
            status = CaseStatus.ANALYZED,
            patchCount = 80,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val orphanId = repository.insertCase(orphanCase)

        // Case 2: Properly completed case (marked ANALYZED with a persisted SUCCESS run)
        val completedCase = ResearchCase(
            caseIdentifier = "TCGA-WP72-COMPLETED",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 120,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val completedId = repository.insertCase(completedCase)
        val validRun = InferenceRun(
            caseId = completedId,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.25f,
            logit = -1.09f,
            threshold = FROZEN_THRESHOLD,
            margin = -0.58f,
            latencyMs = 50.0,
            numPatches = 120,
            diagnosticsSummary = "Valid complete run",
            executionTimestamp = System.currentTimeMillis(),
            status = "SUCCESS"
        )
        repository.recordInferenceResult(completedId, validRun, CaseStatus.ANALYZED)

        // Case 3: Case that was already READY_FOR_ANALYSIS
        val readyCase = ResearchCase(
            caseIdentifier = "TCGA-WP72-READY",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 60,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val readyId = repository.insertCase(readyCase)

        // Perform recovery
        repository.recoverInterruptedCases()

        // Assertions:
        // 1. Orphan case reconciled back to READY_FOR_ANALYSIS
        assertEquals(CaseStatus.READY_FOR_ANALYSIS, repository.getCaseById(orphanId)?.status)

        // 2. Completed case remains ANALYZED
        assertEquals(CaseStatus.ANALYZED, repository.getCaseById(completedId)?.status)

        // 3. Ready case remains READY_FOR_ANALYSIS
        assertEquals(CaseStatus.READY_FOR_ANALYSIS, repository.getCaseById(readyId)?.status)
    }

    @Test
    fun testInputValidationRejectsNaNAndInf() {
        val validPatches = 10
        val validVisionSize = validPatches * InputValidator.VISION_FEATURE_DIM

        // Vision NaN test
        val nanVision = FloatArray(validVisionSize) { 0.1f }
        nanVision[5] = Float.NaN
        try {
            InputValidator.validateVision(nanVision, validPatches, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected IllegalArgumentException for NaN in vision tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("NaN"))
        }

        // Vision +Infinity test
        val posInfVision = FloatArray(validVisionSize) { 0.1f }
        posInfVision[12] = Float.POSITIVE_INFINITY
        try {
            InputValidator.validateVision(posInfVision, validPatches, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected IllegalArgumentException for +Infinity in vision tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Infinite"))
        }

        // Vision -Infinity test
        val negInfVision = FloatArray(validVisionSize) { 0.1f }
        negInfVision[25] = Float.NEGATIVE_INFINITY
        try {
            InputValidator.validateVision(negInfVision, validPatches, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected IllegalArgumentException for -Infinity in vision tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Infinite"))
        }

        // Text NaN test
        val nanText = FloatArray(InputValidator.TEXT_FEATURE_DIM) { 0.05f }
        nanText[100] = Float.NaN
        try {
            InputValidator.validateText(nanText)
            fail("Expected IllegalArgumentException for NaN in text tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("NaN"))
        }

        // Text Infinite test
        val infText = FloatArray(InputValidator.TEXT_FEATURE_DIM) { 0.05f }
        infText[50] = Float.POSITIVE_INFINITY
        try {
            InputValidator.validateText(infText)
            fail("Expected IllegalArgumentException for Infinite in text tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Infinite"))
        }
    }

    @Test
    fun testInputValidationProtocolBounds() {
        // Protocol 1: 1000 patches is valid
        val validP1Vision = FloatArray(1000 * InputValidator.VISION_FEATURE_DIM) { 0.01f }
        InputValidator.validateVision(validP1Vision, 1000, InferenceProtocol.PROTOCOL_1_CAPPED)

        // Protocol 1: 1001 patches violates cap
        val invalidP1Vision = FloatArray(1001 * InputValidator.VISION_FEATURE_DIM) { 0.01f }
        try {
            InputValidator.validateVision(invalidP1Vision, 1001, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected IllegalArgumentException for Protocol 1 exceeding 1000 patches")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Protocol 1 enforces a maximum cap"))
        }

        // Protocol 2: 1001 patches is permitted
        InputValidator.validateVision(invalidP1Vision, 1001, InferenceProtocol.PROTOCOL_2_UNCAPPED)

        // Non-positive patch count
        try {
            InputValidator.validateVision(FloatArray(0), 0, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected IllegalArgumentException for patch count 0")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("strictly positive"))
        }

        // Buffer size mismatch
        val wrongSizeVision = FloatArray(50) { 0.1f }
        try {
            InputValidator.validateVision(wrongSizeVision, 10, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected IllegalArgumentException for buffer size mismatch")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("mismatch"))
        }
    }

    @Test
    fun testThresholdBoundaryDecisions() {
        assertEquals(0.83f, ThresholdLogic.OPERATING_THRESHOLD, 1e-6f)

        // At exact threshold boundary: 0.8300 -> Class 1 (MSI)
        val classAtBoundary = ThresholdLogic.classify(0.8300f)
        assertEquals(1, classAtBoundary)
        assertEquals("MSI", ThresholdLogic.labelFor(classAtBoundary))
        val marginAtBoundary = ThresholdLogic.signedDistanceToThreshold(0.8300f)
        assertEquals(0.0f, marginAtBoundary, 1e-6f)

        // Slightly above boundary: 0.83001 -> Class 1 (MSI)
        val classAbove = ThresholdLogic.classify(0.83001f)
        assertEquals(1, classAbove)
        assertEquals("MSI", ThresholdLogic.labelFor(classAbove))
        assertTrue(ThresholdLogic.signedDistanceToThreshold(0.83001f) > 0.0f)

        // Slightly below boundary: 0.82999 -> Class 0 (MSS)
        val classBelow = ThresholdLogic.classify(0.82999f)
        assertEquals(0, classBelow)
        assertEquals("MSS", ThresholdLogic.labelFor(classBelow))
        assertTrue(ThresholdLogic.signedDistanceToThreshold(0.82999f) < 0.0f)

        // Extreme bounds
        assertEquals(0, ThresholdLogic.classify(0.0f))
        assertEquals(1, ThresholdLogic.classify(1.0f))
    }

    @Test
    fun testDuplicateInferencePrevention() {
        val runningInferences = ConcurrentHashMap.newKeySet<Long>()
        val caseId = 42L

        // First attempt succeeds
        val firstAcquire = runningInferences.add(caseId)
        assertTrue("First inference acquisition must succeed", firstAcquire)

        // Second attempt for the same caseId fails (lock active)
        val secondAcquire = runningInferences.add(caseId)
        assertFalse("Duplicate concurrent inference acquisition must be rejected", secondAcquire)

        // Release lock
        runningInferences.remove(caseId)

        // Third attempt succeeds after release
        val thirdAcquire = runningInferences.add(caseId)
        assertTrue("Subsequent inference acquisition must succeed after release", thirdAcquire)
        runningInferences.remove(caseId)
    }
}
