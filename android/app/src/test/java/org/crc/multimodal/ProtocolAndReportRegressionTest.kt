package org.crc.multimodal

import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.data.ReviewNoteCategory
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputValidator
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.report.ReportGenerator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Regression Test Suite for Protocol 1 & Protocol 2 Fidelity and Research Report Integrity.
 *
 * Covers:
 * 1. Protocol 1 vs Protocol 2 enum representations, display strings, and full descriptions.
 * 2. Fallback behavior: missing or legacy protocol string maps to null -> "Protocol unavailable".
 * 3. Bug-01 Substring fix: verifying "PROTOCOL_2_UNCAPPED" never displays as "Protocol 1".
 * 4. End-to-end repository persistence and restoration for Protocol 1 and Protocol 2 runs.
 * 5. Report generation for Protocol 1 and Protocol 2 runs matching executed result metadata.
 * 6. Report generation resilience with missing/null runs, null protocols, and special characters.
 * 7. Decision boundary safety invariant (tau = 0.83) across both protocols.
 */
class ProtocolAndReportRegressionTest {

    private lateinit var repository: InMemoryWorkstationRepository

    @Before
    fun setUp() {
        repository = InMemoryWorkstationRepository()
    }

    @Test
    fun testProtocolEnumDisplayAndDescriptions() {
        assertEquals("Protocol 1", InferenceProtocol.PROTOCOL_1_CAPPED.displayName)
        assertEquals("Protocol 2", InferenceProtocol.PROTOCOL_2_UNCAPPED.displayName)

        assertEquals("Protocol 1 (Capped ≤ 1,000 max patches)", InferenceProtocol.PROTOCOL_1_CAPPED.fullDescription)
        assertEquals("Protocol 2 (Uncapped candidate patches preserved)", InferenceProtocol.PROTOCOL_2_UNCAPPED.fullDescription)
    }

    @Test
    fun testProtocolFormattingAndSubstringBugFix() {
        // Direct test against the exact bug: contains("CAPPED") matched both
        val p1 = InferenceProtocol.PROTOCOL_1_CAPPED
        val p2 = InferenceProtocol.PROTOCOL_2_UNCAPPED

        // Using formatProtocol:
        assertEquals("Protocol 1", InferenceProtocol.formatProtocol(p1))
        assertEquals("Protocol 2", InferenceProtocol.formatProtocol(p2))

        // Ensure PROTOCOL_2_UNCAPPED NEVER formats as Protocol 1
        assertFalse(InferenceProtocol.formatProtocol(p2) == "Protocol 1")

        // Null / Legacy protocol handling: must return "Protocol unavailable"
        assertEquals("Protocol unavailable", InferenceProtocol.formatProtocol(null))
    }

    @Test
    fun testProtocolFromStringOrNullParsing() {
        // Valid exact enum names
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, InferenceProtocol.fromStringOrNull("PROTOCOL_1_CAPPED"))
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, InferenceProtocol.fromStringOrNull("PROTOCOL_2_UNCAPPED"))

        // Common normalized variations
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, InferenceProtocol.fromStringOrNull("PROTOCOL_1"))
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, InferenceProtocol.fromStringOrNull("Protocol 1"))
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, InferenceProtocol.fromStringOrNull("protocol1"))
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, InferenceProtocol.fromStringOrNull("PROTOCOL_2"))
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, InferenceProtocol.fromStringOrNull("Protocol 2"))
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, InferenceProtocol.fromStringOrNull("protocol2"))

        // Invalid, empty, or unknown strings -> must return null, NEVER default to Protocol 1
        assertNull(InferenceProtocol.fromStringOrNull(null))
        assertNull(InferenceProtocol.fromStringOrNull(""))
        assertNull(InferenceProtocol.fromStringOrNull("   "))
        assertNull(InferenceProtocol.fromStringOrNull("UNKNOWN_PROTOCOL"))
        assertNull(InferenceProtocol.fromStringOrNull("LEGACY_CORRUPT_VALUE"))
    }

    @Test
    fun testProtocol1ExecutionAndRestoration() {
        val caseP1 = ResearchCase(
            caseIdentifier = "TCGA-P1-001",
            specimenDescription = "Protocol 1 test specimen",
            researchContext = "Standard capped protocol evaluation",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 450,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val caseId = repository.insertCase(caseP1)

        val runP1 = InferenceRun(
            caseId = caseId,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.9125f,
            logit = 2.3456f,
            threshold = 0.83f,
            margin = 0.0825f,
            latencyMs = 85.4,
            numPatches = 450,
            diagnosticsSummary = "Evaluation Mode: Protocol 1",
            status = "SUCCESS"
        )
        val runId = repository.recordInferenceResult(caseId, runP1, CaseStatus.ANALYZED)

        // Restore case and run
        val restoredCase = repository.getCaseById(caseId)
        assertNotNull(restoredCase)
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, restoredCase?.protocol)
        assertEquals("Protocol 1", InferenceProtocol.formatProtocol(restoredCase?.protocol))

        val restoredRun = repository.getLatestRunForCase(caseId)
        assertNotNull(restoredRun)
        assertEquals(runId, restoredRun?.id)
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, restoredRun?.protocol)
        assertEquals("Protocol 1", InferenceProtocol.formatProtocol(restoredRun?.protocol))
    }

    @Test
    fun testProtocol2ExecutionAndRestorationForSpecimenCase() {
        // Specimen TCGA-AZ-4315 with 2,140 patches (the exact specimen from observed bug)
        val caseP2 = ResearchCase(
            caseIdentifier = "TCGA-AZ-4315",
            specimenDescription = "Colon adenocarcinoma resection specimen",
            researchContext = "Uncapped high-patch-count candidate evaluation",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 2140,
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED
        )
        val caseId = repository.insertCase(caseP2)

        val runP2 = InferenceRun(
            caseId = caseId,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED,
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.2982f,
            logit = -0.8560f,
            threshold = 0.83f,
            margin = -0.5318f,
            latencyMs = 226.90,
            numPatches = 2140,
            diagnosticsSummary = "Evaluation Mode: Protocol 2",
            status = "SUCCESS"
        )
        val runId = repository.recordInferenceResult(caseId, runP2, CaseStatus.ANALYZED)

        // Restore case and run
        val restoredCase = repository.getCaseById(caseId)
        assertNotNull(restoredCase)
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, restoredCase?.protocol)
        assertEquals("Protocol 2", InferenceProtocol.formatProtocol(restoredCase?.protocol))

        val restoredRun = repository.getLatestRunForCase(caseId)
        assertNotNull(restoredRun)
        assertEquals(runId, restoredRun?.id)
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, restoredRun?.protocol)
        assertEquals("Protocol 2", InferenceProtocol.formatProtocol(restoredRun?.protocol))
        assertEquals(2140, restoredRun?.numPatches)
        assertEquals(0.2982f, restoredRun?.probability ?: 0f, 1e-4f)
        assertEquals(0.83f, restoredRun?.threshold ?: 0f, 1e-4f)
    }

    @Test
    fun testLegacyUnknownProtocolDisplaysAsUnavailable() {
        val legacyCase = ResearchCase(
            caseIdentifier = "LEGACY-CASE-999",
            specimenDescription = "Legacy record with missing protocol",
            status = CaseStatus.ANALYZED,
            patchCount = 300,
            protocol = null // Unknown / Legacy
        )
        val caseId = repository.insertCase(legacyCase)

        val legacyRun = InferenceRun(
            caseId = caseId,
            modelIdentifier = "fusion_model_fp32.onnx",
            protocol = null, // Unknown / Legacy
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.1500f,
            logit = -1.7346f,
            threshold = 0.83f,
            margin = -0.6800f,
            latencyMs = 50.0,
            numPatches = 300
        )
        repository.recordInferenceResult(caseId, legacyRun, CaseStatus.ANALYZED)

        val restoredCase = repository.getCaseById(caseId)
        assertNull(restoredCase?.protocol)
        assertEquals("Protocol unavailable", InferenceProtocol.formatProtocol(restoredCase?.protocol))

        val restoredRun = repository.getLatestRunForCase(caseId)
        assertNull(restoredRun?.protocol)
        assertEquals("Protocol unavailable", InferenceProtocol.formatProtocol(restoredRun?.protocol))
    }

    @Test
    fun testReportGenerationForProtocol2RunPreservesAllMetadata() {
        val caseP2 = ResearchCase(
            caseIdentifier = "TCGA-AZ-4315",
            specimenDescription = "Adenocarcinoma resection (FFPE)",
            researchContext = "Holdout candidate evaluation under Protocol 2",
            status = CaseStatus.ANALYZED,
            patchCount = 2140,
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED
        )
        val runP2 = InferenceRun(
            caseId = 101L,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED,
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.2982f,
            logit = -0.8560f,
            threshold = 0.83f,
            margin = -0.5318f,
            latencyMs = 226.90,
            numPatches = 2140,
            diagnosticsSummary = "Execution Mode: Protocol 2 (2140 patches, latency 226.90ms)",
            status = "SUCCESS"
        )
        val notes = listOf(
            ReviewNote(
                caseId = 101L,
                category = ReviewNoteCategory.HISTOLOGICAL_OBSERVATION,
                authorLabel = "Lead Pathologist",
                content = "Evaluated under Protocol 2 with 2,140 patches."
            )
        )

        val report = ReportGenerator.generateTextReport(caseP2, runP2, notes)

        assertTrue(report.isNotEmpty())
        assertTrue("Report must contain Specimen ID", report.contains("TCGA-AZ-4315"))
        assertTrue("Report must contain 2140 patches", report.contains("2140 patches"))
        assertTrue("Report Section 2 must display Protocol 2", report.contains("Evaluation Protocol: Protocol 2"))
        assertTrue("Report Section 3 must display Protocol 2", report.contains("Execution Protocol : Protocol 2"))
        assertTrue("Report must contain MSS", report.contains("MSS (Class 0)"))
        assertTrue("Report must contain evaluated probability 0.2982", report.contains("0.2982"))
        assertTrue("Report must contain fixed threshold 0.83", report.contains("0.8300"))
        assertTrue("Report must preserve diagnostics", report.contains("Execution Mode: Protocol 2"))
        assertTrue("Report must contain review note", report.contains("Evaluated under Protocol 2 with 2,140 patches."))
    }

    @Test
    fun testReportGenerationWithPendingInference() {
        val pendingCase = ResearchCase(
            caseIdentifier = "TCGA-PENDING-001",
            specimenDescription = "Pending specimen",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 100,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val report = ReportGenerator.generateTextReport(pendingCase, null, emptyList())

        assertTrue(report.isNotEmpty())
        assertTrue(report.contains("TCGA-PENDING-001"))
        assertTrue(report.contains("Evaluation Protocol: Protocol 1"))
        assertTrue(report.contains("Run Status         : PENDING"))
    }

    @Test
    fun testDecisionThresholdSafetyInvariantAcrossProtocols() {
        // Operating threshold is strictly 0.83
        assertEquals(0.83f, ThresholdLogic.OPERATING_THRESHOLD, 1e-6f)

        // Protocol 1: P = 0.8300 -> MSI (1)
        assertEquals(1, ThresholdLogic.classify(0.8300f))
        assertEquals("MSI", ThresholdLogic.labelFor(1))

        // Protocol 2: P = 0.8299 -> MSS (0)
        assertEquals(0, ThresholdLogic.classify(0.8299f))
        assertEquals("MSS", ThresholdLogic.labelFor(0))

        // Specimen TCGA-AZ-4315: P = 0.2982 -> MSS (0)
        assertEquals(0, ThresholdLogic.classify(0.2982f))
        assertEquals("MSS", ThresholdLogic.labelFor(ThresholdLogic.classify(0.2982f)))
    }
}
