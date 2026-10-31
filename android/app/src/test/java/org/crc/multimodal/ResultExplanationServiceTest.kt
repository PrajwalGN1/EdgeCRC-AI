package org.crc.multimodal

import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.data.ReviewNoteCategory
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.report.ResultExplanationService
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultExplanationServiceTest {

    private fun createSampleCase(identifier: String = "TCGA-TEST-001"): ResearchCase {
        return ResearchCase(
            id = 101L,
            caseIdentifier = identifier,
            specimenDescription = "Primary colorectal adenocarcinoma biopsy",
            researchContext = "Standard benchmark evaluation",
            status = CaseStatus.ANALYZED,
            patchCount = 345,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
    }

    private fun createSampleRun(
        probability: Float,
        predictedClass: Int,
        predictedLabel: String,
        logit: Float = 0.0f
    ): InferenceRun {
        return InferenceRun(
            id = 202L,
            caseId = 101L,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = predictedClass,
            predictedLabel = predictedLabel,
            probability = probability,
            logit = logit,
            threshold = 0.83f,
            margin = probability - 0.83f,
            latencyMs = 120.5,
            numPatches = 345,
            diagnosticsSummary = "Single-patient sequential CPU execution completed.",
            executionTimestamp = 1710000000000L,
            status = "SUCCESS"
        )
    }

    @Test
    fun testMsiClassificationExactlyAtThreshold() {
        val researchCase = createSampleCase()
        val run = createSampleRun(
            probability = 0.8300f,
            predictedClass = ThresholdLogic.CLASS_MSI,
            predictedLabel = ThresholdLogic.LABEL_MSI
        )

        val explanation = ResultExplanationService.explainResult(researchCase, run)

        assertTrue(explanation.isConsistent)
        assertFalse(explanation.hasValidationWarnings)
        assertTrue(explanation.classificationSummary.contains("MSI"))
        assertTrue(explanation.thresholdExplanation.contains("0.8300"))
        assertTrue(explanation.thresholdExplanation.contains("+0.0000 above") || explanation.thresholdExplanation.contains("0.0000"))
    }

    @Test
    fun testMsiClassificationAboveThreshold() {
        val researchCase = createSampleCase()
        val run = createSampleRun(
            probability = 0.9525f,
            predictedClass = ThresholdLogic.CLASS_MSI,
            predictedLabel = ThresholdLogic.LABEL_MSI,
            logit = 2.998f
        )

        val explanation = ResultExplanationService.explainResult(researchCase, run)

        assertTrue(explanation.isConsistent)
        assertFalse(explanation.hasValidationWarnings)
        assertTrue(explanation.classificationSummary.contains("MSI"))
        assertTrue(explanation.thresholdExplanation.contains("0.9525"))
        assertTrue(explanation.thresholdExplanation.contains("+0.1225 above"))
    }

    @Test
    fun testMssClassificationBelowThreshold() {
        val researchCase = createSampleCase()
        val run = createSampleRun(
            probability = 0.8299f,
            predictedClass = ThresholdLogic.CLASS_MSS,
            predictedLabel = ThresholdLogic.LABEL_MSS,
            logit = 1.585f
        )

        val explanation = ResultExplanationService.explainResult(researchCase, run)

        assertTrue(explanation.isConsistent)
        assertFalse(explanation.hasValidationWarnings)
        assertTrue(explanation.classificationSummary.contains("MSS"))
        assertTrue(explanation.thresholdExplanation.contains("0.8299"))
        assertTrue(explanation.thresholdExplanation.contains("below"))
    }

    @Test
    fun testMssClassificationLowProbability() {
        val researchCase = createSampleCase()
        val run = createSampleRun(
            probability = 0.1625f,
            predictedClass = ThresholdLogic.CLASS_MSS,
            predictedLabel = ThresholdLogic.LABEL_MSS,
            logit = -1.639f
        )

        val explanation = ResultExplanationService.explainResult(researchCase, run)

        assertTrue(explanation.isConsistent)
        assertFalse(explanation.hasValidationWarnings)
        assertTrue(explanation.classificationSummary.contains("MSS"))
        assertTrue(explanation.thresholdExplanation.contains("0.1625"))
        assertTrue(explanation.thresholdExplanation.contains("below"))
    }

    @Test
    fun testHandlingPendingInferenceRun() {
        val researchCase = createSampleCase()

        val explanation = ResultExplanationService.explainResult(researchCase, null)

        assertTrue(explanation.hasValidationWarnings)
        assertTrue(explanation.validationWarnings.any { it.contains("PENDING") })
        assertTrue(explanation.classificationSummary.contains("PENDING"))
        assertEquals("1.0.0", explanation.templateVersion)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testResearchCaseRejectsBlankCaseIdentifier() {
        createSampleCase(identifier = "")
    }

    @Test
    fun testHandlingMinimalCaseMetadata() {
        val minimalCase = ResearchCase(
            id = 101L,
            caseIdentifier = "TCGA-MINIMAL",
            specimenDescription = "",
            researchContext = "",
            status = CaseStatus.ANALYZED,
            patchCount = 10,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val run = createSampleRun(0.90f, ThresholdLogic.CLASS_MSI, ThresholdLogic.LABEL_MSI)
        val explanation = ResultExplanationService.explainResult(minimalCase, run)

        assertTrue(explanation.isConsistent)
        assertNotNull(explanation.fullFormattedExplanation)
        assertTrue(explanation.fullFormattedExplanation.contains("TCGA-MINIMAL"))
    }

    @Test
    fun testContradictorySavedResultAndProbability() {
        val researchCase = createSampleCase()
        // Contradiction: Probability is 0.95 (should be MSI), but saved record has MSS (Class 0)
        val run = createSampleRun(
            probability = 0.9500f,
            predictedClass = ThresholdLogic.CLASS_MSS,
            predictedLabel = ThresholdLogic.LABEL_MSS
        )

        val explanation = ResultExplanationService.explainResult(researchCase, run)

        assertFalse(explanation.isConsistent)
        assertTrue(explanation.hasValidationWarnings)
        assertTrue(explanation.validationWarnings.any { it.contains("DISCREPANCY DETECTED") })
        assertTrue(explanation.fullFormattedExplanation.contains("VALIDATION WARNINGS"))
    }

    @Test
    fun testDeterministicExplanationIsReproducible() {
        val researchCase = createSampleCase()
        val run = createSampleRun(0.9123f, ThresholdLogic.CLASS_MSI, ThresholdLogic.LABEL_MSI)
        val notes = listOf(
            ReviewNote(
                id = 1L,
                caseId = researchCase.id,
                category = ReviewNoteCategory.HISTOLOGICAL_OBSERVATION,
                authorLabel = "Lead Pathologist",
                content = "Mucinous differentiation observed in right colon biopsy.",
                createdAt = 1710000000000L
            )
        )

        val explanation1 = ResultExplanationService.explainResult(researchCase, run, notes)
        val explanation2 = ResultExplanationService.explainResult(researchCase, run, notes)

        assertEquals(explanation1.fullFormattedExplanation, explanation2.fullFormattedExplanation)
        assertEquals(explanation1.thresholdExplanation, explanation2.thresholdExplanation)
        assertEquals(explanation1.classificationSummary, explanation2.classificationSummary)
        assertEquals(explanation1.sourceRecordSummary, explanation2.sourceRecordSummary)
        assertEquals(explanation1.reviewerNotesSummary, explanation2.reviewerNotesSummary)
    }

    @Test
    fun testResearchDisclaimerIsPresentAndExplicit() {
        val researchCase = createSampleCase()
        val run = createSampleRun(0.90f, ThresholdLogic.CLASS_MSI, ThresholdLogic.LABEL_MSI)

        val explanation = ResultExplanationService.explainResult(researchCase, run)

        assertTrue(explanation.clinicalDisclaimer.contains("RESEARCH USE ONLY"))
        assertTrue(explanation.clinicalDisclaimer.contains("NOT a confirmed clinical diagnosis"))
        assertTrue(explanation.clinicalDisclaimer.contains("IHC/PCR"))
        assertTrue(explanation.fullFormattedExplanation.contains(explanation.clinicalDisclaimer))
    }
}
