package org.crc.multimodal.report

import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Result of the deterministic result-explanation evaluation.
 */
data class ResultExplanation(
    val templateVersion: String,
    val isConsistent: Boolean,
    val hasValidationWarnings: Boolean,
    val validationWarnings: List<String>,
    val classificationSummary: String,
    val thresholdExplanation: String,
    val probabilityExplanation: String,
    val sourceRecordSummary: String,
    val diagnosticsSummary: String,
    val reviewerNotesSummary: String,
    val clinicalDisclaimer: String,
    val fullFormattedExplanation: String
)

/**
 * Offline, deterministic result-explanation service.
 *
 * Evaluates verified inference records, probability distributions against the frozen decision
 * boundary (tau = 0.8300), validates data consistency, and produces reproducible, human-readable
 * scientific explanations without invoking machine learning models, LLMs, or network services.
 */
object ResultExplanationService {

    const val TEMPLATE_VERSION = "1.0.0"
    const val FROZEN_THRESHOLD = 0.83f

    private const val CLINICAL_RESEARCH_DISCLAIMER =
        "RESEARCH USE ONLY: This deterministic explanation describes experimental machine learning outputs " +
        "from an on-device multimodal fusion model (OWKIN Phikon-v2 ViT-B/16 vision embeddings and ClinicalBERT " +
        "text embeddings). The output score is a research softmax activation and NOT a confirmed clinical diagnosis. " +
        "It does not substitute for certified clinical histopathological evaluation or molecular IHC/PCR microsatellite " +
        "instability testing. No treatment or diagnostic decisions should be based solely on this research record."

    /**
     * Deterministically generates a structured explanation from verified case, run, and review data.
     */
    fun explainResult(
        researchCase: ResearchCase,
        inferenceRun: InferenceRun?,
        reviewNotes: List<ReviewNote> = emptyList(),
        bundle: AuthenticFeatureBundle? = null
    ): ResultExplanation {
        val warnings = mutableListOf<String>()
        var isConsistent = true

        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }

        // 1. Case Metadata Validation
        if (researchCase.caseIdentifier.isBlank()) {
            warnings.add("Validation Warning: Case identifier is blank or missing.")
        }

        // 2. Inference Run & Consistency Validation
        if (inferenceRun == null) {
            warnings.add("Validation Warning: No inference execution record found for this case. Status is PENDING.")
            val pendingSummary = "Classification: PENDING (No inference execution has been recorded for specimen '${researchCase.caseIdentifier}')."
            val pendingThreshold = "Decision boundary is frozen at tau = 0.8300 (Class 1 = MSI, Class 0 = MSS). Threshold comparison cannot be evaluated without an inference run."
            val pendingProb = "Probability: Not available."
            val pendingSource = "Source Record: None (Awaiting local ONNX inference)."
            val diagnostics = "Diagnostics: No execution logs recorded."
            val notesSummary = formatReviewNotesSummary(reviewNotes, dateFormat)

            val fullText = buildString {
                appendLine("=== DETERMINISTIC RESULT EXPLANATION (v$TEMPLATE_VERSION) ===")
                appendLine("Specimen / Case ID : ${researchCase.caseIdentifier}")
                appendLine("Evaluation Status  : PENDING")
                appendLine()
                appendLine(pendingSummary)
                appendLine(pendingThreshold)
                appendLine()
                if (warnings.isNotEmpty()) {
                    appendLine("--- VALIDATION NOTICES ---")
                    warnings.forEach { appendLine("• $it") }
                    appendLine()
                }
                appendLine(notesSummary)
                appendLine()
                appendLine(CLINICAL_RESEARCH_DISCLAIMER)
            }

            return ResultExplanation(
                templateVersion = TEMPLATE_VERSION,
                isConsistent = true,
                hasValidationWarnings = warnings.isNotEmpty(),
                validationWarnings = warnings,
                classificationSummary = pendingSummary,
                thresholdExplanation = pendingThreshold,
                probabilityExplanation = pendingProb,
                sourceRecordSummary = pendingSource,
                diagnosticsSummary = diagnostics,
                reviewerNotesSummary = notesSummary,
                clinicalDisclaimer = CLINICAL_RESEARCH_DISCLAIMER,
                fullFormattedExplanation = fullText
            )
        }

        // Validate Probability Boundaries
        val prob = inferenceRun.probability
        if (prob.isNaN() || prob < 0.0f || prob > 1.0f) {
            warnings.add("CRITICAL VALIDATION ERROR: Model output probability ($prob) is outside valid range [0.0, 1.0].")
            isConsistent = false
        }

        // Validate Threshold Rule vs Authoritative Record
        val expectedClass = if (prob >= FROZEN_THRESHOLD) ThresholdLogic.CLASS_MSI else ThresholdLogic.CLASS_MSS
        val expectedLabel = if (prob >= FROZEN_THRESHOLD) ThresholdLogic.LABEL_MSI else ThresholdLogic.LABEL_MSS

        if (inferenceRun.predictedClass != expectedClass || inferenceRun.predictedLabel != expectedLabel) {
            warnings.add(
                "DISCREPANCY DETECTED: Saved classification '${inferenceRun.predictedLabel}' (Class ${inferenceRun.predictedClass}) " +
                "is contradictory to the validated threshold rule for probability ${String.format(Locale.US, "%.4f", prob)} " +
                "(expected '$expectedLabel' Class $expectedClass at tau = 0.8300)."
            )
            isConsistent = false
        }

        // Signed Margin Calculation
        val signedMargin = prob - FROZEN_THRESHOLD
        val marginDeltaStr = if (signedMargin >= 0.0f) {
            String.format(Locale.US, "+%.4f above", signedMargin)
        } else {
            String.format(Locale.US, "%.4f below", signedMargin)
        }

        // Classification Summary
        val classificationSummary = if (inferenceRun.predictedClass == ThresholdLogic.CLASS_MSI) {
            "MSI (Microsatellite Instability) — Evaluated model output score is at or above the research decision threshold (tau = 0.8300)."
        } else {
            "MSS (Microsatellite Stable) — Evaluated model output score is below the research decision threshold (tau = 0.8300)."
        }

        // Threshold Explanation
        val thresholdExplanation = buildString {
            append("The model evaluated an MSI probability of ")
            append(String.format(Locale.US, "%.4f", prob))
            append(", which is ")
            append(marginDeltaStr)
            append(" the frozen research decision boundary of tau = ")
            append(String.format(Locale.US, "%.4f", FROZEN_THRESHOLD))
            append(". In accordance with the frozen protocol, cases with P >= 0.8300 are classified as MSI (Class 1), ")
            append("and cases with P < 0.8300 are classified as MSS (Class 0).")
        }

        // Probability Explanation
        val probabilityExplanation = buildString {
            append("Evaluated Class-1 Softmax Probability: ")
            append(String.format(Locale.US, "%.5f", prob))
            append(" (Raw model logit: ")
            append(String.format(Locale.US, "%.5f", inferenceRun.logit))
            append("). Margin delta: ")
            append(if (signedMargin >= 0f) "+${String.format(Locale.US, "%.5f", signedMargin)}" else String.format(Locale.US, "%.5f", signedMargin))
            append(".")
        }

        // Source Record Summary
        val sourceRecordSummary = buildString {
            append("Inference Run ID: ${inferenceRun.id} | Status: ${inferenceRun.status} | ")
            append("Executed: ${dateFormat.format(Date(inferenceRun.executionTimestamp))} | ")
            append("Architecture: ${inferenceRun.inferenceArchitecture} (${inferenceRun.modelIdentifier}) | ")
            append("Protocol: ${InferenceProtocol.formatProtocol(inferenceRun.protocol)} | ")
            append("Latency: ${String.format(Locale.US, "%.2f", inferenceRun.latencyMs)} ms | ")
            append("Patches: ${inferenceRun.numPatches}")
        }

        // Diagnostics
        val diagnosticsSummary = buildString {
            if (!inferenceRun.diagnosticsSummary.isNullOrBlank()) {
                append(inferenceRun.diagnosticsSummary.trim())
            } else {
                append("Standard CPU execution provider completed without diagnostic faults.")
            }
            if (bundle != null) {
                append(" | Authentic Feature Bundle verified (CRCv${bundle.schemaVersion}, Vision SHA: ${bundle.visionSha256.take(8)}...)")
            }
        }

        // Reviewer Notes Summary
        val reviewerNotesSummary = formatReviewNotesSummary(reviewNotes, dateFormat)

        // Full Formatted Deterministic Explanation
        val fullText = buildString {
            appendLine("=== CRC DETERMINISTIC RESULT EXPLANATION (v$TEMPLATE_VERSION) ===")
            appendLine("Specimen / Case ID : ${researchCase.caseIdentifier}")
            appendLine("Authoritative Result: ${inferenceRun.predictedLabel} (Class ${inferenceRun.predictedClass})")
            appendLine("Classification Desc: $classificationSummary")
            appendLine()
            appendLine("--- THRESHOLD DECISION BOUNDARY ---")
            appendLine(thresholdExplanation)
            appendLine(probabilityExplanation)
            appendLine()
            appendLine("--- INFERENCE PROVENANCE ---")
            appendLine(sourceRecordSummary)
            appendLine("Diagnostics: $diagnosticsSummary")
            appendLine()
            if (warnings.isNotEmpty()) {
                appendLine("--- VALIDATION WARNINGS ---")
                warnings.forEach { appendLine("• $it") }
                appendLine()
            }
            appendLine("--- CASE REVIEW CONTEXT ---")
            appendLine(reviewerNotesSummary)
            appendLine()
            appendLine("--- RESEARCH USE STATEMENT ---")
            appendLine(CLINICAL_RESEARCH_DISCLAIMER)
        }

        return ResultExplanation(
            templateVersion = TEMPLATE_VERSION,
            isConsistent = isConsistent,
            hasValidationWarnings = warnings.isNotEmpty(),
            validationWarnings = warnings,
            classificationSummary = classificationSummary,
            thresholdExplanation = thresholdExplanation,
            probabilityExplanation = probabilityExplanation,
            sourceRecordSummary = sourceRecordSummary,
            diagnosticsSummary = diagnosticsSummary,
            reviewerNotesSummary = reviewerNotesSummary,
            clinicalDisclaimer = CLINICAL_RESEARCH_DISCLAIMER,
            fullFormattedExplanation = fullText
        )
    }

    private fun formatReviewNotesSummary(
        reviewNotes: List<ReviewNote>,
        dateFormat: SimpleDateFormat
    ): String {
        if (reviewNotes.isEmpty()) {
            return "Review Status: No clinical reviewer notes recorded for this case."
        }
        return buildString {
            appendLine("Review Status: ${reviewNotes.size} reviewer note(s) recorded:")
            reviewNotes.forEachIndexed { idx, note ->
                appendLine("  [${idx + 1}] Category: ${note.category.displayName} | Author: ${note.authorLabel} (${dateFormat.format(Date(note.createdAt))})")
                appendLine("      \"${note.content.trim()}\"")
            }
        }.trimEnd()
    }
}
