package org.crc.multimodal.report

import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.inference.InferenceProtocol
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Deterministic report generator producing structured, verifiable research reports
 * from persisted cases, runs, and review notes.
 */
object ReportGenerator {

    private const val MODEL_FP32_SHA256 = "0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73"

    fun generateTextReport(
        researchCase: ResearchCase,
        inferenceRun: InferenceRun?,
        reviewNotes: List<ReviewNote>,
        bundle: org.crc.multimodal.inference.AuthenticFeatureBundle? = null
    ): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.US)
        dateFormat.timeZone = java.util.TimeZone.getTimeZone("UTC")
        val currentDateStr = dateFormat.format(Date())

        val manifest = org.crc.multimodal.security.ProvenanceAuditManager.generateManifest(
            researchCase = researchCase,
            inferenceRun = inferenceRun,
            bundle = bundle
        )

        return buildString {
            appendLine("================================================================================")
            appendLine("          CRC MULTIMODAL PATHOLOGY RESEARCH WORKSTATION REPORT")
            appendLine("                     Multimodal Fusion Workspace")
            appendLine("================================================================================")
            appendLine("Report Generated   : $currentDateStr")
            appendLine("Investigational Tag: LOCAL RESEARCH ENGINE · ZERO NETWORK TRANSMISSION")
            appendLine()

            appendLine("--- SECTION 1: RESEARCH CASE IDENTIFICATION ---")
            appendLine("Research Case ID   : ${researchCase.caseIdentifier}")
            appendLine("Case Status        : ${researchCase.status.displayName}")
            appendLine("Created Timestamp  : ${dateFormat.format(Date(researchCase.createdAt))}")
            appendLine("Updated Timestamp  : ${dateFormat.format(Date(researchCase.updatedAt))}")
            appendLine("Specimen Summary   : ${researchCase.specimenDescription.ifBlank { "Not specified" }}")
            appendLine("Research Context   : ${researchCase.researchContext.ifBlank { "None provided" }}")
            appendLine()

            appendLine("--- SECTION 2: SPECIMEN & MULTIMODAL INPUT SPECIFICATION ---")
            appendLine("Slide Patch Count  : ${researchCase.patchCount} patches")
            val effectiveProtocol = inferenceRun?.protocol ?: researchCase.protocol
            appendLine("Evaluation Protocol: ${InferenceProtocol.formatProtocol(effectiveProtocol)}")
            appendLine("Vision Tensor Shape: [1, ${researchCase.patchCount}, 1024] Float32 (PHIKON-V2 precomputed)")
            appendLine("Text Tensor Shape  : [1, 768] Float32 (ClinicalBERT precomputed)")
            appendLine()

            appendLine("--- SECTION 3: MULTIMODAL FUSION FINDINGS ---")
            if (inferenceRun != null) {
                appendLine("Run Status         : ${inferenceRun.status}")
                appendLine("Execution Protocol : ${InferenceProtocol.formatProtocol(inferenceRun.protocol)}")
                appendLine("Classification     : ${inferenceRun.predictedLabel} (Class ${inferenceRun.predictedClass})")
                appendLine("Evaluated Probability: ${String.format(Locale.US, "%.5f", inferenceRun.probability)}")
                appendLine("Decision Threshold : ${String.format(Locale.US, "%.4f", inferenceRun.threshold)} (Fixed research boundary)")
                val marginStr = if (inferenceRun.margin >= 0) "+${String.format(Locale.US, "%.5f", inferenceRun.margin)}"
                else String.format(Locale.US, "%.5f", inferenceRun.margin)
                appendLine("Threshold Margin   : $marginStr vs 0.8300")
                appendLine("Model Output Logit : ${String.format(Locale.US, "%.5f", inferenceRun.logit)}")
                appendLine("Inference Latency  : ${String.format(Locale.US, "%.2f", inferenceRun.latencyMs)} ms")
                appendLine("Execution Time     : ${dateFormat.format(Date(inferenceRun.executionTimestamp))}")
            } else {
                appendLine("Run Status         : PENDING (No inference execution recorded yet for this case)")
            }
            appendLine()

            appendLine("--- SECTION 4: MODEL & INFERENCE PROVENANCE ---")
            appendLine("Active Engine      : Local Android ONNX Runtime 1.18.0")
            appendLine("Execution Provider : CPUExecutionProvider (Float32 single-patient sequential)")
            appendLine("Model Artifact     : fusion_model_fp32.onnx (20.04 MB FP32)")
            appendLine("Model SHA-256      : $MODEL_FP32_SHA256")
            appendLine("Preprocessing Ver  : ${manifest.encoderProvenance.preprocessingManifestVersion}")
            appendLine("Encoder Status     : ${manifest.encoderProvenance.encoderStatus}")
            appendLine("Text Tensor SHA-256: ${manifest.inputFingerprints.textTensorSha256}")
            appendLine("Vision Tensor SHA  : ${manifest.inputFingerprints.visionPatchesSha256}")
            if (manifest.inputFingerprints.bundleSha256 != null) {
                appendLine("Bundle Payload SHA : ${manifest.inputFingerprints.bundleSha256}")
            }
            appendLine("Data Privacy       : 100% on-device CPU execution, 0 bytes transmitted")
            appendLine("Integrity Type     : ${manifest.integrityStatement.checksumType}")
            appendLine("Integrity Digest   : ${manifest.integrityStatement.manifestPayloadSha256}")
            appendLine("Integrity Notice   : ${manifest.integrityStatement.disclaimer}")
            appendLine()

            if (inferenceRun?.diagnosticsSummary != null && inferenceRun.diagnosticsSummary.isNotBlank()) {
                appendLine("--- SECTION 5: TECHNICAL EXECUTION DIAGNOSTICS ---")
                appendLine(inferenceRun.diagnosticsSummary.trim())
                appendLine()
            }

            appendLine("--- SECTION 6: CASE REVIEW NOTES ---")
            if (reviewNotes.isNotEmpty()) {
                reviewNotes.forEachIndexed { idx, note ->
                    appendLine("  [${idx + 1}] Category: ${note.category.displayName}")
                    appendLine("      Author  : ${note.authorLabel}")
                    appendLine("      Date    : ${dateFormat.format(Date(note.createdAt))}")
                    appendLine("      Note    : ${note.content}")
                }
            } else {
                appendLine("  No review notes recorded.")
            }
            appendLine()

            appendLine("--- SECTION 7: RESPONSIBLE-USE RESEARCH DISCLAIMER ---")
            appendLine("INVESTIGATIONAL RESEARCH USE ONLY. Not approved for clinical diagnosis or patient care.")
            appendLine("This report summarizes experimental machine learning benchmark outputs from an edge AI")
            appendLine("research prototype. The MSI/MSS evaluation score does not substitute for certified clinical")
            appendLine("IHC/PCR microsatellite instability testing.")
            appendLine("================================================================================")
        }
    }

    fun generateProvenanceManifest(
        researchCase: ResearchCase,
        inferenceRun: InferenceRun?,
        bundle: org.crc.multimodal.inference.AuthenticFeatureBundle? = null
    ): org.crc.multimodal.security.SpecimenProvenanceManifest {
        return org.crc.multimodal.security.ProvenanceAuditManager.generateManifest(
            researchCase = researchCase,
            inferenceRun = inferenceRun,
            bundle = bundle
        )
    }

    fun generateProvenanceManifestJson(
        researchCase: ResearchCase,
        inferenceRun: InferenceRun?,
        bundle: org.crc.multimodal.inference.AuthenticFeatureBundle? = null
    ): String {
        val manifest = generateProvenanceManifest(researchCase, inferenceRun, bundle)
        return org.crc.multimodal.security.ProvenanceAuditManager.serializeManifestJson(manifest)
    }
}
