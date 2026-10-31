package org.crc.multimodal.security

/**
 * Domain data models for security, provenance tracking, and audit compliance.
 *
 * Mandated invariants (WP7.5):
 * - Plaintext SHA-256 digests represent data integrity checksums, NOT authenticated digital signatures.
 * - Missing or upstream encoder provenance must be marked explicitly as [UPSTREAM_PRECOMPUTED_UNVERIFIED].
 * - Preprocessing manifest version: "phase7-v1.0".
 * - Decision threshold: 0.8300 (Class 0: MSS, Class 1: MSI).
 */

data class InputFingerprints(
    val format: String,
    val textTensorSha256: String,
    val visionPatchesSha256: String,
    val bundleSha256: String? = null,
    val patchCount: Int
)

data class EncoderProvenance(
    val encoderStatus: String = "UPSTREAM_PRECOMPUTED_UNVERIFIED",
    val visionEncoderReported: String = "Phikon-v2 (Reported upstream; unverified on-device)",
    val textEncoderReported: String = "ClinicalBERT (Reported upstream; unverified on-device)",
    val preprocessingManifestVersion: String = "phase7-v1.0",
    val verificationNote: String = "Upstream encoder processing and model weights cannot be cryptographically verified on-device. Input tensors are ingested as precomputed feature vectors."
)

data class ModelProvenance(
    val modelIdentifier: String = "fusion_model_fp32.onnx",
    val modelSha256: String = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73",
    val onnxRuntimeVersion: String = "1.18.0",
    val executionProvider: String = "CPUExecutionProvider",
    val isFrozenBaseline: Boolean = true
)

data class ThresholdConfig(
    val decisionThreshold: Double = 0.8300,
    val classZeroLabel: String = "MSS",
    val classOneLabel: String = "MSI",
    val decisionBoundaryRule: String = "P >= 0.8300 -> MSI (Class 1); P < 0.8300 -> MSS (Class 0)"
)

data class InferenceExecutionRecord(
    val status: String,
    val predictedClass: Int?,
    val predictedLabel: String?,
    val probability: Double?,
    val logit: Double?,
    val margin: Double?,
    val latencyMs: Double?,
    val executionTimestampUtc: String
)

data class EnvironmentProvenance(
    val applicationId: String = "org.crc.multimodal",
    val applicationVersion: String = "1.0.0",
    val buildNumber: Int = 1,
    val targetDevice: String = "Samsung Galaxy S24 (SM-S921E)",
    val targetOs: String = "Android 16 (API 36)",
    val networkIsolated: Boolean = true
)

data class IntegrityStatement(
    val checksumType: String = "SHA-256 Plaintext Digest",
    val manifestPayloadSha256: String,
    val isDigitalSignature: Boolean = false,
    val disclaimer: String = "Plaintext SHA-256 integrity checksum for detecting accidental or malicious modifications against a trusted digest. This is NOT an authenticated digital signature or public-key cryptographic certificate."
)

data class SpecimenProvenanceManifest(
    val manifestSchemaVersion: String = "1.0.0",
    val specimenIdentifier: String,
    val inputFingerprints: InputFingerprints,
    val encoderProvenance: EncoderProvenance = EncoderProvenance(),
    val modelProvenance: ModelProvenance = ModelProvenance(),
    val thresholdConfig: ThresholdConfig = ThresholdConfig(),
    val inferenceExecution: InferenceExecutionRecord,
    val environment: EnvironmentProvenance = EnvironmentProvenance(),
    val integrityStatement: IntegrityStatement
)

enum class AuditActionType(val displayName: String) {
    SPECIMEN_INGESTED("Specimen Ingested"),
    INFERENCE_EXECUTED("Inference Executed"),
    REVIEW_NOTE_RECORDED("Review Note Recorded"),
    REPORT_COMPILED("Research Report Compiled"),
    PROVENANCE_MANIFEST_EXPORTED("Provenance Manifest Exported"),
    SPECIMEN_DELETED("Specimen Record Deleted"),
    SPECIMEN_ARCHIVED("Specimen Record Archived"),
    SPECIMEN_RESTORED("Specimen Record Restored"),
    ALL_DATA_CLEARED("All Data Cleared"),
    REFERENCE_DATA_LOADED("Reference Data Loaded"),
    SECURITY_VALIDATION_FAILURE("Security Validation Failure")
}

data class AuditTrailRecord(
    val id: Long = 0,
    val eventTimestamp: Long = System.currentTimeMillis(),
    val actionType: AuditActionType,
    val specimenId: String?,
    val details: String,
    val sha256Fingerprint: String? = null,
    val performedBy: String = "Local Research User"
)

data class TraceabilityVerificationResult(
    val isTraceable: Boolean,
    val specimenMatched: Boolean,
    val modelHashMatched: Boolean,
    val manifestHashMatched: Boolean,
    val details: String
)
