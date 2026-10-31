package org.crc.multimodal.inference

/**
 * Authentic multimodal feature bundle containing precomputed FP32 embeddings
 * for a single de-identified colorectal cancer specimen.
 *
 * Contract:
 * - [specimenId]: Unique de-identified specimen/patient identifier (e.g. "TCGA-A6-3807").
 * - [patchCount]: Number of histopathology slide patches (N >= 1).
 * - [visionDim]: Histopathology feature dimension (must be 1024, matching PHIKON-V2 output).
 * - [textDim]: Clinical text summary feature dimension (must be 768, matching ClinicalBERT output).
 * - [visionData]: Flattened FP32 array of shape [1, patchCount, 1024].
 * - [textData]: Flattened FP32 array of shape [1, 768].
 * - [visionSha256]: Hex-encoded SHA-256 checksum of raw vision float bytes.
 * - [textSha256]: Hex-encoded SHA-256 checksum of raw text float bytes.
 * - [schemaVersion]: Bundle specification version (default: 2).
 */
data class AuthenticFeatureBundle(
    val specimenId: String,
    val description: String = "",
    val patchCount: Int,
    val visionDim: Int = InputValidator.VISION_FEATURE_DIM,
    val textDim: Int = InputValidator.TEXT_FEATURE_DIM,
    val visionSha256: String = "",
    val textSha256: String = "",
    val visionData: FloatArray,
    val textData: FloatArray,
    val schemaVersion: Int = 2,
    val importTimestamp: Long = System.currentTimeMillis()
) {
    init {
        require(specimenId.isNotBlank()) { "Specimen identifier must not be blank." }
        require(patchCount > 0) { "Patch count must be strictly positive, got $patchCount." }
        require(visionDim == InputValidator.VISION_FEATURE_DIM) {
            "Vision dimension mismatch: expected ${InputValidator.VISION_FEATURE_DIM}, got $visionDim."
        }
        require(textDim == InputValidator.TEXT_FEATURE_DIM) {
            "Text dimension mismatch: expected ${InputValidator.TEXT_FEATURE_DIM}, got $textDim."
        }
        val expectedVisionFloats = patchCount * visionDim
        require(visionData.size == expectedVisionFloats) {
            "Vision data size mismatch: expected $expectedVisionFloats floats, got ${visionData.size}."
        }
        require(textData.size == textDim) {
            "Text data size mismatch: expected $textDim floats, got ${textData.size}."
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as AuthenticFeatureBundle

        if (specimenId != other.specimenId) return false
        if (patchCount != other.patchCount) return false
        if (visionDim != other.visionDim) return false
        if (textDim != other.textDim) return false
        if (schemaVersion != other.schemaVersion) return false
        if (visionSha256 != other.visionSha256) return false
        if (textSha256 != other.textSha256) return false
        if (!visionData.contentEquals(other.visionData)) return false
        if (!textData.contentEquals(other.textData)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = specimenId.hashCode()
        result = 31 * result + patchCount
        result = 31 * result + visionDim
        result = 31 * result + textDim
        result = 31 * result + schemaVersion
        result = 31 * result + visionSha256.hashCode()
        result = 31 * result + textSha256.hashCode()
        result = 31 * result + visionData.contentHashCode()
        result = 31 * result + textData.contentHashCode()
        return result
    }
}
