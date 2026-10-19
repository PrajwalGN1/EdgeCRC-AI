package org.crc.multimodal.inference

/**
 * Encapsulates the output of a single patient inference execution.
 *
 * Class mapping:
 * - 0 = MSS (Microsatellite Stable)
 * - 1 = MSI (Microsatellite Instability)
 *
 * Threshold:
 * - Fixed operating point: 0.83f (probability >= 0.83f -> 1/MSI, else 0/MSS)
 */
data class InferenceResult(
    val caseId: String? = null,
    val predictedClass: Int,
    val predictedLabel: String,
    val probability: Float,
    val logit: Float,
    val threshold: Float = 0.83f,
    val numPatches: Int,
    val protocol: InferenceProtocol = InferenceProtocol.PROTOCOL_1_CAPPED,
    val attentionWeights: FloatArray? = null,
    val latencyMs: Long = 0L,
    val disclaimer: String = "RESEARCH USE ONLY: Multimodal fusion research output. Not approved for clinical diagnostics."
) {
    init {
        require(predictedClass in 0..1) {
            "predictedClass must be 0 (MSS) or 1 (MSI), got $predictedClass"
        }
        require(!probability.isNaN() && !probability.isInfinite() && probability in 0.0f..1.0f) {
            "probability must be a finite float within [0.0, 1.0], got $probability"
        }
        require(!logit.isNaN() && !logit.isInfinite()) {
            "logit must be a finite float, got $logit"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as InferenceResult

        if (caseId != other.caseId) return false
        if (predictedClass != other.predictedClass) return false
        if (predictedLabel != other.predictedLabel) return false
        if (probability != other.probability) return false
        if (logit != other.logit) return false
        if (threshold != other.threshold) return false
        if (numPatches != other.numPatches) return false
        if (protocol != other.protocol) return false
        if (attentionWeights != null) {
            if (other.attentionWeights == null) return false
            if (!attentionWeights.contentEquals(other.attentionWeights)) return false
        } else if (other.attentionWeights != null) return false
        if (latencyMs != other.latencyMs) return false
        if (disclaimer != other.disclaimer) return false

        return true
    }

    override fun hashCode(): Int {
        var result = caseId?.hashCode() ?: 0
        result = 31 * result + predictedClass
        result = 31 * result + predictedLabel.hashCode()
        result = 31 * result + probability.hashCode()
        result = 31 * result + logit.hashCode()
        result = 31 * result + threshold.hashCode()
        result = 31 * result + numPatches
        result = 31 * result + protocol.hashCode()
        result = 31 * result + (attentionWeights?.contentHashCode() ?: 0)
        result = 31 * result + latencyMs.hashCode()
        result = 31 * result + disclaimer.hashCode()
        return result
    }
}
