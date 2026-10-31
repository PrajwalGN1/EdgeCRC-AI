package org.crc.multimodal.data

import org.crc.multimodal.inference.InferenceProtocol

/**
 * Domain entity representing an individual execution of the multimodal fusion model.
 */
data class InferenceRun(
    val id: Long = 0L,
    val caseId: Long,
    val modelIdentifier: String = "fusion_model_fp32.onnx",
    val inferenceArchitecture: String = "LOCAL_ONNX_FP32",
    val protocol: InferenceProtocol? = InferenceProtocol.PROTOCOL_1_CAPPED,
    val predictedClass: Int,
    val predictedLabel: String,
    val probability: Float,
    val logit: Float,
    val threshold: Float = 0.83f,
    val margin: Float = probability - threshold,
    val latencyMs: Double,
    val numPatches: Int,
    val diagnosticsSummary: String? = null,
    val executionTimestamp: Long = System.currentTimeMillis(),
    val status: String = "SUCCESS"
) {
    init {
        require(predictedClass in 0..1) { "predictedClass must be 0 or 1." }
        require(!probability.isNaN() && probability in 0.0f..1.0f) { "probability must be in [0.0, 1.0]." }
    }
}
