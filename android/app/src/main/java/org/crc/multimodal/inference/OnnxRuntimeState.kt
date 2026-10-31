package org.crc.multimodal.inference

/**
 * Single authoritative lifecycle state for the on-device ONNX Runtime inference engine.
 *
 * Distinctly models library readiness, session creation, offline capability, and error conditions.
 */
sealed class OnnxRuntimeState {
    /**
     * Engine has not yet begun model loading or native session creation.
     */
    object NotStarted : OnnxRuntimeState() {
        override fun toString(): String = "NOT_STARTED"
    }

    /**
     * Model asset reading, SHA-256 validation, and native CPU session allocation are in progress.
     */
    object Initializing : OnnxRuntimeState() {
        override fun toString(): String = "INITIALIZING"
    }

    /**
     * Frozen FP32 model graph validated and native OrtSession is ready for sequential inference.
     */
    data class Ready(
        val assetPath: String = "models/fusion_model_fp32.onnx",
        val executionProvider: String = "CPUExecutionProvider",
        val threshold: Float = 0.83f,
        val initDurationMs: Long = 0L,
        val sha256Fingerprint: String = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
    ) : OnnxRuntimeState() {
        override fun toString(): String = "READY(path='$assetPath', initMs=$initDurationMs)"
    }

    /**
     * Initialization failed due to missing asset, corrupt bytes, or native ONNX exception.
     */
    data class Failed(
        val errorMessage: String,
        val throwable: Throwable? = null
    ) : OnnxRuntimeState() {
        override fun toString(): String = "FAILED(error='$errorMessage')"
    }

    val isReady: Boolean
        get() = this is Ready

    val isInitializing: Boolean
        get() = this is Initializing
}
