package org.crc.multimodal.inference

/**
 * Exception thrown when ONNX model execution, tensor creation, or output parsing fails.
 */
class InferenceException(
    message: String,
    cause: Throwable? = null
) : RuntimeException(message, cause)

