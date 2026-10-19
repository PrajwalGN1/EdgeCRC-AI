package org.crc.multimodal.inference

/**
 * Validates precomputed vision and text embedding tensors prior to model execution.
 *
 * Enforces:
 * - Expected dimensionality: vision is [1, N, 1024], text is [1, 768].
 * - Protocol 1 patch bounds: 1 <= N <= 1000 (when protocol is PROTOCOL_1_CAPPED).
 * - Protocol 2 uncapped patch bounds: 1 <= N <= maxSafePatches (when protocol is PROTOCOL_2_UNCAPPED).
 * - Finite float values (no NaN, +Infinity, or -Infinity).
 * - Buffer length exact matching.
 * - 64-bit integer overflow protection against DirectByteBuffer capacity limits.
 */
object InputValidator {
    const val VISION_FEATURE_DIM = 1024
    const val TEXT_FEATURE_DIM = 768
    const val PROTOCOL1_MAX_PATCHES = 1000

    /**
     * Validates a flattened vision embedding buffer representing [1, numPatches, 1024].
     *
     * @param visionData Flattened array of Float values.
     * @param numPatches Number of slide image patches (N).
     * @param protocol Explicit evaluation protocol (default: PROTOCOL_1_CAPPED).
     * @throws IllegalArgumentException if dimensions, sizes, or values are invalid.
     */
    fun validateVision(
        visionData: FloatArray,
        numPatches: Int,
        protocol: InferenceProtocol = InferenceProtocol.PROTOCOL_1_CAPPED
    ) {
        require(numPatches > 0) {
            "Patch count must be strictly positive (> 0), but received $numPatches"
        }

        when (protocol) {
            InferenceProtocol.PROTOCOL_1_CAPPED -> {
                require(numPatches <= PROTOCOL1_MAX_PATCHES) {
                    "Protocol 1 enforces a maximum cap of $PROTOCOL1_MAX_PATCHES patches, but received $numPatches"
                }
            }
            InferenceProtocol.PROTOCOL_2_UNCAPPED -> {
                // Protocol 2 is uncapped: all valid patches are preserved without truncation.
                // It is NOT constrained to PROTOCOL1_MAX_PATCHES.
            }
        }

        val expectedFloatsLong = numPatches.toLong() * VISION_FEATURE_DIM.toLong()
        require(expectedFloatsLong <= (Int.MAX_VALUE / Float.SIZE_BYTES).toLong()) {
            "Requested patch count $numPatches requires $expectedFloatsLong floats, which exceeds memory allocation capacity limit."
        }
        val expectedSize = expectedFloatsLong.toInt()
        require(visionData.size == expectedSize) {
            "Vision buffer size mismatch: expected $expectedSize floats ($numPatches patches * $VISION_FEATURE_DIM dim), got ${visionData.size}"
        }

        for (i in visionData.indices) {
            val v = visionData[i]
            if (v.isNaN()) {
                throw IllegalArgumentException("Vision embedding contains NaN at index $i")
            }
            if (v.isInfinite()) {
                throw IllegalArgumentException("Vision embedding contains Infinite value at index $i: $v")
            }
        }
    }

    /**
     * Backward-compatible overload for boolean enforceProtocol1Cap.
     */
    fun validateVision(
        visionData: FloatArray,
        numPatches: Int,
        enforceProtocol1Cap: Boolean
    ) {
        val protocol = if (enforceProtocol1Cap) {
            InferenceProtocol.PROTOCOL_1_CAPPED
        } else {
            InferenceProtocol.PROTOCOL_2_UNCAPPED
        }
        validateVision(visionData, numPatches, protocol)
    }

    /**
     * Validates a flattened text embedding buffer representing [1, 768].
     *
     * @param textData Flattened array of Float values.
     * @throws IllegalArgumentException if dimensions, sizes, or values are invalid.
     */
    fun validateText(textData: FloatArray) {
        require(textData.size == TEXT_FEATURE_DIM) {
            "Text buffer size mismatch: expected $TEXT_FEATURE_DIM floats, got ${textData.size}"
        }

        for (i in textData.indices) {
            val v = textData[i]
            if (v.isNaN()) {
                throw IllegalArgumentException("Text embedding contains NaN at index $i")
            }
            if (v.isInfinite()) {
                throw IllegalArgumentException("Text embedding contains Infinite value at index $i: $v")
            }
        }
    }
}
