package org.crc.multimodal.inference

/**
 * Encapsulates the frozen decision threshold and label mapping rules for colorectal cancer MSI/MSS detection.
 *
 * Invariants:
 * - Operating decision threshold: 0.83f
 * - Class 0 = MSS (Microsatellite Stable)
 * - Class 1 = MSI (Microsatellite Instability)
 * - Boundary semantics: probability >= threshold evaluates to Class 1 (MSI)
 */
object ThresholdLogic {
    const val OPERATING_THRESHOLD: Float = 0.83f
    const val CLASS_MSS: Int = 0
    const val CLASS_MSI: Int = 1
    const val LABEL_MSS: String = "MSS"
    const val LABEL_MSI: String = "MSI"

    /**
     * Maps a predicted probability to an integer class using the frozen 0.83 threshold.
     * Boundary rule: prob >= threshold -> MSI (1), prob < threshold -> MSS (0).
     */
    fun classify(probability: Float, threshold: Float = OPERATING_THRESHOLD): Int {
        require(!probability.isNaN()) { "Probability cannot be NaN" }
        require(!probability.isInfinite()) { "Probability cannot be Infinite" }
        require(probability in 0.0f..1.0f) { "Probability must be in [0.0, 1.0], got $probability" }
        return if (probability >= threshold) CLASS_MSI else CLASS_MSS
    }

    /**
     * Converts integer class index to human-readable label string.
     */
    fun labelFor(predictedClass: Int): String {
        return when (predictedClass) {
            CLASS_MSS -> LABEL_MSS
            CLASS_MSI -> LABEL_MSI
            else -> throw IllegalArgumentException("Invalid class index $predictedClass (must be 0 or 1)")
        }
    }

    /**
     * Calculates the signed distance to the threshold: (probability - threshold).
     * Positive indicates MSI prediction; negative indicates MSS prediction.
     */
    fun signedDistanceToThreshold(probability: Float, threshold: Float = OPERATING_THRESHOLD): Float {
        return probability - threshold
    }

    /**
     * Calculates absolute distance to threshold: |probability - threshold|.
     */
    fun marginToThreshold(probability: Float, threshold: Float = OPERATING_THRESHOLD): Float {
        return kotlin.math.abs(probability - threshold)
    }
}
