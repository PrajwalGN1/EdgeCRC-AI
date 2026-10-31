package org.crc.multimodal.evaluation

/**
 * Domain models for Work Package 7.4: Research Cohort Evaluation.
 *
 * Implements offline evaluation against independently established reference ground-truth labels.
 * Ground-truth labels are strictly decoupled from model predictions.
 */

/**
 * Independently established reference ground-truth label.
 */
data class ReferenceLabel(
    val specimenId: String,
    val groundTruthLabel: Int, // 0 = MSS, 1 = MSI
    val groundTruthClass: String, // "MSS" or "MSI"
    val cohortId: String = "TCGA_HOLDOUT_66",
    val source: String? = null,
    val importedAt: Long = System.currentTimeMillis()
) {
    init {
        require(groundTruthLabel == 0 || groundTruthLabel == 1) {
            "groundTruthLabel must be 0 (MSS) or 1 (MSI), got $groundTruthLabel"
        }
        val expectedClass = if (groundTruthLabel == 1) "MSI" else "MSS"
        require(groundTruthClass.equals(expectedClass, ignoreCase = true)) {
            "groundTruthClass '$groundTruthClass' does not match label $groundTruthLabel (expected '$expectedClass')"
        }
    }
}

/**
 * Diagnostic outcome category for binary classification.
 */
enum class EvaluationOutcome(val code: String, val displayName: String, val isCorrect: Boolean) {
    TRUE_POSITIVE("TP", "True Positive (MSI)", true),
    TRUE_NEGATIVE("TN", "True Negative (MSS)", true),
    FALSE_POSITIVE("FP", "False Positive (Type I)", false),
    FALSE_NEGATIVE("FN", "False Negative (Type II)", false)
}

/**
 * Individual matched specimen comparing ground truth with model prediction.
 */
data class SpecimenEvaluationMatch(
    val specimenId: String,
    val groundTruthLabel: Int,
    val groundTruthClass: String,
    val predictedProbability: Double,
    val predictedClass: Int,
    val predictedLabel: String,
    val decisionMargin: Double, // |P - 0.83|
    val outcome: EvaluationOutcome,
    val source: String? = null
)

/**
 * 2x2 Confusion Matrix for binary classification.
 * Convention: Positive Class = MSI (1), Negative Class = MSS (0).
 */
data class ConfusionMatrix(
    val tp: Int,
    val tn: Int,
    val fp: Int,
    val fn: Int
) {
    val total: Int get() = tp + tn + fp + fn
    val actualPositive: Int get() = tp + fn // Total actual MSI
    val actualNegative: Int get() = tn + fp // Total actual MSS
    val predictedPositive: Int get() = tp + fp // Total predicted MSI
    val predictedNegative: Int get() = tn + fn // Total predicted MSS
}

/**
 * Represents a metric calculation that may either be defined or mathematically undefined
 * due to cohort composition (e.g. division by zero).
 */
sealed class MetricValue {
    data class Defined(val value: Double) : MetricValue() {
        fun format(decimals: Int = 4): String = String.format(java.util.Locale.US, "%.${decimals}f", value)
        fun formatPercentage(decimals: Int = 2): String = String.format(java.util.Locale.US, "%.${decimals}f%%", value * 100.0)
        override fun toString(): String = format(4)
    }

    data class Undefined(val reason: String) : MetricValue() {
        override fun toString(): String = "Not defined: $reason"
    }

    val isDefined: Boolean get() = this is Defined
    val doubleValueOrNull: Double? get() = (this as? Defined)?.value
}

/**
 * Discrete coordinate point for ROC or PR curve visualization.
 */
data class CurvePoint(
    val x: Double, // FPR for ROC, Recall for PR
    val y: Double, // TPR for ROC, Precision for PR
    val threshold: Double? = null
)

/**
 * Complete cohort evaluation summary aggregating diagnostic performance and discrimination.
 */
data class CohortEvaluationSummary(
    val cohortId: String,
    val evaluationTimestamp: Long = System.currentTimeMillis(),
    val decisionThreshold: Double = 0.8300,
    val positiveClassSemantics: String = "MSI = Positive (Class 1), MSS = Negative (Class 0)",
    val totalReferenceLabels: Int,
    val totalPredictionsAvailable: Int,
    val matchedCount: Int,
    val unmatchedReferenceCount: Int,
    val unmatchedPredictionCount: Int,
    val confusionMatrix: ConfusionMatrix,
    val accuracy: MetricValue,
    val precision: MetricValue, // Positive Predictive Value (PPV)
    val recallSensitivity: MetricValue, // Sensitivity
    val specificity: MetricValue, // True Negative Rate
    val balancedAccuracy: MetricValue,
    val npv: MetricValue, // Negative Predictive Value
    val f1Score: MetricValue, // Harmonic mean of Precision and Recall
    val rocAuc: MetricValue,
    val prAuc: MetricValue,
    val rocCurvePoints: List<CurvePoint> = emptyList(),
    val prCurvePoints: List<CurvePoint> = emptyList(),
    val specimenMatches: List<SpecimenEvaluationMatch> = emptyList()
)

/**
 * Specimen explorer filter criteria.
 */
enum class EvaluationFilterStatus {
    ALL,
    TP_ONLY,
    TN_ONLY,
    FP_ONLY,
    FN_ONLY,
    MISCLASSIFIED_ONLY
}

/**
 * Specimen explorer sorting options.
 */
enum class EvaluationSortOption {
    SPECIMEN_ID,
    MARGIN_SMALLEST_FIRST,
    MARGIN_LARGEST_FIRST,
    PROBABILITY_DESCENDING,
    PROBABILITY_ASCENDING
}
