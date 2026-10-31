package org.crc.multimodal.evaluation

import org.crc.multimodal.analytics.SpecimenPredictionRecord
import kotlin.math.abs

/**
 * Deterministic cohort evaluation engine for Work Package 7.4.
 *
 * Computes confusion matrices, diagnostic metrics, and discrimination curves (ROC-AUC & PR-AUC)
 * by joining independently established ground-truth reference labels with persisted predictions.
 *
 * Threshold tau = 0.8300 is fixed and invariant.
 */
object CohortEvaluationEngine {

    const val FROZEN_DECISION_THRESHOLD = 0.8300

    /**
     * Evaluates a cohort by joining reference labels and model predictions.
     */
    fun evaluateCohort(
        referenceLabels: List<ReferenceLabel>,
        predictions: List<SpecimenPredictionRecord>,
        cohortId: String = "RESEARCH_COHORT"
    ): CohortEvaluationSummary {
        // Build map of predictions by normalized specimenId
        val predictionsBySpecimen = predictions.associateBy { it.specimenId.trim().lowercase() }
        val referenceBySpecimen = referenceLabels.associateBy { it.specimenId.trim().lowercase() }

        val matches = mutableListOf<SpecimenEvaluationMatch>()
        var unmatchedRefCount = 0

        for (ref in referenceLabels) {
            val key = ref.specimenId.trim().lowercase()
            val pred = predictionsBySpecimen[key]
            if (pred != null && pred.probability != null && pred.predictedClass != null) {
                val gtLabel = ref.groundTruthLabel
                val predLabel = pred.predictedClass // 1 = MSI, 0 = MSS
                val prob = pred.probability.toDouble()
                val predName = pred.predictedLabel ?: if (predLabel == 1) "MSI" else "MSS"

                val outcome = when {
                    gtLabel == 1 && predLabel == 1 -> EvaluationOutcome.TRUE_POSITIVE
                    gtLabel == 0 && predLabel == 0 -> EvaluationOutcome.TRUE_NEGATIVE
                    gtLabel == 0 && predLabel == 1 -> EvaluationOutcome.FALSE_POSITIVE
                    else -> EvaluationOutcome.FALSE_NEGATIVE
                }

                matches.add(
                    SpecimenEvaluationMatch(
                        specimenId = ref.specimenId,
                        groundTruthLabel = gtLabel,
                        groundTruthClass = ref.groundTruthClass,
                        predictedProbability = prob,
                        predictedClass = predLabel,
                        predictedLabel = predName,
                        decisionMargin = abs(prob - FROZEN_DECISION_THRESHOLD),
                        outcome = outcome,
                        source = ref.source
                    )
                )
            } else {
                unmatchedRefCount++
            }
        }

        val unmatchedPredCount = predictions.count { pred ->
            !referenceBySpecimen.containsKey(pred.specimenId.trim().lowercase())
        }

        // Confusion matrix counts
        val tp = matches.count { it.outcome == EvaluationOutcome.TRUE_POSITIVE }
        val tn = matches.count { it.outcome == EvaluationOutcome.TRUE_NEGATIVE }
        val fp = matches.count { it.outcome == EvaluationOutcome.FALSE_POSITIVE }
        val fn = matches.count { it.outcome == EvaluationOutcome.FALSE_NEGATIVE }
        val confusionMatrix = ConfusionMatrix(tp = tp, tn = tn, fp = fp, fn = fn)

        val total = confusionMatrix.total
        val actualPos = confusionMatrix.actualPositive
        val actualNeg = confusionMatrix.actualNegative
        val predPos = confusionMatrix.predictedPositive
        val predNeg = confusionMatrix.predictedNegative

        // Diagnostic Metrics
        val accuracy: MetricValue = if (total > 0) {
            MetricValue.Defined((tp + tn).toDouble() / total)
        } else {
            MetricValue.Undefined("No matched specimens available")
        }

        val precision: MetricValue = if (predPos > 0) {
            MetricValue.Defined(tp.toDouble() / predPos)
        } else {
            MetricValue.Undefined("No positive predictions (TP + FP = 0)")
        }

        val recall: MetricValue = if (actualPos > 0) {
            MetricValue.Defined(tp.toDouble() / actualPos)
        } else {
            MetricValue.Undefined("No positive cases in cohort (TP + FN = 0)")
        }

        val specificity: MetricValue = if (actualNeg > 0) {
            MetricValue.Defined(tn.toDouble() / actualNeg)
        } else {
            MetricValue.Undefined("No negative cases in cohort (TN + FP = 0)")
        }

        val balancedAccuracy: MetricValue = if (recall is MetricValue.Defined && specificity is MetricValue.Defined) {
            MetricValue.Defined((recall.value + specificity.value) / 2.0)
        } else {
            MetricValue.Undefined("Sensitivity or Specificity is undefined")
        }

        val npv: MetricValue = if (predNeg > 0) {
            MetricValue.Defined(tn.toDouble() / predNeg)
        } else {
            MetricValue.Undefined("No negative predictions (TN + FN = 0)")
        }

        val f1Score: MetricValue = if (precision is MetricValue.Defined && recall is MetricValue.Defined) {
            val p = precision.value
            val r = recall.value
            if ((p + r) > 0.0) {
                MetricValue.Defined(2.0 * p * r / (p + r))
            } else {
                MetricValue.Undefined("Precision + Recall is zero")
            }
        } else {
            MetricValue.Undefined("Precision or Recall is undefined")
        }

        // Discrimination Curves (ROC and PR)
        val (rocAuc, rocCurvePoints) = computeRocCurve(matches, actualPos, actualNeg)
        val (prAuc, prCurvePoints) = computePrCurve(matches, actualPos)

        return CohortEvaluationSummary(
            cohortId = cohortId,
            decisionThreshold = FROZEN_DECISION_THRESHOLD,
            totalReferenceLabels = referenceLabels.size,
            totalPredictionsAvailable = predictions.size,
            matchedCount = matches.size,
            unmatchedReferenceCount = unmatchedRefCount,
            unmatchedPredictionCount = unmatchedPredCount,
            confusionMatrix = confusionMatrix,
            accuracy = accuracy,
            precision = precision,
            recallSensitivity = recall,
            specificity = specificity,
            balancedAccuracy = balancedAccuracy,
            npv = npv,
            f1Score = f1Score,
            rocAuc = rocAuc,
            prAuc = prAuc,
            rocCurvePoints = rocCurvePoints,
            prCurvePoints = prCurvePoints,
            specimenMatches = matches
        )
    }

    /**
     * Computes the ROC curve points and trapezoidal AUC matching scikit-learn roc_auc_score.
     */
    private fun computeRocCurve(
        matches: List<SpecimenEvaluationMatch>,
        actualPos: Int,
        actualNeg: Int
    ): Pair<MetricValue, List<CurvePoint>> {
        if (actualPos == 0) {
            return Pair(MetricValue.Undefined("Cohort contains no positive cases (MSI=1)"), emptyList())
        }
        if (actualNeg == 0) {
            return Pair(MetricValue.Undefined("Cohort contains no negative cases (MSS=0)"), emptyList())
        }

        // Sort descending by predicted probability
        val sorted = matches.sortedByDescending { it.predictedProbability }

        // Find distinct threshold boundaries
        val distinctThresholdIndices = mutableListOf<Int>()
        for (i in 0 until sorted.size - 1) {
            if (sorted[i].predictedProbability != sorted[i + 1].predictedProbability) {
                distinctThresholdIndices.add(i)
            }
        }
        distinctThresholdIndices.add(sorted.size - 1)

        // Cumulative counts
        var cumTp = 0
        val tpsAtThresholds = mutableListOf<Int>()
        val fpsAtThresholds = mutableListOf<Int>()

        var currentIdx = 0
        for (thresholdIdx in distinctThresholdIndices) {
            while (currentIdx <= thresholdIdx) {
                if (sorted[currentIdx].groundTruthLabel == 1) {
                    cumTp++
                }
                currentIdx++
            }
            tpsAtThresholds.add(cumTp)
            fpsAtThresholds.add((thresholdIdx + 1) - cumTp)
        }

        val fprs = mutableListOf<Double>()
        val tprs = mutableListOf<Double>()

        // Origin (0, 0)
        fprs.add(0.0)
        tprs.add(0.0)

        for (k in tpsAtThresholds.indices) {
            fprs.add(fpsAtThresholds[k].toDouble() / actualNeg)
            tprs.add(tpsAtThresholds[k].toDouble() / actualPos)
        }

        // Trapezoidal integration
        var auc = 0.0
        val curvePoints = mutableListOf<CurvePoint>()
        curvePoints.add(CurvePoint(0.0, 0.0, null))

        for (i in 1 until fprs.size) {
            val dFpr = fprs[i] - fprs[i - 1]
            val avgTpr = (tprs[i] + tprs[i - 1]) / 2.0
            auc += dFpr * avgTpr

            val thresh = if (i - 1 < distinctThresholdIndices.size) {
                sorted[distinctThresholdIndices[i - 1]].predictedProbability
            } else null
            curvePoints.add(CurvePoint(fprs[i], tprs[i], thresh))
        }

        return Pair(MetricValue.Defined(auc), curvePoints)
    }

    /**
     * Computes the Precision-Recall curve and Average Precision (PR-AUC) matching scikit-learn average_precision_score.
     */
    private fun computePrCurve(
        matches: List<SpecimenEvaluationMatch>,
        actualPos: Int
    ): Pair<MetricValue, List<CurvePoint>> {
        if (actualPos == 0) {
            return Pair(MetricValue.Undefined("Cohort contains no positive cases (MSI=1)"), emptyList())
        }

        // Sort descending by predicted probability
        val sorted = matches.sortedByDescending { it.predictedProbability }

        // Find distinct threshold boundaries
        val distinctThresholdIndices = mutableListOf<Int>()
        for (i in 0 until sorted.size - 1) {
            if (sorted[i].predictedProbability != sorted[i + 1].predictedProbability) {
                distinctThresholdIndices.add(i)
            }
        }
        distinctThresholdIndices.add(sorted.size - 1)

        var cumTp = 0
        var prevRecall = 0.0
        var ap = 0.0
        val prPoints = mutableListOf<CurvePoint>()

        var currentIdx = 0
        for (thresholdIdx in distinctThresholdIndices) {
            while (currentIdx <= thresholdIdx) {
                if (sorted[currentIdx].groundTruthLabel == 1) {
                    cumTp++
                }
                currentIdx++
            }
            val cumFp = (thresholdIdx + 1) - cumTp
            val precision = cumTp.toDouble() / (cumTp + cumFp)
            val recall = cumTp.toDouble() / actualPos

            // Average Precision (step integration)
            val dRecall = recall - prevRecall
            ap += dRecall * precision
            prevRecall = recall

            val thresh = sorted[thresholdIdx].predictedProbability
            prPoints.add(CurvePoint(x = recall, y = precision, threshold = thresh))
        }

        return Pair(MetricValue.Defined(ap), prPoints)
    }

    /**
     * Filter and sort specimen matches for explorer UI.
     */
    fun filterAndSortMatches(
        matches: List<SpecimenEvaluationMatch>,
        filter: EvaluationFilterStatus,
        sortOption: EvaluationSortOption,
        searchQuery: String? = null
    ): List<SpecimenEvaluationMatch> {
        val filtered = matches.filter { item ->
            val matchesFilter = when (filter) {
                EvaluationFilterStatus.ALL -> true
                EvaluationFilterStatus.TP_ONLY -> item.outcome == EvaluationOutcome.TRUE_POSITIVE
                EvaluationFilterStatus.TN_ONLY -> item.outcome == EvaluationOutcome.TRUE_NEGATIVE
                EvaluationFilterStatus.FP_ONLY -> item.outcome == EvaluationOutcome.FALSE_POSITIVE
                EvaluationFilterStatus.FN_ONLY -> item.outcome == EvaluationOutcome.FALSE_NEGATIVE
                EvaluationFilterStatus.MISCLASSIFIED_ONLY -> !item.outcome.isCorrect
            }

            val matchesSearch = if (searchQuery.isNullOrBlank()) {
                true
            } else {
                val q = searchQuery.trim().lowercase()
                item.specimenId.lowercase().contains(q) ||
                        item.groundTruthClass.lowercase().contains(q) ||
                        item.predictedLabel.lowercase().contains(q) ||
                        item.outcome.code.lowercase().contains(q)
            }

            matchesFilter && matchesSearch
        }

        return when (sortOption) {
            EvaluationSortOption.SPECIMEN_ID -> filtered.sortedBy { it.specimenId }
            EvaluationSortOption.MARGIN_SMALLEST_FIRST -> filtered.sortedBy { it.decisionMargin }
            EvaluationSortOption.MARGIN_LARGEST_FIRST -> filtered.sortedByDescending { it.decisionMargin }
            EvaluationSortOption.PROBABILITY_DESCENDING -> filtered.sortedByDescending { it.predictedProbability }
            EvaluationSortOption.PROBABILITY_ASCENDING -> filtered.sortedBy { it.predictedProbability }
        }
    }
}
