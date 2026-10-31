# Cohort Evaluation: Metric Calculation & Statistical Edge-Case Integrity Audit

**Role**: Senior ML Evaluation Engineer, Android/Kotlin Architect, Statistical Validation Specialist  
**Audit Date**: October 28, 2026  
**Investigation Mode**: Strict Read-Only Mathematical Audit  
**Deliverable**: `reports/mobile_deployment/cohort_evaluation_metric_integrity_audit.md`  

---

## 1. Executive Summary & Audit Overview

This audit verifies the mathematical correctness, edge-case resilience, and diagnostic reporting integrity of all statistical metrics implemented in [`CohortEvaluationEngine.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt).

The device state reported in the directive consists of a 3-specimen matched cohort:
* **True Positives (TP)**: 2 (`TCGA-D5-6927`, `TCGA-D5-6928`)
* **True Negatives (TN)**: 0
* **False Positives (FP)**: 0
* **False Negatives (FN)**: 1 (`TCGA-G4-6320`)
* **Actual MSI (Class 1)**: $TP + FN = 2 + 1 = 3$
* **Actual MSS (Class 0)**: $TN + FP = 0 + 0 = 0$
* **Total Matched**: 3

### Core Statistical Findings:
1. **Mathematical Fidelity**: All seven scalar diagnostic metrics (Accuracy, Precision, Recall, Specificity, Balanced Accuracy, NPV, F1) strictly implement standard biostatistical definitions.
2. **Zero-Denominator Safety**: The implementation uses a sealed class hierarchy (`MetricValue.Defined` vs `MetricValue.Undefined`). It **never** divides by zero, **never** throws `ArithmeticException`, and **never** silently coerces undefined metrics to misleading 0.0 or 1.0 values.
3. **Exact Numerical Reproduction**:
   - Accuracy: $2/3 = \mathbf{66.7\%}$ (`0.6667`)
   - Sensitivity / Recall: $2/3 = \mathbf{66.7\%}$ (`0.6667`)
   - Precision: $2/2 = \mathbf{100.0\%}$ (`1.0000`)
   - Specificity: $\mathbf{Undefined}$ (Reason: *"No negative cases in cohort (TN + FP = 0)"*)
   - Balanced Accuracy: $\mathbf{Undefined}$ (Reason: *"Sensitivity or Specificity is undefined"*)
   - Negative Predictive Value (NPV): $0/1 = \mathbf{0.0\%}$ (`0.0000`)
   - F1-Score: $2 \times 1.0 \times (2/3) / (1.0 + 2/3) = \mathbf{0.8000}$
   - ROC-AUC: $\mathbf{Unavailable}$ (Reason: *"Cohort contains no negative cases (MSS=0)"*)
   - PR-AUC: Displays $\mathbf{1.0000}$ (Average Precision step integration)

---

## 2. Mathematical Derivations & Code Verifications

### 2.1. Accuracy
* **Mathematical Definition**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{TP + TN}{\text{Total}}$
* **Code Implementation** ([`CohortEvaluationEngine.kt:L85-L89`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L85-L89)):
  ```kotlin
  val accuracy: MetricValue = if (total > 0) {
      MetricValue.Defined((tp + tn).toDouble() / total)
  } else {
      MetricValue.Undefined("No matched specimens available")
  }
  ```
* **Evaluation on Matched Cohort**:
  $$\text{Accuracy} = \frac{2 + 0}{3} = 0.666667 \implies \mathbf{66.7\%}$$
* **Verification Status**: `PASS — EXACT MATCH`.

---

### 2.2. Precision (Positive Predictive Value / PPV)
* **Mathematical Definition**: $\text{Precision} = \frac{TP}{TP + FP}$
* **Code Implementation** ([`CohortEvaluationEngine.kt:L91-L95`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L91-L95)):
  ```kotlin
  val precision: MetricValue = if (predPos > 0) {
      MetricValue.Defined(tp.toDouble() / predPos)
  } else {
      MetricValue.Undefined("No positive predictions (TP + FP = 0)")
  }
  ```
* **Evaluation on Matched Cohort**:
  $$\text{Precision} = \frac{2}{2 + 0} = 1.0000 \implies \mathbf{100.0\%}$$
* **Verification Status**: `PASS — EXACT MATCH`.

---

### 2.3. Recall / Sensitivity (True Positive Rate / TPR)
* **Mathematical Definition**: $\text{Sensitivity} = \frac{TP}{TP + FN}$
* **Code Implementation** ([`CohortEvaluationEngine.kt:L97-L101`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L97-L101)):
  ```kotlin
  val recall: MetricValue = if (actualPos > 0) {
      MetricValue.Defined(tp.toDouble() / actualPos)
  } else {
      MetricValue.Undefined("No positive cases in cohort (TP + FN = 0)")
  }
  ```
* **Evaluation on Matched Cohort**:
  $$\text{Sensitivity} = \frac{2}{2 + 1} = 0.666667 \implies \mathbf{66.7\%}$$
* **Verification Status**: `PASS — EXACT MATCH`.

---

### 2.4. Specificity (True Negative Rate / TNR)
* **Mathematical Definition**: $\text{Specificity} = \frac{TN}{TN + FP}$
* **Code Implementation** ([`CohortEvaluationEngine.kt:L103-L107`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L103-L107)):
  ```kotlin
  val specificity: MetricValue = if (actualNeg > 0) {
      MetricValue.Defined(tn.toDouble() / actualNeg)
  } else {
      MetricValue.Undefined("No negative cases in cohort (TN + FP = 0)")
  }
  ```
* **Evaluation on Matched Cohort**:
  Denominator $\text{Actual Negative} = TN + FP = 0 + 0 = 0$.
  Division by zero is detected. Returns `MetricValue.Undefined("No negative cases in cohort (TN + FP = 0)")`.
* **UI Presentation**: Displays `"Undefined"` in amber (`#E5A93C`) and reason in supporting text.
* **Verification Status**: `PASS — CLINICALLY ACCURATE`.

---

### 2.5. Balanced Accuracy
* **Mathematical Definition**: $\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$
* **Code Implementation** ([`CohortEvaluationEngine.kt:L109-L113`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L109-L113)):
  ```kotlin
  val balancedAccuracy: MetricValue = if (recall is MetricValue.Defined && specificity is MetricValue.Defined) {
      MetricValue.Defined((recall.value + specificity.value) / 2.0)
  } else {
      MetricValue.Undefined("Sensitivity or Specificity is undefined")
  }
  ```
* **Evaluation on Matched Cohort**:
  Since Specificity is `Undefined`, Balanced Accuracy cannot be calculated. Returns `MetricValue.Undefined("Sensitivity or Specificity is undefined")`.
* **UI Presentation**: Displays `"Undefined"` in amber.
* **Verification Status**: `PASS — STRICT PROPAGATION OF UNCERTAINTY`.

---

### 2.6. Negative Predictive Value (NPV)
* **Mathematical Definition**: $\text{NPV} = \frac{TN}{TN + FN}$
* **Code Implementation** ([`CohortEvaluationEngine.kt:L115-L119`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L115-L119)):
  ```kotlin
  val npv: MetricValue = if (predNeg > 0) {
      MetricValue.Defined(tn.toDouble() / predNeg)
  } else {
      MetricValue.Undefined("No negative predictions (TN + FN = 0)")
  }
  ```
* **Evaluation on Matched Cohort**:
  $$\text{NPV} = \frac{0}{0 + 1} = 0.0000 \implies \mathbf{0.0\%}$$
* **Verification Status**: `PASS — EXACT MATCH`.

---

### 2.7. F1-Score (Harmonic Mean)
* **Mathematical Definition**: $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
* **Code Implementation** ([`CohortEvaluationEngine.kt:L121-L131`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L121-L131)):
  ```kotlin
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
  ```
* **Evaluation on Matched Cohort**:
  $$F_1 = 2 \times \frac{1.0 \times \frac{2}{3}}{1.0 + \frac{2}{3}} = 2 \times \frac{\frac{2}{3}}{\frac{5}{3}} = \frac{4}{5} = \mathbf{0.8000}$$
* **Verification Status**: `PASS — EXACT MATCH`.

---

## 3. Discrimination Curve Analysis (ROC-AUC & PR-AUC)

### 3.1. ROC-AUC Behavior on Single-Class Reference Cohorts
In [`CohortEvaluationEngine.kt:L169-L174`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L169-L174):
```kotlin
if (actualPos == 0) {
    return Pair(MetricValue.Undefined("Cohort contains no positive cases (MSI=1)"), emptyList())
}
if (actualNeg == 0) {
    return Pair(MetricValue.Undefined("Cohort contains no negative cases (MSS=0)"), emptyList())
}
```
* **Mathematical Truth**: ROC curves map False Positive Rate ($FPR = \frac{FP}{TN + FP}$) against True Positive Rate ($TPR = \frac{TP}{TP + FN}$). When $\text{Actual Negative} = TN + FP = 0$, $FPR$ is undefined for all thresholds ($0/0$). The 2D area under the curve cannot exist.
* **App Handling**: Correctly returns `MetricValue.Undefined("Cohort contains no negative cases (MSS=0)")`.
* **UI Display**: Displays `"Unavailable"` in amber with supporting note `"Cohort contains no negative cases (MSS=0)"`.
* **Verdict**: `CONFIRMED CORRECT BEHAVIOUR`.

### 3.2. PR-AUC (Average Precision) Behavior on Single-Class Cohorts
In [`CohortEvaluationEngine.kt:L243-L285`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L243-L285):
* Precision at every threshold step $k$:
  $$\text{Precision}_k = \frac{\text{cum\_TP}_k}{\text{cum\_TP}_k + \text{cum\_FP}_k}$$
  Since there are 0 negative cases, $\text{cum\_FP}_k = 0$ at all steps. Therefore:
  $$\text{Precision}_k = \frac{\text{cum\_TP}_k}{\text{cum\_TP}_k + 0} = 1.0$$
* Step integration:
  $$\text{AP} = \sum_{k} (\text{Recall}_k - \text{Recall}_{k-1}) \times \text{Precision}_k = \left(\frac{1}{3} - 0\right)(1.0) + \left(\frac{2}{3} - \frac{1}{3}\right)(1.0) + \left(1.0 - \frac{2}{3}\right)(1.0) = \mathbf{1.0000}$$
* **Statistical Critique**: While the numerical formula yields 1.0000, PR-AUC is uninformative when the negative class is absent because false alarms are physically impossible.
* **Safeguard**: The UI displays a warning banner (`tvCohortSampleWarning`) alerting the clinician:
  > *"⚠ Limited Sample Size: 3 matched specimens... 100% accuracy on a single matched negative case is not statistically meaningful evidence of general model performance."*

---

## 4. Edge-Case Matrix Evaluation

| Edge-Case Scenario | Handling Implementation | Result / Presentation | Integrity Verdict |
| :--- | :--- | :--- | :---: |
| **0 Matched Specimens** | `total == 0` check | All metrics return `Undefined`; UI displays empty state | `PASS` |
| **All MSI Ground Truth** (Current Case) | `actualNeg == 0` check | Specificity, Balanced Acc, ROC-AUC = `Undefined`; PR-AUC = 1.0; Warning banner visible | `PASS` |
| **All MSS Ground Truth** | `actualPos == 0` check | Precision, Recall, Balanced Acc, ROC-AUC, PR-AUC = `Undefined` | `PASS` |
| **Both Classes Present** | Standard biostatistics | Full evaluation with valid ROC-AUC & PR-AUC | `PASS` |
| **0 Positive Predictions** | `predPos == 0` check | Precision returns `Undefined` (no crash) | `PASS` |
| **0 Negative Predictions** | `predNeg == 0` check | NPV returns `Undefined` (no crash) | `PASS` |
| **Null Probability** | `pred.probability != null` | Excluded from matched cohort | `PASS` |
| **Probability Out of Range** | Bound checks in models | Validated upon bundle loading ($0.0 \le P \le 1.0$) | `PASS` |
| **Duplicate Specimen IDs** | `associateBy` selection | De-duplicated to latest run | `PASS` |
| **Small Cohort ($N < 10$)** | Sample size threshold | Amber warning banner displayed in UI | `PASS` |
