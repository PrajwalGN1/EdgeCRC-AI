# Cohort Evaluation Source Code Map & Architectural Component Inventory

**Role**: Senior ML Evaluation Engineer, Android/Kotlin Architect, Data-Pipeline Auditor  
**Audit Date**: October 28, 2026  
**Investigation Mode**: Strict Read-Only Architecture Audit  
**Module**: CRC Edge AI — Research Cohort Evaluation (`org.crc.multimodal.evaluation`)  

---

## 1. Executive Summary & Inventory Overview

This document provides a line-level, read-only architectural source map of the complete Cohort Evaluation subsystem within the CRC Multimodal Fusion Edge AI Android application. Every user interface element, data-loading pathway, matching algorithm, confusion matrix, diagnostic metric calculation, and discrimination curve has been traced directly to its authoritative Kotlin source files, XML layouts, and local SQLite database layers.

```
+--------------------------------------------------------------------------------------------------------+
|                                  COHORT EVALUATION SUBSYSTEM INVENTORY                                 |
+--------------------------------------------------------------------------------------------------------+
| UI Layer              | CohortEvaluationFragment.kt                                                    |
| Layout Resource       | fragment_cohort_evaluation.xml                                                 |
| View Adapter          | CohortSpecimenAdapter.kt                                                       |
| Mathematical Engine   | CohortEvaluationEngine.kt                                                      |
| Domain Models         | CohortEvaluationModels.kt                                                      |
| Label Importer        | ReferenceLabelImporter.kt                                                      |
| Workstation Storage   | SqliteWorkstationRepository.kt (WorkstationDbHelper.kt)                        |
| Unified Record Engine | ResearchAnalyticsEngine.kt (buildUnifiedRecords)                              |
| Batch Storage         | SqliteBatchRepository.kt (BatchDbHelper.kt)                                    |
| Static Assets         | assets/reference_labels/tcga_holdout_reference_labels.json                     |
+--------------------------------------------------------------------------------------------------------+
```

---

## 2. Source-Level Component Breakdown

### 2.1. User Interface & View Layer

#### 1. Fragment Implementation
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/ui/fragments/CohortEvaluationFragment.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/CohortEvaluationFragment.kt)
* **Class**: `org.crc.multimodal.ui.fragments.CohortEvaluationFragment : Fragment()`
* **Key Methods & Line References**:
  - `onCreate(savedInstanceState: Bundle?)` (Lines 49–52): Retrieves optional `ARG_COHORT_ID` argument (`cohortIdParam`).
  - `onCreateView(...)` (Lines 54–60): Inflates `R.layout.fragment_cohort_evaluation`.
  - `onViewCreated(view: View, ...)` (Lines 62–146): Initializes RecyclerView adapter, filter chips, search watcher, sort toggle, and click handlers.
  - `loadAssetReferenceLabels(rootView: View)` (Lines 148–178): Reads static asset `reference_labels/tcga_holdout_reference_labels.json`, invokes `ReferenceLabelImporter.parseJsonStream`, and inserts labels into `WorkstationRepository`.
  - `refreshEvaluation(rootView: View)` (Lines 180–190): Invokes `repo.evaluateCohort(cohortId = cohortIdParam, batchRepo = batchRepo)` and triggers view binding.
  - `bindSummaryViews(rootView: View, summary: CohortEvaluationSummary)` (Lines 192–302): Binds cohort headers, status badges, statistical sample size warnings, confusion matrix table cells, diagnostic metric cards, and ROC/PR curve points.
  - `bindMetric(...)` (Lines 304–328): Renders scalar metrics (`Defined` vs `Undefined`).
  - `bindCurveMetric(...)` (Lines 330–353): Renders discrimination curves (`Defined` vs `Undefined`/`Unavailable`).
  - `updateSpecimenList(rootView: View)` (Lines 355–367): Applies filtering, searching, and sorting via `CohortEvaluationEngine.filterAndSortMatches` and submits the list to the adapter.
  - `newInstance(cohortId: String?)` (Lines 372–381): Static companion factory.

#### 2. Layout Resource
* **File Path**: [`android/app/src/main/res/layout/fragment_cohort_evaluation.xml`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/res/layout/fragment_cohort_evaluation.xml)
* **Structure**:
  - Research disclaimer notice card (Lines 20–68).
  - Reference Cohort & statistical summary card (Lines 71–192):
    - `tvCohortTitle` (Line 94), `tvCohortMatchedStatus` (Line 104).
    - Badges: `tvRefLabelsBadge` (Line 121), `tvPredictionsAvailableBadge` (Line 130), `tvUnmatchedRefBadge` (Line 139).
    - Warning banner: `tvCohortSampleWarning` (Line 150).
    - Action buttons: `btnLoadHoldoutLabels` (Line 170), `btnClearReferenceLabels` (Line 183).
  - 2x2 Confusion Matrix card (Lines 195–350):
    - Horizontally scrollable TableLayout container (Lines 228–348).
    - Matrix cells: `tvMatrixTN` (Line 304), `tvMatrixFP` (Line 317), `tvMatrixFN` (Line 333), `tvMatrixTP` (Line 346).
    - Row & column totals: `tvMatrixTotalMSS`, `tvMatrixTotalMSI`, `tvMatrixPredMSS`, `tvMatrixPredMSI`, `tvMatrixGrandTotal`.
  - Diagnostic Performance Metrics card (Lines 353–640):
    - Accuracy (`tvMetricAccuracy`, `tvMetricAccuracyNote`).
    - Balanced Accuracy (`tvMetricBalancedAcc`, `tvMetricBalancedAccNote`).
    - Precision (`tvMetricPrecision`, `tvMetricPrecisionNote`).
    - Sensitivity / Recall (`tvMetricSensitivity`, `tvMetricSensitivityNote`).
    - Specificity (`tvMetricSpecificity`, `tvMetricSpecificityNote`).
    - F1 Score (`tvMetricF1`, `tvMetricF1Note`).
    - NPV (`tvMetricNPV`, `tvMetricNPVNote`).
  - Discrimination Capacity card (Lines 643–760):
    - ROC-AUC (`tvMetricRocAuc`, `tvRocPointsSummary`).
    - PR-AUC (`tvMetricPrAuc`, `tvPrPointsSummary`).
  - Specimen Evaluation Explorer card (Lines 763–1014):
    - Search field: `etCohortSearch`.
    - Filter chips: `chipGroupCohortFilters` (`chipFilterAll`, `chipFilterTP`, `chipFilterTN`, `chipFilterFP`, `chipFilterFN`, `chipFilterErrors`).
    - Sort button: `btnCohortSortToggle`.
    - Specimen RecyclerView: `rvCohortSpecimens`.
    - Empty state view: `tvCohortEmptyState`.

#### 3. Specimen List View Adapter
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/ui/adapters/CohortSpecimenAdapter.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/adapters/CohortSpecimenAdapter.kt)
* **Class**: `org.crc.multimodal.ui.adapters.CohortSpecimenAdapter : ListAdapter<SpecimenEvaluationMatch, ...>`
* **Layout Item**: `R.layout.item_cohort_specimen`
* **Features**: Binds specimen ID, ground truth class pill, prediction class pill, probability percentage, threshold margin, diagnostic outcome badge (TP, TN, FP, FN), and provenance source.

---

### 2.2. Mathematical Evaluation & Metrics Engine

#### 1. Evaluation Engine
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt)
* **Singleton Object**: `org.crc.multimodal.evaluation.CohortEvaluationEngine`
* **Constant**: `FROZEN_DECISION_THRESHOLD = 0.8300` (Line 16)
* **Key Functions & Line References**:
  - `evaluateCohort(referenceLabels, predictions, cohortId)` (Lines 21–159):
    - Specimen indexing & normalization (Lines 27–28).
    - Inner matching join (Lines 33–65).
    - Unmatched counting (Lines 67–69).
    - 2x2 confusion matrix tallying (Lines 72–76).
    - Accuracy calculation (Lines 85–89).
    - Precision calculation (Lines 91–95).
    - Recall / Sensitivity calculation (Lines 97–101).
    - Specificity calculation (Lines 103–107).
    - Balanced Accuracy calculation (Lines 109–113).
    - Negative Predictive Value calculation (Lines 115–119).
    - F1-Score calculation (Lines 121–131).
    - Curve dispatch (Lines 134–135).
    - Construction of `CohortEvaluationSummary` (Lines 137–158).
  - `computeRocCurve(matches, actualPos, actualNeg)` (Lines 164–234):
    - Pre-condition zero checks for `actualPos == 0` or `actualNeg == 0` (Lines 169–174).
    - Descending probability sorting (Line 177).
    - Distinct decision threshold partitioning (Lines 180–186).
    - Cumulative TP / FP scanning (Lines 188–203).
    - FPR / TPR coordinate generation (Lines 205–215).
    - Trapezoidal numerical integration (Lines 217–231).
  - `computePrCurve(matches, actualPos)` (Lines 239–286):
    - Pre-condition zero check for `actualPos == 0` (Lines 243–245).
    - Step-wise Average Precision (AP) calculation (Lines 264–283).
  - `filterAndSortMatches(matches, filter, sortOption, searchQuery)` (Lines 291–327):
    - Multi-criteria filtering by outcome, classification correctness, and substring search.
    - Sorting by specimen ID, margin (ascending/descending), or probability (descending/ascending).

#### 2. Domain Data Models
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationModels.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationModels.kt)
* **Models**:
  - `ReferenceLabel` (Lines 13–30): Data entity with immutable class-label consistency assertion.
  - `EvaluationOutcome` (Lines 35–40): Enum representing `TRUE_POSITIVE`, `TRUE_NEGATIVE`, `FALSE_POSITIVE`, `FALSE_NEGATIVE`.
  - `SpecimenEvaluationMatch` (Lines 45–55): Domain representation of a joined pair.
  - `ConfusionMatrix` (Lines 61–72): Matrix holding TP, TN, FP, FN and derived margins.
  - `MetricValue` (Lines 78–91): Sealed hierarchy representing either `Defined(val value: Double)` or `Undefined(val reason: String)`.
  - `CurvePoint` (Lines 96–100): 2D coordinate `(x, y, threshold)`.
  - `CohortEvaluationSummary` (Lines 105–128): Top-level summary aggregate.
  - `EvaluationFilterStatus` (Lines 133–140), `EvaluationSortOption` (Lines 145–151).

#### 3. Label Importer & Validator
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/evaluation/ReferenceLabelImporter.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/ReferenceLabelImporter.kt)
* **Singleton Object**: `org.crc.multimodal.evaluation.ReferenceLabelImporter`
* **Methods**:
  - `parseJson(jsonString: String): ImportResult` (Lines 22–83): Validates schema, specimen ID presence, integer label range (0 or 1), and class name string match.
  - `parseCsv(csvString: String, ...): ImportResult` (Lines 88–150): Parses CSV token streams.
  - `parseJsonStream(stream: InputStream): ImportResult` (Lines 155–160): Stream ingestion adapter.

---

### 2.3. Data Ingestion, Repositories, and Persistence

#### 1. Repository Contract
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/data/WorkstationRepository.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/data/WorkstationRepository.kt)
* **Key Cohort Evaluation Signatures**:
  - `getUnifiedPredictionRecords(batchRepo: BatchRepository?): List<SpecimenPredictionRecord>` (Line 44)
  - `insertReferenceLabels(labels: List<ReferenceLabel>): Int` (Line 72)
  - `getAllReferenceLabels(): List<ReferenceLabel>` (Line 73)
  - `clearReferenceLabels(): Int` (Line 74)
  - `evaluateCohort(cohortId: String?, batchRepo: BatchRepository?): CohortEvaluationSummary` (Lines 76–79)

#### 2. SQLite Workstation Persistence Implementation
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/data/SqliteWorkstationRepository.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/data/SqliteWorkstationRepository.kt)
* **Database File**: `/data/data/org.crc.multimodal/databases/crc_workstation.db`
* **Methods & Line References**:
  - `getUnifiedPredictionRecords(batchRepo: BatchRepository?)` (Lines 336–343): Fetches `allCases = getAllCases()`, `allRuns = getAllInferenceRuns()`, `batchRuns = batchRepo?.getAllBatchRuns() ?: emptyList()`, and delegates aggregation to `ResearchAnalyticsEngine.buildUnifiedRecords`.
  - `insertReferenceLabels(labels: List<ReferenceLabel>)` (Lines 777–802): Opens transaction, inserts rows into `reference_labels` with `CONFLICT_REPLACE`, records `REFERENCE_LABELS_LOADED` audit trail event.
  - `getAllReferenceLabels()` (Lines 804–814): Queries table `reference_labels`.
  - `clearReferenceLabels()` (Lines 816–821): Executes `db.delete("reference_labels", null, null)`.
  - `evaluateCohort(cohortId: String?, batchRepo: BatchRepository?)` (Lines 823–839): Orchestrates evaluation pipeline by loading labels, loading predictions from `getUnifiedPredictionRecords(batchRepo)`, and calling `CohortEvaluationEngine.evaluateCohort`.

#### 3. Prediction Unification Engine
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/analytics/ResearchAnalyticsEngine.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/analytics/ResearchAnalyticsEngine.kt)
* **Function**: `buildUnifiedRecords(cases: List<ResearchCase>, runs: List<InferenceRun>, batchRuns: List<BatchRun>): List<SpecimenPredictionRecord>` (Lines 29–132):
  - Section 1 (Lines 36–65): Converts `InferenceRun` instances from single-case workflow into `SpecimenPredictionRecord(origin = "SINGLE_CASE", ...)`.
  - Section 1b (Lines 68–92): Appends unanalyzed cases (DRAFT / READY_FOR_ANALYSIS) without runs.
  - Section 2 (Lines 94–129): Iterates over all `batchRuns` and appends all `batch.items` as `SpecimenPredictionRecord(origin = "BATCH", ...)`.

#### 4. Batch Ingestion Persistence Implementation
* **File Path**: [`android/app/src/main/java/org/crc/multimodal/batch/BatchRepository.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/batch/BatchRepository.kt)
* **Database File**: `/data/data/org.crc.multimodal/databases/crc_batch_runs.db`
* **Tables**: `batch_runs`, `batch_items`
* **Method**: `getAllBatchRuns(): List<BatchRun>` (Lines 240–280): Queries SQLite database and reconstructs complete batches with child `items`.

#### 5. Ground Truth Reference Asset
* **File Path**: [`android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json)
* **Payload**: Exactly 66 independently established CAP pathology ground-truth labels for TCGA holdout patients (50 MSS, 16 MSI).
* **Format**: Standard TCGA barcode patient prefix format (`TCGA-XX-XXXX`).
