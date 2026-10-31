# Batch Specimen Analysis to Cohort Evaluation: End-to-End Dataflow & Integration Architecture

**Role**: Senior ML Evaluation Engineer, Android/Kotlin Architect, Data-Pipeline Auditor  
**Audit Date**: October 28, 2026  
**Investigation Mode**: Strict Read-Only Architecture Audit  
**Deliverable**: `reports/mobile_deployment/batch_to_cohort_evaluation_dataflow.md`  

---

## 1. Executive Summary & Integration Architecture

This report details the architectural data pipeline connecting **Batch Specimen Analysis** and **Cohort Evaluation**. It answers whether, where, and how predictions generated through batch execution flow into the cohort benchmark engine.

### Architectural Classification
The actual implementation uses a **Federated Hybrid Unification Pipeline**:

```
+----------------------------------------------------------------------------------------------------+
|                                    FEDERATED PREDICTION PIPELINE                                   |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   [ Individual Analysis ]               [ Batch Specimen Analysis ]                                |
|             |                                        |                                             |
|             v                                        v                                             |
|   SqliteWorkstationRepo                    SqliteBatchRepository                                   |
|   (crc_workstation.db)                     (crc_batch_runs.db)                                     |
|     - cases                                  - batch_runs                                          |
|     - inference_runs                         - batch_items                                         |
|             |                                        |                                             |
|             |        Batch Bridge Sync               |                                             |
|             |<---------------------------------------+                                             |
|             |   (BatchCoordinator.syncBatchToCases)  |                                             |
|             |                                        |                                             |
|             +-------------------+--------------------+                                             |
|                                 |                                                                  |
|                                 v                                                                  |
|                   getUnifiedPredictionRecords(...)                                                 |
|                   ResearchAnalyticsEngine.buildUnifiedRecords                                      |
|                                 |                                                                  |
|                                 v                                                                  |
|                      Unified Prediction Records (List)                                             |
|                                 |                                                                  |
|   [ Reference Labels ]          |                                                                  |
|   assets/tcga_holdout.json      |                                                                  |
|   (reference_labels table)      |                                                                  |
|             |                   |                                                                  |
|             +--------->[ CohortEvaluationEngine ]<----------+                                      |
|                                 |                                                                  |
|                                 v                                                                  |
|                      CohortEvaluationSummary                                                       |
|                                 |                                                                  |
|                                 v                                                                  |
|                      CohortEvaluationFragment                                                      |
+----------------------------------------------------------------------------------------------------+
```

Cohort Evaluation does **not** rely solely on single-case records. Through `getUnifiedPredictionRecords(batchRepo)`, it explicitly reads from **both** `crc_workstation.db` and `crc_batch_runs.db`, merging them in memory and de-duplicating by normalized specimen ID. Furthermore, eligible batch runs are synchronized to the shared Cases registry via `BatchCoordinator.syncBatchToCases`.

---

## 2. End-to-End Pipeline Tracing

### Stage 1: Bundle Selection & Import
* **Component**: [`AuthenticBundleLoader.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt) / [`BatchAnalysisFragment.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/BatchAnalysisFragment.kt)
* **Action**: User imports `.crcbundle` zip archives.
* **Storage**: Initial metadata and files are registered into table `batch_runs` and `batch_items` in `crc_batch_runs.db`.
* **State**: Items enter status `PENDING` or `VALIDATED`.

### Stage 2: Batch Inference Execution
* **Component**: [`BatchCoordinator.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/batch/BatchCoordinator.kt)
* **Model**: ONNX Runtime engine running `fusion_model_fp32.onnx`.
* **Contract**: Input shapes Vision `[1, N, 1024]`, Text `[1, 768]`, fixed operating threshold $\tau = 0.8300$.
* **Output**: Continuous probability $P$, predicted class (1 = MSI if $P \ge 0.83$, 0 = MSS if $P < 0.83$), latency in ms, patch count.

### Stage 3: Result Persistence in Batch Database
* **Component**: [`SqliteBatchRepository.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/batch/BatchRepository.kt)
* **Destination**: Database `crc_batch_runs.db`, table `batch_items`.
* **Data Stored**: `probability`, `predicted_class`, `predicted_label`, `status = COMPLETED`, `completed_timestamp`.
* **Failure Handling**: Failed items are updated to `status = FAILED` or `REJECTED` with `probability = null`.

### Stage 4: Synchronization to Shared Cases Registry
* **Component**: [`BatchCoordinator.syncBatchToCases(...)`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/batch/BatchCoordinator.kt)
* **Destination**: Database `crc_workstation.db`, tables `cases`, `specimen_inputs`, `inference_runs`.
* **Policy**:
  - Checks if a case with `caseIdentifier == item.specimenId` already exists.
  - If existing case is already `ANALYZED`, preserves it (overwrite protection).
  - If new, creates a case record, attaches `specimen_inputs`, records the completed `inference_runs` entry, and logs `INFERENCE_EXECUTED` in `audit_trail`.

### Stage 5: Prediction Loading for Cohort Evaluation
* **Component**: [`SqliteWorkstationRepository.evaluateCohort(...)`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/data/SqliteWorkstationRepository.kt#L823-L839)
* **Method**: Calls `getUnifiedPredictionRecords(batchRepo)`:
  ```kotlin
  val allCases = getAllCases()
  val allRuns = getAllInferenceRuns()
  val batchRuns = batchRepo?.getAllBatchRuns() ?: emptyList()
  return ResearchAnalyticsEngine.buildUnifiedRecords(allCases, allRuns, batchRuns)
  ```
* **Unification Logic**:
  - `ResearchAnalyticsEngine.buildUnifiedRecords` maps all `inference_runs` from `crc_workstation.db` to `SpecimenPredictionRecord(origin = "SINGLE_CASE", ...)`.
  - Appends unanalyzed draft cases with `probability = null`.
  - Iterates through all `batchRuns` from `crc_batch_runs.db` and appends all `batch.items` as `SpecimenPredictionRecord(origin = "BATCH", ...)`.
* **Total Result**: Every batch item in `crc_batch_runs.db` is included in the unified prediction stream, whether or not it was synced to `crc_workstation.db`.

### Stage 6: Evaluation Join against Reference Labels
* **Component**: [`CohortEvaluationEngine.evaluateCohort(...)`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L21-L159)
* **Join Operation**:
  - Maps predictions: `predictionsBySpecimen = predictions.associateBy { it.specimenId.trim().lowercase() }`.
  - Iterates over reference labels from `reference_labels` table.
  - If `pred != null && pred.probability != null && pred.predictedClass != null`, records a `SpecimenEvaluationMatch`.
  - Otherwise, increments `unmatchedReferenceCount`.

---

## 3. Explicit Answers to Directive Questions

### 1. Where are batch predictions stored?
Batch predictions are stored primarily in `crc_batch_runs.db` (table `batch_items`), with completed runs also synchronized to `crc_workstation.db` (tables `cases` and `inference_runs`).

### 2. Where are individual-analysis predictions stored?
Individual single-case predictions are stored exclusively in `crc_workstation.db` (tables `cases` and `inference_runs`).

### 3. Do both use the same repository or database tables?
No. Individual analysis uses `WorkstationRepository` (`crc_workstation.db`), while batch analysis uses `BatchRepository` (`crc_batch_runs.db`). They are bridged via `BatchCoordinator.syncBatchToCases` and unified at read-time via `getUnifiedPredictionRecords(batchRepo)`.

### 4. Does successful batch inference update the shared Cases registry?
**Yes.** `BatchCoordinator.persistCompletedSpecimenToCases` and `BatchCoordinator.syncBatchToCases` insert records into `cases` and `inference_runs` with `INFERENCE_EXECUTED` audit logging upon completion.

### 5. Does Cohort Evaluation read from the shared Cases registry?
**Yes.** It reads all cases via `workstationRepo.getAllCases()` and all runs via `workstationRepo.getAllInferenceRuns()`.

### 6. Does Cohort Evaluation read from a separate prediction registry or file?
Cohort Evaluation reads from **both** the shared Cases registry and the batch database directly via `batchRepo.getAllBatchRuns()`. It does not read from any isolated JSON prediction cache.

### 7. Are rejected and failed specimens excluded correctly?
**Yes.** `CohortEvaluationEngine.kt` checks:
```kotlin
if (pred != null && pred.probability != null && pred.predictedClass != null)
```
Rejected or failed batch specimens have `probability = null` and are excluded from the matched set.

### 8. Are archived specimens included or excluded?
In the current implementation of `ResearchAnalyticsEngine.buildUnifiedRecords` (lines 95–129), `batch.items` are iterated without checking `item.isArchived`. Thus, archived batch items with completed predictions are currently passed to the unification pool unless filtered prior to evaluation.

### 9. Do repeated batch runs create duplicate prediction records?
In `crc_batch_runs.db`, each batch run stores its own rows in `batch_items`. This produces multiple historical records for the same specimen ID (e.g. `TCGA-AZ-4315` has 8 rows). In `CohortEvaluationEngine`, `predictions.associateBy { it.specimenId.trim().lowercase() }` de-duplicates them by retaining the **last** record in the list.

### 10. Can a batch prediction be found in Cohort Evaluation without manual re-entry?
**Yes, immediately.** As soon as a batch inference completes, it is visible in Cohort Evaluation upon fragment creation or refresh.

### 11. Does `LOAD TCGA LABELS` only load reference labels or also trigger prediction loading?
`LOAD TCGA LABELS` (`loadAssetReferenceLabels`) parses `tcga_holdout_reference_labels.json` and inserts 66 rows into `reference_labels` in `crc_workstation.db`. It does not load or generate predictions. Once labels are inserted, it calls `refreshEvaluation()` which queries existing predictions.

### 12. Does `CLEAR` clear only evaluation state or delete persistent data?
`CLEAR` (`btnClearReferenceLabels`) calls `repo.clearReferenceLabels()`, which executes:
```kotlin
db.delete("reference_labels", null, null)
```
It deletes rows from the `reference_labels` table in `crc_workstation.db`. It **does not** delete cases, inference runs, or batch items.
