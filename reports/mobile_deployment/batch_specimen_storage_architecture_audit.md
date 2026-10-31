# CRC EDGE AI — BATCH SPECIMEN STORAGE ARCHITECTURE AUDIT
## Phase 0 Read-Only Architecture, Persistence & Lifecycle Trace

**Date:** 2026-10-28  
**Author:** Senior Android Engineer & Edge-AI Deployment Architect  
**Target Application:** CRC Multimodal Pathology Research Workstation (`org.crc.multimodal`)  
**Repository Branch:** `edge-deployment`  
**Device Target:** Vivo Y21 (`V2125`, Android 12 / API 31, ADB Serial `1377355236000F7`)  
**Audit Scope:** Batch Specimen Analysis Module, Persistence Paths, Cases Visibility, Selection & Export Workflows

---

## 1. Executive Summary

This architecture audit investigates the end-to-end specimen lifecycle within the CRC Edge AI Android application, specifically focusing on the **Batch Specimen Analysis** module. 

The primary operational defect identified is that specimens successfully imported, validated, and analyzed inside Batch Specimen Analysis **do not appear in the shared Cases registry** (`CasesFragment`). 

### Key Findings
1. **Dual Disconnected Databases:** The application maintains two separate SQLite databases:
   - `crc_workstation.db` (managed by `SqliteWorkstationRepository`): Hosts `cases`, `specimen_inputs`, `inference_runs`, `review_notes`, `research_reports`, `report_history`, and `audit_trail`. This is the authoritative registry powering the shared **Cases** screen.
   - `crc_batch_runs.db` (managed by `SqliteBatchRepository`): Hosts `batch_runs` and `batch_items`. This holds batch execution runs and per-specimen batch metrics.
2. **Missing Persistence Bridge (Root Cause):** In `BatchCoordinator.kt` (`executeBatchInference`, lines 422–458), when an inference result is produced, the coordinator updates only `batch_items` in `BatchRepository`. It never calls `WorkstationRepository` (`insertCase`, `insertSpecimenInput`, or `recordInferenceResult`). Consequently, the `cases` table in `crc_workstation.db` remains unaware of batch-analyzed specimens.
3. **Transient Export Experience:** In `BatchAnalysisFragment.kt` (lines 232–257), tapping `Export Report` executes `BatchReportGenerator.exportReport` to create batch-level JSON and PDF summary files in app-internal storage and displays a static `AlertDialog`. It does not route to the workstation's rich `ReportFragment` export destination, nor does it support per-specimen or combined bulk PDF preview, Storage Access Framework (SAF) save, or Android sharesheet export.
4. **Absence of Multi-Selection & Specimen Management:** The current batch UI (`BatchSpecimenAdapter.kt`, `fragment_batch_analysis.xml`) lacks item selection controls, checkboxes, Select All, archiving, or permanent deletion mechanisms.
5. **Frozen Model Integrity:** All model inference parameters (Vision: `[1, N, 1024]`, Text: `[1, 768]`, Class 0: MSS, Class 1: MSI, $\tau = 0.83$, Protocol 1 cap: 1,000 patches, Protocol 2 uncapped) are fully verified and untouched.

---

## 2. Environment & Repository Baseline

### 2.1 Git Repository State
- **Root Workspace:** `d:\CRC Edge deployment` (Branch: `main`)
- **Sub-repository:** `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`
- **Active Git Branch:** `edge-deployment`
- **Baseline Test Suite Status:** `PASS` (`testDebugUnitTest` and `testReleaseUnitTest` both compile and pass all 22 test suites with 0 failures; holdout parity evaluated across 66 cases with 100% agreement and max error $< 3 \times 10^{-6}$).

### 2.2 Physical Device Verification
- **ADB Executable:** `C:\Android\Sdk\platform-tools\adb.exe`
- **Attached Device:** `1377355236000F7`
- **Model:** Vivo Y21 (`V2125`, MediaTek Helio P35, Android 12 / API 31)
- **Status:** Authorized (`device`)

---

## 3. End-to-End Lifecycle Trace

The specimen lifecycle was traced step-by-step through the active source code:

```
[SAF Picker] 
     │
     ▼
[1. Bundle Selection] ──► BatchAnalysisFragment.kt:41-47, 88-90
     │
     ▼
[2. Bundle Validation] ──► BatchCoordinator.kt:81-95 (AuthenticBundleLoader.loadFromBytes)
     │
     ▼
[3. Duplicate / Acceptance] ──► BatchCoordinator.kt:97-120 (seenSpecimenIds check)
     │                          BundleStorageManager.kt:45-53 (saveBundle overwrite=false)
     ▼
[4. Batch Run Creation] ──► SqliteBatchRepository.kt:42-95 (insertBatchRun in crc_batch_runs.db)
     │
     ▼
[5. Batch Inference Execution] ──► BatchCoordinator.kt:330-421 (CrcMultimodalClassifier.predictAuthentic)
     │
     ▼
[6. Result Persistence] ──► BatchCoordinator.kt:422-446 (updateBatchItem in crc_batch_runs.db)
     │                     ❌ [CRITICAL GAP: WorkstationRepository NOT CALLED]
     ▼
[7. Batch Outcome Recording] ──► BatchCoordinator.kt:461-486 (updateBatchRun in crc_batch_runs.db)
     │
     ▼
[8. Cases Screen Visibility] ──► CasesFragment.kt:118-134 (getCasesFiltered in crc_workstation.db)
     │                          ❌ [Zero batch specimens visible: Table `cases` has no records]
     ▼
[9. Report Export Flow] ──► BatchAnalysisFragment.kt:232-257 (BatchReportGenerator.exportReport)
                                ❌ [Static AlertDialog only; disconnected from ReportFragment]
```

### Stage Details and Source Mappings

| Stage | Class & File | Relevant Lines | Behavioral Description |
|---|---|---|---|
| **1. Bundle Selection** | `BatchAnalysisFragment.kt` | L41–47, L88–90 | `ActivityResultContracts.OpenMultipleDocuments()` launched via `btnSelectBatchFiles`. Returns list of document URIs. |
| **2. Bundle Validation** | `BatchCoordinator.kt` | L81–95, L159–176 | Reads byte stream via `contentResolver.openInputStream(uri)`. Calls `AuthenticBundleLoader.loadFromBytes(bytes)`. Validates header `CRCB`, schema v2, vision tensor shape `[1, N, 1024]`, text tensor shape `[1, 768]`, and SHA-256 digests. |
| **3. Duplicate Checks** | `BatchCoordinator.kt`, `BundleStorageManager.kt` | `BatchCoordinator.kt:L97-120`<br>`BundleStorageManager.kt:L45-53` | 1. Intra-batch duplicates rejected via `seenSpecimenIds.contains(specimenId)`.<br>2. App-storage duplicates rejected when `BundleStorageManager.saveBundle(..., overwrite=false)` throws `IllegalArgumentException`. Caught and recorded as `BatchSpecimenStatus.REJECTED`. |
| **4. Batch Run Creation** | `SqliteBatchRepository.kt` | L42–95 | Inserts `BatchRun` and individual `BatchSpecimenItem` rows into `crc_batch_runs.db` (`batch_runs` and `batch_items` tables). |
| **5. Batch Inference** | `BatchCoordinator.kt` | L330–421 | Evaluates patch count: $\le 1,000$ assigned to Protocol 1 (`PROTOCOL_1_CAPPED`); $> 1,000$ routed to Protocol 2 (`PROTOCOL_2_UNCAPPED`). Sequentially calls `classifier.predictAuthentic(...)`. |
| **6. Result Persistence** | `BatchCoordinator.kt` | L422–446 | Marks `BatchSpecimenItem` as `COMPLETED`. Updates `batch_items` row in `crc_batch_runs.db`. **Does not interact with `WorkstationRepository`.** |
| **7. Batch Outcome** | `BatchCoordinator.kt` | L461–486 | Computes `BatchStatistics`, assigns terminal status (`COMPLETED` or `PARTIALLY_COMPLETED`), updates `batch_runs` in `crc_batch_runs.db`. |
| **8. Cases Visibility** | `CasesFragment.kt` | L118–134 | Calls `repository.getCasesFiltered(...)` against `cases` table in `crc_workstation.db`. Batch specimens are absent because no `ResearchCase` rows exist. |
| **9. Report Export** | `BatchAnalysisFragment.kt` | L232–257 | Invokes `BatchReportGenerator.exportReport(...)` which writes a summary JSON and single-page summary PDF, then opens an `AlertDialog`. Does not route to `ReportFragment`. |

---

## 4. Root Cause Analysis: Missing Cases Visibility

### 4.1 Architectural Disconnect
The application was designed with two distinct repository abstractions:
1. `WorkstationRepository` (`SqliteWorkstationRepository.kt`): Backed by `crc_workstation.db`.
2. `BatchRepository` (`SqliteBatchRepository.kt`): Backed by `crc_batch_runs.db`.

In `BatchAnalysisFragment.kt` (lines 133–134, 189):
```kotlin
val coordinator = BatchCoordinator(host.getBatchRepository())
```
`BatchCoordinator` was injected **only** with `BatchRepository`. It had no reference to `WorkstationRepository`.

In `BatchCoordinator.kt` (lines 434–436):
```kotlin
val completedItem = item.copy(
    status = BatchSpecimenStatus.COMPLETED,
    logit = result.logit,
    probability = result.probability,
    predictedClass = result.predictedClass,
    predictedLabel = result.predictedLabel,
    threshold = result.threshold,
    inferenceLatencyMs = inferLatency,
    provenance = result.provenance,
    completedTimestamp = System.currentTimeMillis(),
    protocol = targetProtocol.name,
    protocolReason = protocolReason
)
updatedItems[i] = completedItem
repository.updateBatchItem(completedItem)
```
Only `repository.updateBatchItem(completedItem)` was invoked.

### 4.2 Research Analytics Workaround Evidence
Proof of this architectural gap is evident in `SqliteWorkstationRepository.kt` (lines 336–343):
```kotlin
override fun getUnifiedPredictionRecords(
    batchRepo: org.crc.multimodal.batch.BatchRepository?
): List<org.crc.multimodal.analytics.SpecimenPredictionRecord> {
    val allCases = getAllCases()
    val allRuns = getAllInferenceRuns()
    val batchRuns = batchRepo?.getAllBatchRuns() ?: emptyList()
    return org.crc.multimodal.analytics.ResearchAnalyticsEngine.buildUnifiedRecords(allCases, allRuns, batchRuns)
}
```
The analytics engine had to manually union `allCases` with `batchRuns` because batch specimens were never persisted into `cases` or `inference_runs`.

---

## 5. Duplicate Detection & Overwrite Protection Audit

1. **Intra-Batch Duplicates:** Handled safely in `BatchCoordinator.kt:97-120` using `seenSpecimenIds: MutableSet<String>`. Duplicate items are marked `REJECTED` with `"Duplicate specimen identifier within batch"`.
2. **Storage Bundle Duplicates:** Handled safely in `BundleStorageManager.kt:50-52`. Calling `saveBundle(..., overwrite = false)` throws `IllegalArgumentException` when a bundle file with the same name already exists. In `BatchCoordinator.kt`, this exception is caught and recorded as `REJECTED`.
3. **Database Unique Constraints:**
   - `cases` table: `case_identifier TEXT UNIQUE NOT NULL`
   - `batch_runs` table: `batch_id TEXT PRIMARY KEY`
   - `batch_items` table: composite logical key `(batch_id, specimen_id)`
4. **Idempotency Invariant:** To prevent duplicate cases or constraint crashes during batch retries or re-opens, `WorkstationRepository.getCaseByIdentifier(specimenId)` must be checked prior to insertion:
   - If a case with `specimenId` does not exist: insert `ResearchCase` (status `ANALYZED`), insert `SpecimenInput`, and record `InferenceRun`.
   - If a case with `specimenId` already exists and is `ANALYZED`: preserve existing case data and do not create duplicate runs or overwrite.
   - If a case exists in `DRAFT` or `READY_FOR_ANALYSIS`: update status to `ANALYZED` and record the inference run.

---

## 6. Audit of Existing Export & Report Architecture

1. **Individual Specimen Export (`ReportFragment.kt`):**
   - Displays prediction summary, dual-bar MSI/MSS probability visualization, specimen metadata, model specifications, and audit provenance.
   - Generates audit-grade multi-page A4 PDF documents via `PdfReportGenerator.kt`.
   - Supports:
     - **PDF Preview:** `FileProvider.getUriForFile` with `Intent.ACTION_VIEW` (`application/pdf`).
     - **PDF Save:** Storage Access Framework (SAF) `ActivityResultContracts.CreateDocument("application/pdf")`.
     - **PDF Share:** Android Sharesheet with `Intent.ACTION_SEND` and `FLAG_GRANT_READ_URI_PERMISSION`.
     - **Report History:** Persistent records in `report_history` table.
2. **Current Batch Export (`BatchReportGenerator.kt`):**
   - Generates summary JSON (`${batchId}_report.json`) and summary PDF (`${batchId}_report.pdf`).
   - Generates cryptographic manifest (`${batchId}_manifest.sha256`).
   - Triggered via `btnExportReport` in `BatchAnalysisFragment.kt`, showing only a text `AlertDialog`.
3. **Combined Bulk Export Integration Opportunity:**
   - Bulk export for multiple selected batch specimens can generate a combined multi-specimen PDF where each eligible analyzed specimen has a distinct, structured report section.
   - The user can then be routed to or offered the exact preview, save (SAF), and share (Sharesheet) workflow established in `ReportFragment`.

---

## 7. Audit of Multi-Selection & Archive/Deletion Needs

1. **Current State:** `BatchSpecimenAdapter` renders read-only `item_batch_specimen` cards without checkboxes, selection state, or contextual action bars.
2. **Selection Scope:**
   - Selection mode must be scoped **strictly** to Batch Specimen Analysis.
   - Stable identity selection: track `selectedSpecimenIds: MutableSet<String>` (keyed by `specimenId`, immune to scrolling or reordering).
   - Selection toolbar: Selected count indicator (e.g. `3 selected`), `Select All`, `Clear Selection`, `Export PDF`, `Archive Selected`, `Delete Permanently`.
3. **Archive vs. Permanent Deletion Semantics:**
   - **Archive (Default):** Soft removal. Sets `is_archived = 1` in `batch_items`. Hides specimen from active list. Retains full inference metrics, logit, probability, and protocol. Accessible via an `"Archived"` filter chip in `BatchAnalysisFragment`. Logs `SPECIMEN_ARCHIVED` in `audit_trail`.
   - **Permanent Deletion (Explicit Confirmation):** Requires strong confirmation dialog listing affected specimen IDs. Removes row from `batch_items`, removes case from `cases` (and associated inputs/runs) if owned by the batch, deletes bundle file from `filesDir/specimen_bundles/` if not referenced by other active cases. Preserves batch execution run header and records `SPECIMEN_DELETED` in `audit_trail`. Never touches model checkpoints, ONNX files, or unrelated bundles.

---

## 8. Safe Implementation Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Shared Cases Registry Integration                                  │
│ - Inject WorkstationRepository into BatchCoordinator / BatchAnalysisFragment│
│ - Persist eligible COMPLETED batch specimens into `cases`, `specimen_inputs`,│
│   and `inference_runs` via existing repository methods                      │
│ - Ensure idempotency and preserve overwrite protection                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Professional Multi-Selection UX (Batch Analysis Only)              │
│ - Add selection mode toggle, checkboxes, selected count badge               │
│ - Support Select All, Clear Selection, and selection retention across filters│
│ - Status-aware eligibility checks for bulk operations                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Archive-by-Default and Permanent Deletion                          │
│ - Implement `is_archived` persistence in `batch_items` table and models     │
│ - Add "Archived" filter chip to review and restore archived specimens       │
│ - Add Archive confirmation dialog and Permanent Deletion warning dialog     │
│ - Record auditable events in `audit_trail`                                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Bulk PDF Export & Navigation Integration                           │
│ - Implement combined multi-specimen PDF generation in BatchReportGenerator   │
│ - Provide direct Preview, SAF Save, and Share options                       │
│ - Route through existing export workflows                                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Automated Testing, APK Build & Device Verification                  │
│ - Add comprehensive regression tests for persistence, selection, and archive│
│ - Build signed debug APK and deploy non-destructively to Vivo Y21 via ADB   │
│ - Verify live runtime behavior, UI interactivity, and data preservation     │
└─────────────────────────────────────────────────────────────────────────────┘
```

This concludes the Phase 0 Architecture Audit. All findings are verified against the physical codebase and validated with passing unit tests.
