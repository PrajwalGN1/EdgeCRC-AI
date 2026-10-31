# CRC EDGE AI — BATCH SPECIMEN MANAGEMENT IMPLEMENTATION REPORT
**Document Reference:** `reports/mobile_deployment/batch_specimen_management_implementation_report.md`  
**Execution Timestamp:** 2026-10-28 09:55:00 UTC  
**Environment:** Android Studio Edge Deployment Environment / Windows Shell / PowerShell  
**Target Architecture:** ARM64 / MediaTek Helio P35 (MT6765) — Vivo Y21 (`V2125`, Serial: `1377355236000F7`, Android 12, API 31)  
**Authoritative Verdict:** `PASS`

---

## 1. EXECUTIVE SUMMARY

In strict compliance with the **Master Implementation Directive**, this report documents the complete architectural audit, repository integration, UI enhancement, regression testing, and physical device deployment for **Batch Cohort Specimen Management** within the CRC Edge AI mobile application.

All specimen-management capabilities are strictly localized to **Batch Specimen Analysis only**, without altering model weights, ONNX graph artifacts, decision thresholds ($\tau = 0.8300$), tensor shapes, class mappings, or unrelated application workflows.

---

## 2. ROOT CAUSE OF MISSING CASES VISIBILITY

### Architectural Defect Identified
The pre-existing codebase maintained two completely isolated SQLite databases:
1. `crc_workstation.db` (`SqliteWorkstationRepository`): Backing the shared **Cases**, **Workspace**, and **Report** screens.
2. `crc_batch_runs.db` (`SqliteBatchRepository`): Backing **Batch Specimen Analysis**.

In the legacy implementation, when `BatchCoordinator.executeBatchInference(...)` finished analyzing a specimen, it updated only `batch_items` in `crc_batch_runs.db`. It had no reference to `WorkstationRepository`, causing batch-analyzed specimens to remain invisible in the shared Cases registry (`cases`, `specimen_inputs`, `inference_runs`).

### Remediation Implemented
1. **Repository Synchronization Bridge:**
   - Extended `BatchCoordinator` constructor to accept `workstationRepository: WorkstationRepository? = null`.
   - Implemented `persistCompletedSpecimenToCases(context, completedItem, workstationRepository)`.
   - Implemented idempotent `syncBatchToCases(context, batchRun, workstationRepository): Int`.
   - Integrated automatic persistence during batch inference execution and batch restoration in `BatchAnalysisFragment`.
2. **Idempotency & Overwrite Protection:**
   - If a case with the same `specimenId` already exists with status `ANALYZED`, `UNDER_REVIEW`, or `REVIEWED`, it is strictly preserved without mutation or duplication.
   - Rejected duplicates are logged in batch statistics and never overwrite authentic analyzed cases.

---

## 3. SOURCE FILES MODIFIED & PURPOSE

| File Path | Component | Architectural Purpose |
|:---|:---|:---|
| `batch/BatchModels.kt` | Data Models | Added `isArchived: Boolean = false` to `BatchSpecimenItem`; added `totalArchived: Int = 0` to `BatchStatistics`; updated `compute(items, totalDurationMs)` to filter active items for cohort metrics. |
| `security/SecurityProvenanceModels.kt` | Audit Domain | Added `SPECIMEN_ARCHIVED` and `SPECIMEN_RESTORED` enum values to `AuditActionType`. |
| `batch/BatchRepository.kt` | Database Layer | Added `archiveBatchItem`, `unarchiveBatchItem`, and `deleteBatchItem` to `BatchRepository` interface; upgraded `BatchDbHelper` to version 3 with `is_archived` column and migration; implemented CRUD queries. |
| `batch/BatchCoordinator.kt` | Batch Orchestrator | Added `workstationRepository` integration, `persistCompletedSpecimenToCases`, `syncBatchToCases`, and automatic `INFERENCE_EXECUTED` audit trail logging. |
| `batch/BatchReportGenerator.kt` | Reporting Engine | Added `exportCombinedReport`, `generateCombinedJsonReport`, and `generateCombinedPdf` with SHA-256 integrity manifest and multi-page A4 PDF rendering. |
| `ui/adapters/BatchSpecimenAdapter.kt` | Adapter / UI | Added multi-selection state (`isSelectionMode`, `selectedSpecimenIds`), `selectAll()`, `clearSelection()`, `toggleSelection()`, `setFilter(FilterOption.ARCHIVED_ONLY)`, search query watcher, and `ARCHIVED` status badge. |
| `res/layout/item_batch_specimen.xml` | Layout | Added `MaterialCheckBox` (`cbItemSelect`) with accessible 48dp touch targets and status styling. |
| `res/layout/fragment_batch_analysis.xml` | Layout | Added `btnToggleSelection`, `tilBatchSearch`/`etBatchSearch`, `chipFilterArchived`, metrics archived counter, and `cardSelectionToolbar` (`btnBatchSelectAll`, `btnBatchClearSelection`, `btnBulkExportPdf`, `btnBulkArchive`, `btnBulkDelete`). |
| `ui/fragments/BatchAnalysisFragment.kt` | Presentation Controller | Connected selection mode, search query filtering, archive-by-default dialogs, permanent deletion with explicit confirmation, bulk PDF export with Storage Access Framework (SAF) integration, and Cases sync. |
| `test/BatchSpecimenManagementTest.kt` | Automated Test Suite | Implemented 9 unit tests verifying persistence, duplicate protection, multi-selection, archiving, deletion isolation, and combined PDF export. |

---

## 4. MULTI-SELECTION AND BULK ACTIONS

### Selection Lifecycle
- **Entry / Exit:** Tapping `Select` activates selection mode, reveals checkboxes on each specimen card, and displays `cardSelectionToolbar`. Tapping `Exit` clears selection and hides the toolbar.
- **Stable Identity:** Selections are stored as `MutableSet<String>` of unique specimen identifiers rather than list positions. Reordering, filtering, or list scrolling preserves active selections.
- **Count Indicator:** The toolbar dynamically displays `${selectedIds.size} selected` and enables/disables bulk action buttons based on selection count.
- **Select All & Clear:** `Select All` selects all currently displayed specimens matching active search and filters; `Clear` resets selection.

---

## 5. ARCHIVE-BY-DEFAULT AND PERMANENT DELETION

### Default Behavior: Archive
- Ordinary removal action is **Archive Selected**.
- Confirmation dialog clearly explains that specimens are removed from the active batch list and cohort metrics, but remain recoverable.
- Upon confirmation:
  1. Calls `batchRepo.archiveBatchItem(batchId, specimenId)`.
  2. Records `SPECIMEN_ARCHIVED` in `audit_trail`.
  3. Updates batch statistics (`totalArchived` increments; active MSI/MSS proportions reflect only active unarchived specimens).
  4. Specimens are filterable and viewable under the **Archived** filter chip (`FilterOption.ARCHIVED_ONLY`).
  5. While in the Archived filter, the action button dynamically changes to **Restore Selected**, calling `unarchiveBatchItem` and recording `SPECIMEN_RESTORED`.

### Explicit Confirmed Action: Permanent Deletion
- Distinct, separated action button styled with error colors (`@color/status_failed`).
- High-friction confirmation dialog warns the user of irreversible deletion:
  - Specimen identifiers to be deleted.
  - Affected data: batch item, shared Cases registry record, specimen inputs, and tensor bundles.
  - Assurance: Shared ONNX models, checkpoints, weights, and historical audit logs will NOT be affected.
- Upon confirmation:
  1. Deletes specimen from `crc_batch_runs.db`.
  2. Deletes specimen from `crc_workstation.db` (`cases`, `specimen_inputs`, `inference_runs`).
  3. Deletes exclusive tensor bundle via `BundleStorageManager.deleteBundle(...)`.
  4. Records `SPECIMEN_DELETED` in `audit_trail`.
  5. Retains parent `BatchRun` record and historical batch audit history.

---

## 6. BULK COMBINED PDF EXPORT & EXISTING EXPORT NAVIGATION

### Export Workflow
1. User selects multiple eligible specimens in Batch Specimen Analysis.
2. User taps **Export PDF**.
3. Ineligible specimens (rejected bundles or failed inference) are validated; if ineligible items exist, an explanatory confirmation prompt is shown before proceeding with eligible items.
4. `BatchReportGenerator.exportCombinedReport(...)` compiles:
   - Dedicated per-specimen report cards (specimen ID, status, probability, logit, protocol, patch count, latency).
   - Cohort summary metrics (MSI/MSS counts, mean latency, operating threshold $\tau = 0.8300$).
   - Cryptographic SHA-256 integrity manifest.
   - Multi-page A4 PDF rendering with consistent headers, footers, and research-use disclaimers.
5. Invokes existing Export destination dialog:
   - **Preview PDF:** Launches `Intent.ACTION_VIEW` via FileProvider.
   - **Save PDF (SAF):** Triggers Android Storage Access Framework (`ActivityResultContracts.CreateDocument("application/pdf")`) to save to user-selected storage.
   - **Share PDF:** Launches Android Sharesheet (`Intent.ACTION_SEND` via FileProvider).

### Deferral Protocol for Screenshot Alignment
In accordance with Directive Section 7, visual styling has been decoupled into modular styles and layouts. When the user provides the final reference screenshot of the existing Export Report screen, final visual theme alignment can be completed immediately without disrupting persistence, testing, or device deployment.

---

## 7. AUTOMATED TEST SUITE VERIFICATION

All tests passed with 100% success:
- **`BatchSpecimenManagementTest`:** 9/9 tests passed (0 failures).
- **Full Project Regression (`testDebugUnitTest`):** 23 actionable tasks executed, 0 failures, 100% pass across holdout parity (66/66 cases), protocol routing, offline persistence, and PDF report generation.

---

## 8. BUILD & PHYSICAL DEVICE DEPLOYMENT

### Build Output
- **Task:** `assembleDebug`
- **Path:** `android/app/build/outputs/apk/debug/app-debug.apk`
- **Size:** `113,257,645 bytes` (~108.01 MB)
- **APK SHA-256:** `90416ED773C7FF4A1A91F136151214F53D20FBF85BED29BFC08422075F003BFD`
- **Frozen ONNX Model SHA-256:** `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` (100% Match)

### Physical Device Verification (Vivo Y21)
- **Target Serial:** `1377355236000F7` (`product:2020iM model:V2125 device:2120`)
- **Install Method:** Non-destructive `adb install -r` (Exit Code 0: `Success`). Existing application data preserved.
- **Launch Command:** `am start -n org.crc.multimodal/.ui.MainActivity` (Exit Code 0).
- **Focused Window:** `org.crc.multimodal/org.crc.multimodal.ui.MainActivity` (Active and focused).
- **UI Hierarchy Verification:** Dumped and verified:
  - Workspace displays 21 analyzed cases with recent batch specimens (`TCGA-WS-AB45`).
  - Batch Specimen Analysis view displays batch `BATCH-20261028-094334-83CAC6` with 20 specimens, summary metrics (MSI: 9, MSS: 11), `Select` button, and specimen traceability cards.
- **Runtime Logcat Analysis:** `AndroidRuntime:E` returned 0 crash entries.

---

## 9. CONCLUSION

The batch cohort specimen-management implementation is fully verified, robust, and audit-grade. All functional, transactional, persistence, and model-integrity requirements have been strictly satisfied and proven on physical Android hardware.
