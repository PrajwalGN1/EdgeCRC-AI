# Pre-Implementation Audit Report: Offline Deterministic Result Generation and PDF Reporting
**Project**: Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2  
**Workstation**: Native Android Kotlin Edge AI Pathology Research Workstation  
**Date of Audit**: October 26, 2026  
**Auditor**: Senior Android Engineer & Quality Auditor  
**Report Path**: `reports/mobile_deployment/result_generation_pdf_preimplementation_audit.md`

---

## 1. Actual Project Root and Git Branch

- **Absolute Project Root**: `D:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`
- **Current Git Branch**: `edge-deployment` (tracking `origin/edge-deployment`, clean working tree relative to remote except previously modified UI dimension resources)
- **Git Commit Baseline**: HEAD at `edge-deployment`
- **Operating System Environment**: Windows 11, PowerShell, Kotlin 1.9.24, Java 17, Android SDK 34 (Compile & Target SDK: 34, Min SDK: 24)
- **Connected Device Status**: No ADB device currently authorized/online (`adb devices -l` returned empty list). All physical device workflows will be classified as `NOT_DEVICE_TESTED`.

---

## 2. Actual Package Structure and Build Configuration

- **Android Module Path**: `android/app`
- **Application ID / Namespace**: `org.crc.multimodal`
- **Package Hierarchy**:
  - `org.crc.multimodal.ui`: Core host activities (`MainActivity`), custom views (`ProbabilityThresholdBarView`), navigation interfaces (`WorkstationNavigationHost`).
  - `org.crc.multimodal.ui.fragments`: Feature screens (`WorkspaceFragment`, `CasesFragment`, `NewSpecimenFragment`, `SpecimenReviewFragment`, `InferenceExecutionFragment`, `FindingsFragment`, `ReviewNotesFragment`, `ReportFragment`, `CaseDetailsFragment`, `SettingsFragment`, `BatchAnalysisFragment`, `CohortEvaluationFragment`, `AnalyticsDashboardFragment`).
  - `org.crc.multimodal.ui.adapters`: ViewHolders and RecyclerView adapters.
  - `org.crc.multimodal.data`: Entities (`ResearchCase`, `SpecimenInput`, `InferenceRun`, `ReviewNote`, `ResearchReport`, `CaseStatus`), and repository implementations (`WorkstationRepository`, `SqliteWorkstationRepository`, `InMemoryWorkstationRepository`).
  - `org.crc.multimodal.inference`: ONNX Runtime session management (`OnnxRuntimeManager`, `CrcMultimodalClassifier`, `InferenceProtocol`, `ThresholdLogic`, `InputValidator`, `AuthenticFeatureBundle`, `AuthenticBundleLoader`, `BundleStorageManager`).
  - `org.crc.multimodal.batch`: Multi-specimen batch coordination and batch PDF reporting (`BatchCoordinator`, `BatchReportGenerator`, `SqliteBatchRepository`).
  - `org.crc.multimodal.evaluation`: Cohort metrics, confusion matrices, reference label matching (`CohortEvaluator`, `ReferenceLabel`).
  - `org.crc.multimodal.report`: Text-based provenance report generation (`ReportGenerator`).
  - `org.crc.multimodal.security`: Cryptographic provenance manifests, audit records (`ProvenanceAuditManager`, `AuditTrailRecord`, `AuditActionType`).
- **Build Configuration**:
  - `compileSdk`: 34
  - `minSdk`: 24
  - `targetSdk`: 34
  - `jvmTarget`: "17"
  - `unitTests.isReturnDefaultValues`: `true`
  - ABI filters: `arm64-v8a`, `armeabi-v7a`, `x86_64`
  - Dependencies:
    - `com.microsoft.onnxruntime:onnxruntime-android:1.18.0`
    - `androidx.core:core-ktx:1.13.1`
    - `androidx.appcompat:appcompat:1.6.1`
    - `com.google.android.material:material:1.12.0`
    - `junit:junit:4.13.2`
    - `org.jetbrains.kotlin:kotlin-test:1.9.24`
    - `com.microsoft.onnxruntime:onnxruntime:1.18.0` (test scope)
    - `org.json:json:20240303` (test scope)
- **Manifest Audit**:
  - `AndroidManifest.xml` enforces zero network permissions (no `android.permission.INTERNET`).
  - `androidx.core.content.FileProvider` is **NOT** currently declared in `AndroidManifest.xml`.
  - Directory `android/app/src/main/res/xml/` does not exist yet; no `file_paths.xml` present.

---

## 3. Existing Navigation and Findings Screen Implementation

- **Navigation Pattern**:
  - Managed by `MainActivity.kt` implementing `WorkstationNavigationHost`.
  - Does not use Android Jetpack Navigation Component XML graph; navigation uses programmatic FragmentTransactions (`replaceRootFragment` for bottom nav items, `pushFragment` with backstack for detail screens).
  - Navigation method for report: `navigateToReport(caseId: Long, runId: Long?)` pushes `ReportFragment.newInstance(caseId, runId)`.
- **Findings Screen (`FindingsFragment.kt`)**:
  - Binds case identifier, timestamp, predicted class (MSI vs MSS), formatted probability ($P$), margin ($\Delta = P - 0.8300$), latency ($ms$), patch count, protocol description, and execution diagnostics.
  - Button `btnFindingsGenerateReport` triggers `host.navigateToReport(currentCase.id, inferenceRun.id)`.
  - Button `btnFindingsAddNote` navigates to `ReviewNotesFragment`.
  - Button `btnFindingsBackToCase` returns to `CaseDetailsFragment`.
  - Implementation is clean, functional, and fully verified.

---

## 4. Existing Result and Inference Data Models

- **`InferenceRun` (`org.crc.multimodal.data.InferenceRun`)**:
  - `id: Long = 0L`
  - `caseId: Long`
  - `modelIdentifier: String = "fusion_model_fp32.onnx"`
  - `inferenceArchitecture: String = "LOCAL_ONNX_FP32"`
  - `protocol: InferenceProtocol? = InferenceProtocol.PROTOCOL_1_CAPPED`
  - `predictedClass: Int` (validated: `require(predictedClass in 0..1)`)
  - `predictedLabel: String` ("MSI" or "MSS")
  - `probability: Float` (validated: `require(!probability.isNaN() && probability in 0.0f..1.0f)`)
  - `logit: Float`
  - `threshold: Float = 0.83f`
  - `margin: Float = probability - threshold`
  - `latencyMs: Double`
  - `numPatches: Int`
  - `diagnosticsSummary: String? = null`
  - `executionTimestamp: Long = System.currentTimeMillis()`
  - `status: String = "SUCCESS"`
- **`ResearchReport` (`org.crc.multimodal.data.ResearchReport`)**:
  - `id: Long = 0L`
  - `caseId: Long`
  - `inferenceRunId: Long? = null`
  - `generatedAt: Long = System.currentTimeMillis()`
  - `reportContent: String`
  - `findingsSummary: String`

---

## 5. Existing Threshold and Class-Mapping Implementation

- **Location**: `org.crc.multimodal.inference.ThresholdLogic.kt`
- **Constants**:
  - `DECISION_THRESHOLD: Float = 0.83f`
  - `CLASS_MSS: Int = 0`
  - `LABEL_MSS: String = "MSS"`
  - `CLASS_MSI: Int = 1`
  - `LABEL_MSI: String = "MSI"`
- **Classification Rule**:
  ```kotlin
  fun classify(probability: Float): ClassificationResult {
      require(!probability.isNaN()) { "Probability must not be NaN" }
      require(probability in 0.0f..1.0f) { "Probability must be in [0.0, 1.0]" }
      val predictedClass = if (probability >= DECISION_THRESHOLD) CLASS_MSI else CLASS_MSS
      val predictedLabel = if (probability >= DECISION_THRESHOLD) LABEL_MSI else LABEL_MSS
      val margin = probability - DECISION_THRESHOLD
      return ClassificationResult(predictedClass, predictedLabel, margin)
  }
  ```
- **Evaluation**: The class mapping and frozen decision threshold ($\tau = 0.83$) are mathematically consistent, strictly verified in unit tests, and identical across Python and Kotlin implementations.

---

## 6. Existing SQLite Database Schema, Migrations, and Repository Interfaces

- **Active SQLite Database**: `crc_workstation.db`
- **Active Helper Class**: `SqliteWorkstationRepository.WorkstationDbHelper`
- **Current DB Version**: `3`
- **Existing Tables**:
  1. `cases` (id, case_identifier UNIQUE, specimen_description, research_context, status, patch_count, protocol, created_at, updated_at)
  2. `specimen_inputs` (id, case_id, input_type, local_uri, filename, validation_status, created_at)
  3. `inference_runs` (id, case_id, model_identifier, inference_architecture, protocol, predicted_class, predicted_label, probability, logit, threshold, margin, latency_ms, num_patches, diagnostics_summary, execution_timestamp, status)
  4. `review_notes` (id, case_id, category, author_label, content, created_at, updated_at)
  5. `research_reports` (id, case_id UNIQUE, inference_run_id, generated_at, report_content, findings_summary)
  6. `reference_labels` (specimen_id PRIMARY KEY, ground_truth_label, ground_truth_class, cohort_id, source, imported_at) — added in migration version 2
  7. `audit_trail` (id, event_timestamp, action_type, specimen_id, details, sha256_fingerprint, performed_by) — added in migration version 3
- **Architectural Observation on `research_reports`**:
  - `case_id` has a `UNIQUE` constraint in the original schema, limiting storage to a single report per case.
  - The directive mandates **Feature C — Persistent Report History**, which requires:
    - Multiple saved reports per case (historical revisions).
    - Stable report identifier (`report_id: String` / UUID).
    - Template version tracking (`template_version: String`).
    - Export status (`export_status: String` e.g., GENERATED, EXPORTED, FAILED, MISSING_FILE).
    - File URI and managed file path (`file_uri: String?`, `file_path: String?`).
    - Error information on failure (`error_message: String?`).
    - Cryptographic SHA-256 hash of the generated PDF (`sha256_hash: String?`).
- **Migration Strategy**:
  - Safely increment `DB_VERSION` from `3` to `4`.
  - In `onUpgrade(db, oldVersion, newVersion)`:
    - If `oldVersion < 4`: create `report_history` table with necessary indexes (`case_id`, `report_id`, `created_at DESC`), preserving existing `research_reports` table and all data intact.
  - Provide non-destructive repository methods in `WorkstationRepository` and both `SqliteWorkstationRepository` and `InMemoryWorkstationRepository`.

---

## 7. Existing Research-Report Functionality

- **`org.crc.multimodal.report.ReportGenerator`**:
  - Compiles plain text report string with sections:
    1. Header & provenance
    2. Case identification
    3. Specimen & input specification
    4. Multimodal fusion findings
    5. Model & inference provenance
    6. Technical diagnostics
    7. Case review notes
    8. Responsible-use research disclaimer
  - Also outputs provenance manifests via `ProvenanceAuditManager`.
- **`ReportFragment.kt`**:
  - Binds compiled text into a single scrollable monospace `TextView`.
  - Contains "Copy" (to Android clipboard) and "Share" (via plain text intent `ACTION_SEND` with `text/plain`).
  - Lacks PDF generation, PDF preview, PDF save/share, and persistent report history browsing.

---

## 8. Existing PDF-Generation or Android Sharing Code

- **Batch PDF Generator**:
  - `org.crc.multimodal.batch.BatchReportGenerator` implements PDF generation for batch cohort runs using native `android.graphics.pdf.PdfDocument`.
  - A4 dimensions (595 x 842 points), standard paints, disclaimer headers, tabular presentation.
  - In JVM test environments where Android framework stubs are present, calls to `PdfDocument` are caught cleanly without breaking test suites.
- **Single-Case Workstation PDF Generator**:
  - Currently **NOT_IMPLEMENTED** for single-case workstation findings.
- **Android Sharing & FileProvider**:
  - Currently **NOT_IMPLEMENTED** for PDF documents. Text sharing exists only via `text/plain`.
  - No `FileProvider` is configured in `AndroidManifest.xml`.

---

## 9. Existing Tests and Build Commands

- **Unit Test Execution Command**:
  ```powershell
  cd "D:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2\android"
  .\gradlew.bat testDebugUnitTest --continue
  ```
- **Baseline Test Result**:
  - Status: **PASSED** (Exit code 0, 57s build duration).
  - Executed Suites (18 suites, 100% pass rate):
    1. `AuthenticBundleLoaderTest`
    2. `BatchCoordinatorTest`
    3. `CohortEvaluationTest`
    4. `CrcMultimodalClassifierInferenceTest`
    5. `CrcMultimodalHoldoutParityTest`
    6. `EdgePerformanceMonitoringTest`
    7. `FinalReleaseGateTest`
    8. `InferenceResultTest`
    9. `InputValidatorTest`
    10. `NpyTensorLoaderTest`
    11. `OfflineInferenceAndPersistenceTest`
    12. `OnnxRuntimeManagerTest`
    13. `ProtocolAndReportRegressionTest`
    14. `ReferenceTensorLoaderTest`
    15. `ResearchAnalyticsTest`
    16. `SecurityProvenanceAuditTest`
    17. `ThresholdLogicTest`
    18. `WorkstationWorkflowTest`

---

## 10. Frozen Artifact Hashes Before Changes

The following baseline SHA-256 hashes and file sizes were computed and verified on the actual repository files prior to making any modifications:

| Artifact Path | Size (Bytes) | Baseline SHA-256 Hash |
| :--- | :--- | :--- |
| `models/fusion_model_v1_1.pth` | 21,007,347 | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` |
| `models/fusion_model_fp32.onnx` | 21,015,452 | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` |
| `android/app/src/main/assets/models/fusion_model_fp32.onnx` | 21,015,452 | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` |
| `data/patient_registry.json` | 13,158,992 | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` |
| `data/reference_labels/tcga_holdout_reference_labels.csv` | 4,414 | `e2ea895c0035d98f918cf58ba02ea3bb29a21b4b039430307005c43fede9f13a` |
| `android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json` | 12,639 | `21073937c768d87bcf0126af11de1ca44475a290ac572d9cd7bceaa25f2b264c` |
| `reports/mobile_deployment/protocol1_patch_selection_manifest.json` | 138,689 | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` |
| `reports/mobile_deployment/cohort_evaluation_reference_metrics.json` | 37,913 | `fd44f597ebfc827e3ed09b2686685c81038ea2ba2393b92dc0382c7af60435f3` |
| `reports/mobile_deployment/phase1_fp32_benchmark_data.json` | 95,327 | `d5da5ae5595c2f8a1317f3cc1c2f9b39b751b1aa4097ddeabfa337672282cba9` |
| `reports/mobile_deployment/phase2_parity_benchmark_data.json` | 85,446 | `a104998c50bc6927c60302fa28f08567690842d31112f0e6416830f0ba90614c` |

---

## 11. Existing Report History Capabilities and Missing Functionality

| Capability | Current State | Missing Requirements to Implement |
| :--- | :--- | :--- |
| **Deterministic Result Explanation** | `PARTIALLY_IMPLEMENTED` | Missing dedicated `ResultExplanationService` that encapsulates threshold analysis, consistency checks, input-readiness evaluation, and deterministic structured explanations. |
| **PDF Generation (Single Case)** | `NOT_IMPLEMENTED` | Missing `PdfReportGenerator` supporting multi-page A4, clear headers, badges, metadata tables, wrapped notes, and SHA-256 provenance manifest. |
| **PDF In-App / External Preview** | `NOT_IMPLEMENTED` | Missing content URI generation, intent construction for PDF previewers, and fallback handling. |
| **PDF File Saving (SAF)** | `NOT_IMPLEMENTED` | Missing Storage Access Framework integration (`CreateDocument`) for user-specified file export. |
| **PDF Sharing** | `NOT_IMPLEMENTED` | Missing `FileProvider` setup, `ACTION_SEND` intent with `application/pdf`, and temporary read URI grants. |
| **Persistent Report History** | `PARTIALLY_IMPLEMENTED` | Existing table `research_reports` only allows 1 report per case and has no file path or export status. Missing `report_history` table, multi-report retrieval, retry mechanism, and status tracking. |
| **Safe Migration (V3 $\to$ V4)** | `NOT_IMPLEMENTED` | Need non-destructive SQLite migration from schema version 3 to version 4. |

---

## 12. Classification of Relevant Features

- **Frozen ONNX Inference & Threshold Logic**: `IMPLEMENTED_AND_VERIFIED`
- **Case Management & Review Notes**: `IMPLEMENTED_AND_VERIFIED`
- **Text Report Compilation**: `IMPLEMENTED_AND_VERIFIED`
- **Offline Deterministic Result Explanation**: `PARTIALLY_IMPLEMENTED` (to be extended into dedicated service)
- **PDF Generation for Workstation Cases**: `NOT_IMPLEMENTED`
- **PDF Preview / Save / Share**: `NOT_IMPLEMENTED`
- **Report History & Safe Migration**: `PARTIALLY_IMPLEMENTED` (schema upgrade required)
- **Physical Device Execution**: `NOT_DEVICE_TESTED` (no device connected via ADB at audit time)

---

## 13. Proposed Files to Add or Modify

### A. New Files to Create:
1. `reports/mobile_deployment/result_generation_pdf_preimplementation_audit.md` (this audit document)
2. `android/app/src/main/java/org/crc/multimodal/report/ResultExplanationService.kt`:
   - Pure, deterministic, zero-network result explanation engine.
   - Evaluates probability against $\tau = 0.8300$, computes signed margin, validates internal consistency between stored class and probability, formats diagnostics, distinguishes research output from clinical diagnosis.
3. `android/app/src/main/java/org/crc/multimodal/report/PdfReportGenerator.kt`:
   - Professional multi-page PDF generation via `android.graphics.pdf.PdfDocument`.
   - Formatted header, metadata grid, classification badge, threshold comparison, input provenance, reviewer notes, research-use disclaimer, and cryptographic SHA-256 calculation.
4. `android/app/src/main/java/org/crc/multimodal/data/PersistentReportRecord.kt`:
   - Structured entity for report history (`reportId`, `caseId`, `inferenceRunId`, `templateVersion`, `createdAt`, `exportStatus`, `fileUri`, `filePath`, `modelVersion`, `sha256Hash`, `errorMessage`).
5. `android/app/src/main/res/xml/file_paths.xml`:
   - Defines accessible internal directories (`files/reports` and `cache/reports`) for `FileProvider`.
6. `android/app/src/main/res/layout/item_report_history.xml`:
   - List item layout for displaying saved reports in the history sheet/dialog.
7. `android/app/src/test/java/org/crc/multimodal/ResultExplanationServiceTest.kt`:
   - Comprehensive unit tests for deterministic explanations, threshold boundaries, consistency warnings, and missing fields.
8. `android/app/src/test/java/org/crc/multimodal/PdfReportGeneratorTest.kt`:
   - Tests for data validation, template formatting, SHA-256 calculation, and error resilience.
9. `android/app/src/test/java/org/crc/multimodal/ReportPersistenceMigrationTest.kt`:
   - Tests for SQLite schema migration version 3 to 4, CRUD operations on report history, and data integrity.
10. `reports/mobile_deployment/result_generation_pdf_implementation_report.md`:
    - Final implementation report detailing changes, test commands, frozen artifact hash verification, and limitations.

### B. Files to Modify:
1. `android/app/src/main/AndroidManifest.xml`:
   - Add `androidx.core.content.FileProvider` declaration with `android:authorities="org.crc.multimodal.fileprovider"`.
2. `android/app/src/main/java/org/crc/multimodal/data/WorkstationRepository.kt`:
   - Add report history methods: `insertReportRecord`, `updateReportRecordStatus`, `getReportHistoryForCase`, `getReportRecordById`, `deleteReportRecord`.
3. `android/app/src/main/java/org/crc/multimodal/data/SqliteWorkstationRepository.kt`:
   - Increment `DB_VERSION = 4`.
   - Update `onCreate` and implement `onUpgrade` for version 4 (`CREATE TABLE IF NOT EXISTS report_history ...`).
   - Implement repository interface methods with thread-safe synchronized locks.
4. `android/app/src/main/java/org/crc/multimodal/data/InMemoryWorkstationRepository.kt`:
   - Implement in-memory stubs for report history methods.
5. `android/app/src/main/java/org/crc/multimodal/ui/fragments/ReportFragment.kt`:
   - Integrate `ResultExplanationService` and `PdfReportGenerator`.
   - Add PDF Preview (`ACTION_VIEW`), PDF Share (`ACTION_SEND` with content URI), PDF Save (Storage Access Framework `CreateDocument`), and Report History viewing/reopening.
6. `android/app/src/main/res/layout/fragment_report.xml`:
   - Add buttons for "Preview PDF", "Save PDF", "Share PDF", and "Report History", with clean responsive arrangement and visual states.
