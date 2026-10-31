# Phase 3.6 — Android Workstation & ML/ONNX Regression Audit

- **Project:** EdgeCRC-AI — Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2
- **Repository:** `https://github.com/PrajwalGN1/EdgeCRC-AI`
- **Audited Branch & Commit Baseline:** `main` (`265061de8c3ffa15c811cc0ac9228a0286d2e2f2`, integrated from `edge-deployment` `db72e5f046b92a3a408cb8b3ca9f6e49a4c9e039`)
- **Audit Timestamp:** `2026-10-31T14:59:12+05:30`
- **Overall Regression Gate Verdict:** **`PASS`** *(Zero build, unit-test, model-checksum, or numerical-parity regressions; physical-device instrumentation suite honestly recorded as `NOT RUN` per non-installation rule)*

---

## 1. Executive Summary Matrix

| Verification Domain | Command / Method | Result | Summary Evidence |
| :--- | :--- | :---: | :--- |
| **Android Gradle Configuration & Manifest** | Static inspection (`build.gradle.kts`, `AndroidManifest.xml`, `file_paths.xml`) | **PASS** | AGP `8.4.2`, Kotlin `1.9.24`, SDK `24`–`34`, Java/JVM `17`, `versionCode = 3`, `versionName = "1.0.2"`, zero `<uses-permission>` tags (`allowBackup="false"`), `noCompress("onnx")`. |
| **Android Debug APK Compilation** | `.\gradlew.bat assembleDebug` | **PASS** | `BUILD SUCCESSFUL in 2m 29s` (`37` actionable tasks); produced `android/app/build/outputs/apk/debug/app-debug.apk` (`113,023,989` B, SHA-256 `5f61b2809d6cafe7cb07641a90bdf6ff57ed7110487527f84c7a19591e18e151`). APK was **not** installed or published automatically. |
| **Android JVM Unit Test Suite** | `.\gradlew.bat testDebugUnitTest` | **PASS** | `BUILD SUCCESSFUL in 1m 14s` — **`23` test suites, `229` unit tests executed, `229` passed, `0` failed, `0` errors, `0` skipped** (`6.090s` JVM test runtime). |
| **Android Connected Instrumentation Suite** | `connectedDebugAndroidTest` (`7` `androidTest` classes) | **NOT RUN** | Intentionally not executed during headless release preparation in accordance with directive rule *"Do not install or publish an APK automatically"*. |
| **Python Research & ONNX Parity Suite** | `python -m pytest tests -v` | **PASS** | **`9` test modules, `66` tests executed, `66` passed, `0` failed, `0` errors, `0` skipped** (`14.23s`). |
| **Frozen Checkpoints, ONNX & Cohort SHA-256** | Independent SHA-256 & cohort verification | **PASS** | Root `models/fusion_model_fp32.onnx` and Android asset `android/app/src/main/assets/models/fusion_model_fp32.onnx` match `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` bit-for-bit (`21,015,452` B). Zero frozen files modified. |
| **ONNX Runtime Contract & Golden Case Parity** | ONNX Runtime `CPUExecutionProvider` (`7` `CRCT` `.bin` cases + `N=1500` dynamic axis) | **PASS** | Exact I/O shapes verified; `7/7` golden reference cases achieved `max_logit_diff = 9.54e-07`, `max_prob_diff = 4.62e-07`, and `0` label flips at `tau = 0.83`; Protocol 2 `N=1500` executed with `attention_weights` sum `1.000000`. |

---

## 2. Android Architecture & Subsystem Verification

### 2.1 Build, Manifest, Security & Storage Inspection

| Subsystem | Inspected Source Files | Status | Technical Verification Details |
| :--- | :--- | :---: | :--- |
| **Gradle Build & ONNX Runtime Dependency** | `android/build.gradle.kts`, `android/app/build.gradle.kts` | **PASS** | `com.microsoft.onnxruntime:onnxruntime-android:1.18.0` (`implementation`) + `com.microsoft.onnxruntime:onnxruntime:1.18.0` (`testImplementation`); `abiFilters = ["arm64-v8a", "armeabi-v7a", "x86_64"]`; `noCompress.addAll(listOf("onnx"))` preserves direct memory-mapped asset loading. |
| **Manifest & `FileProvider` Isolation** | `android/app/src/main/AndroidManifest.xml`, `res/xml/file_paths.xml` | **PASS** | Zero `<uses-permission>` declarations (strictly offline); `android:allowBackup="false"`; `androidx.core.content.FileProvider` (`authorities="org.crc.multimodal.fileprovider"`, `exported="false"`, `grantUriPermissions="true"`) scoped exclusively to `filesDir/reports/` and `cacheDir/reports/`. |
| **SQLite Schema & Non-Destructive Migrations** | `SqliteWorkstationRepository.kt`, `BatchRepository.kt` | **PASS** | `WorkstationDbHelper` (`crc_workstation.db` `v4`, `8` tables) uses additive `CREATE TABLE IF NOT EXISTS` migrations (`v1->v2` `reference_labels`, `v2->v3` `audit_trail`, `v3->v4` `report_history`). `BatchDbHelper` (`crc_batch_runs.db` `v3`, `2` tables) uses additive `ALTER TABLE batch_items ADD COLUMN` (`v1->v2` `protocol` & `protocol_reason`, `v2->v3` `is_archived`). Zero destructive `DROP TABLE` calls. |
| **Specimen Import/Export & Validation** | `AuthenticBundleLoader.kt`, `NpyTensorLoader.kt`, `InputValidator.kt`, `SafeFileValidator.kt` | **PASS** | Verified `CRCB` v2 per-modality SHA-256 checks, `CRCT` v1 binary parsing, JSON v2.0 manifest loading, `.npy` v1.0/v2.0 loading, NaN/Inf rejection, `100 MB` file cap, and canonical path-traversal (`..`) rejection. |
| **Batch Processing, Cohort Evaluation & PDF Reports** | `BatchCoordinator.kt`, `CohortEvaluationEngine.kt`, `ResearchAnalyticsEngine.kt`, `PdfReportGenerator.kt`, `BatchReportGenerator.kt` | **PASS** | Verified `MAX_BATCH_SIZE = 100`, isolated per-item failure handling, automatic Protocol 2 routing for `N > 1000`, bulk Archive/Restore/Delete/Export PDF actions, 66-case TCGA holdout confusion-matrix & ROC-AUC/PR-AUC computation, and native A4 PDF + SHA-256 generation. |

### 2.2 Android JVM Unit Test Breakdown (`23` Suites, `229/229` Passed)

| JUnit Test Suite | Tests | Failures | Errors | Skipped | Time (`s`) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AuthenticBundleLoaderTest` | `14` | `0` | `0` | `0` | `0.595` | **PASS** |
| `BatchCoordinatorTest` | `25` | `0` | `0` | `0` | `0.426` | **PASS** |
| `BatchProtocol2AndStatusColorTest` | `14` | `0` | `0` | `0` | `0.342` | **PASS** |
| `BatchSpecimenManagementTest` | `9` | `0` | `0` | `0` | `0.059` | **PASS** |
| `CohortEvaluationTest` | `7` | `0` | `0` | `0` | `0.114` | **PASS** |
| `CrcMultimodalClassifierInferenceTest` | `8` | `0` | `0` | `0` | `0.477` | **PASS** |
| `CrcMultimodalHoldoutParityTest` | `2` | `0` | `0` | `0` | `2.693` | **PASS** |
| `EdgePerformanceMonitoringTest` | `11` | `0` | `0` | `0` | `0.011` | **PASS** |
| `FinalReleaseGateTest` | `12` | `0` | `0` | `0` | `0.160` | **PASS** |
| `InferenceResultTest` | `3` | `0` | `0` | `0` | `0.002` | **PASS** |
| `InputValidatorTest` | `15` | `0` | `0` | `0` | `0.033` | **PASS** |
| `NpyTensorLoaderTest` | `16` | `0` | `0` | `0` | `0.021` | **PASS** |
| `OfflineInferenceAndPersistenceTest` | `8` | `0` | `0` | `0` | `0.023` | **PASS** |
| `OnnxRuntimeManagerTest` | `9` | `0` | `0` | `0` | `0.943` | **PASS** |
| `PdfReportGeneratorTest` | `13` | `0` | `0` | `0` | `0.011` | **PASS** |
| `ProtocolAndReportRegressionTest` | `9` | `0` | `0` | `0` | `0.009` | **PASS** |
| `ReferenceTensorLoaderTest` | `2` | `0` | `0` | `0` | `0.052` | **PASS** |
| `ReportPersistenceMigrationTest` | `5` | `0` | `0` | `0` | `0.005` | **PASS** |
| `ResearchAnalyticsTest` | `6` | `0` | `0` | `0` | `0.009` | **PASS** |
| `ResultExplanationServiceTest` | `10` | `0` | `0` | `0` | `0.022` | **PASS** |
| `SecurityProvenanceAuditTest` | `15` | `0` | `0` | `0` | `0.021` | **PASS** |
| `ThresholdLogicTest` | `4` | `0` | `0` | `0` | `0.004` | **PASS** |
| `WorkstationWorkflowTest` | `12` | `0` | `0` | `0` | `0.058` | **PASS** |
| **Total (`23` Suites)** | **`229`** | **`0`** | **`0`** | **`0`** | **`6.090`** | **PASS** |

---

## 3. ML Artifacts, ONNX Tensor Contracts & Numerical Parity Verification

### 3.1 Frozen Artifact SHA-256 Checksum Verification

| Repository Path | Size (Bytes) | Verified SHA-256 Digest | Integrity Status |
| :--- | ---: | :--- | :---: |
| `models/fusion_model_v1_1.pth` | `21,007,347` | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **PASS** |
| `models/late_fusion_model_v1_0.pth` | `5,255,861` | `e98b945354578055aa691f2469390346753523ce7e50315f77098f33d2626ae7` | **PASS** |
| `models/fusion_model_fp32.onnx` | `21,015,452` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **PASS** |
| `android/app/src/main/assets/models/fusion_model_fp32.onnx` | `21,015,452` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **PASS** *(Bit-identical to root ONNX)* |
| `models/fusion_model_fp16.onnx` | `10,519,654` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | **PASS** |
| `results/actual_validation_keys.json` (`66` holdout keys) | `1,056` | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | **PASS** |
| `data/patient_registry.json` (`326` TCGA-COAD cases) | `13,158,992` | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | **PASS** |
| `android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json` (`66` labels: `50` MSS, `16` MSI) | `12,639` | `21073937c768d87bcf0126af11de1ca44475a290ac572d9cd7bceaa25f2b264c` | **PASS** |

### 3.2 ONNX Graph Contract, Threshold & Protocol Verification

- **Graph Inputs Verified (`CPUExecutionProvider`):**
  - `vision_seq`: `['batch_size', 'num_patches', 1024]` (`tensor(float)`) -> runtime `[1, N, 1024]`
  - `text_summary`: `['batch_size', 768]` (`tensor(float)`) -> runtime `[1, 768]`
- **Graph Outputs Verified:**
  - `logit`: `['batch_size', 1]` (`tensor(float)`) -> runtime `[1, 1]`
  - `probability`: `['batch_size', 1]` (`tensor(float)`) -> runtime `[1, 1]`
  - `attention_weights`: `['batch_size', 1, 'num_patches']` (`tensor(float)`) -> runtime `[1, 1, N]`
- **Operating Threshold & Class Mapping Verified (`CrcMultimodalClassifier.kt` & `CrcMultimodalHoldoutParityTest.kt`):**
  - Frozen operating threshold `tau = 0.83` (`P >= 0.83 -> Class 1 [MSI]`; `P < 0.83 -> Class 0 [MSS]`).
- **Evaluation Protocols Verified (`InputValidator.kt`, `ProtocolAndReportRegressionTest.kt`, `test_mobile_inference_contract.py`):**
  - **Protocol 1 (`PROTOCOL_1_CAPPED`):** Accepts `1 <= N <= 1000` patches; explicitly rejects `N > 1000` without silent truncation.
  - **Protocol 2 (`PROTOCOL_2_UNCAPPED`):** Accepts full sequence `N >= 1` (`N = 1500` and `N = 4305` verified with `attention_weights` shape `[1, 1, N]` and attention sum `1.000000`).

### 3.3 Golden Reference Tensor Parity (`7` Bundled `CRCT` v1 Cases)

| Specimen ID | Patch Count (`N`) | ONNX `P(MSI)` | PyTorch Ref `P(MSI)` | `|Delta P|` | Predicted Class (`tau = 0.83`) | Attention Shape | Status |
| :--- | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| `TCGA-AA-3845` | `502` | `0.986665` | `0.986665` | `3.58e-07` | `MSI` (`1`) | `(1, 1, 502)` | **PASS** |
| `TCGA-AF-2693` | `52` | `0.014129` | `0.014129` | `4.00e-07` | `MSS` (`0`) | `(1, 1, 52)` | **PASS** |
| `TCGA-AG-A00C` | `99` | `0.011387` | `0.011387` | `1.99e-07` | `MSS` (`0`) | `(1, 1, 99)` | **PASS** |
| `TCGA-CM-6674` | `1000` | `0.985367` | `0.985367` | `5.96e-08` | `MSI` (`1`) | `(1, 1, 1000)` | **PASS** |
| `TCGA-CM-6675` | `541` | `0.162250` | `0.162250` | `4.62e-07` | `MSS` (`0`) | `(1, 1, 541)` | **PASS** |
| `TCGA-D5-6539` | `10` | `0.031855` | `0.031855` | `4.06e-07` | `MSS` (`0`) | `(1, 1, 10)` | **PASS** |
| `TCGA-D5-6540` | `345` | `0.990385` | `0.990385` | `2.98e-07` | `MSI` (`1`) | `(1, 1, 345)` | **PASS** |

- **Aggregate Golden Parity:** `max_logit_diff = 9.54e-07`, `max_prob_diff = 4.62e-07`, **`0` classification label flips**.

### 3.4 Python `pytest` Regression Suite (`9` Modules, `66/66` Passed)

- `tests/test_encoders_interface.py` (`4` passed)
- `tests/test_export_wrapper.py` (`3` passed)
- `tests/test_fp16_export_and_drift.py` (`11` passed — includes 66-case cohort parity & boundary Case 36)
- `tests/test_fusion_and_predictor.py` (`7` passed)
- `tests/test_image_preprocessing.py` (`4` passed)
- `tests/test_input_validation.py` (`11` passed)
- `tests/test_mobile_inference_contract.py` (`13` passed — includes Protocol 1/2 contracts & NaN/Inf rejection)
- `tests/test_onnx_export_and_parity.py` (`9` passed — dynamic patch counts `1` through `4305`)
- `tests/test_sampling_manifest.py` (`3` passed — order-independent sampling manifest fingerprints)
- **Result:** `66 passed, 3 warnings in 14.23s` (`huggingface_hub` deprecation warnings in mock fail-closed tests only; `0` failures).
