# Phase 3.6 — Android Workstation & ML/ONNX Regression Audit (`v1.1.0` Aligned)

- **Project:** EdgeCRC-AI — Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2
- **Repository:** `https://github.com/PrajwalGN1/EdgeCRC-AI`
- **Audited Branch & Version Metadata:** `main` (`versionCode = 4`, `versionName = "1.1.0"`, following `20ad70fa34c2a23da8f2e66073e4f55ee5c9b399`)
- **Audit Date:** `2026-10-31`
- **Overall Regression Gate Verdict:** **`PASS`** *(Zero build, unit-test, model-checksum, or numerical-parity regressions after `v1.1.0` version alignment and licensing documentation updates; physical-device instrumentation suite recorded as `NOT RUN`)*

---

## 1. Executive Summary Matrix

| Verification Domain | Command / Method | Result | Summary Evidence |
| :--- | :--- | :---: | :--- |
| **Android Gradle Version & Manifest Alignment** | Static & `output-metadata.json` inspection (`build.gradle.kts`, `AndroidManifest.xml`, `fragment_settings.xml`) | **PASS** | AGP `8.4.2`, Kotlin `1.9.24`, SDK `24`–`34`, Java/JVM `17`, **`versionCode = 4`**, **`versionName = "1.1.0"`**, zero `<uses-permission>` tags (`allowBackup="false"`), `noCompress("onnx")`. |
| **Android Debug APK Compilation** | `.\gradlew.bat assembleDebug` | **PASS** | `BUILD SUCCESSFUL in 59s` (`37` actionable tasks); verified `output-metadata.json` (`versionCode: 4`, `versionName: "1.1.0"`) and `app-debug.apk` (`113,023,985` B, SHA-256 `956de57169774880d1dfb877e98da419f8fce6ffdfdfee2ce06bd4a2c795b44e`). Not auto-installed or published. |
| **Android JVM Unit Test Suite (Fresh Uncached Run)** | `.\gradlew.bat testDebugUnitTest --rerun-tasks` | **PASS** | `BUILD SUCCESSFUL in 2m 42s` (`23` actionable tasks executed) — **`23` test suites, `229` unit tests executed, `229` passed, `0` failed, `0` errors, `0` skipped**. |
| **Android Connected Instrumentation Suite** | `connectedDebugAndroidTest` (`7` `androidTest` classes) | **NOT RUN** | Requires physical USB device installation; intentionally not run during release preparation. |
| **Python Research & ONNX Parity Suite** | `python -m pytest tests -v` | **PASS** | **`9` test modules, `66` tests executed, `66` passed, `0` failed, `0` errors, `0` skipped** (`112.32s`). |
| **Frozen Checkpoints, ONNX & Cohort SHA-256** | Independent SHA-256 & cohort verification | **PASS** | Root `models/fusion_model_fp32.onnx` and Android asset `android/app/src/main/assets/models/fusion_model_fp32.onnx` remain unchanged at `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` (`21,015,452` B); `models/fusion_model_v1_1.pth` unchanged at `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` (`21,007,347` B). |
| **ONNX Runtime Contract & Golden Case Parity** | ONNX Runtime `CPUExecutionProvider` (`7` `CRCT` `.bin` cases + `N=1500` dynamic axis) | **PASS** | Exact I/O shapes verified; `7/7` golden reference cases achieved `max_logit_diff = 9.54e-07`, `max_prob_diff = 4.62e-07`, and `0` label flips at `tau = 0.83`; Protocol 2 `N=1500` executed with `attention_weights` sum `1.000000`. |

---

## 2. Android JVM Unit Test Suite Breakdown (`23` Suites, `229/229` Passed)

All `23` JUnit XML reports generated under `android/app/build/test-results/testDebugUnitTest/` during the uncached `v1.1.0` run passed with `0` failures, `0` errors, and `0` skipped tests:
- `AuthenticBundleLoaderTest` (`14`), `BatchCoordinatorTest` (`25`), `BatchProtocol2AndStatusColorTest` (`14`), `BatchSpecimenManagementTest` (`9`), `CohortEvaluationTest` (`7`), `CrcMultimodalClassifierInferenceTest` (`8`), `CrcMultimodalHoldoutParityTest` (`2`), `EdgePerformanceMonitoringTest` (`11`), `FinalReleaseGateTest` (`12`), `InferenceResultTest` (`3`), `InputValidatorTest` (`15`), `NpyTensorLoaderTest` (`16`), `OfflineInferenceAndPersistenceTest` (`8`), `OnnxRuntimeManagerTest` (`9`), `PdfReportGeneratorTest` (`13`), `ProtocolAndReportRegressionTest` (`9`), `ReferenceTensorLoaderTest` (`2`), `ReportPersistenceMigrationTest` (`5`), `ResearchAnalyticsTest` (`6`), `ResultExplanationServiceTest` (`10`), `SecurityProvenanceAuditTest` (`15`), `ThresholdLogicTest` (`4`), and `WorkstationWorkflowTest` (`12`) -> **`229` passed**.

---

## 3. Frozen Model & Cohort Checksums Verified Post-Alignment

| Repository Path | Size (Bytes) | Verified SHA-256 Digest | Integrity Status |
| :--- | ---: | :--- | :---: |
| `models/fusion_model_v1_1.pth` | `21,007,347` | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **PASS** |
| `models/late_fusion_model_v1_0.pth` | `5,255,861` | `e98b945354578055aa691f2469390346753523ce7e50315f77098f33d2626ae7` | **PASS** |
| `models/fusion_model_fp32.onnx` | `21,015,452` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **PASS** |
| `android/app/src/main/assets/models/fusion_model_fp32.onnx` | `21,015,452` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **PASS** *(Bit-identical)* |
| `models/fusion_model_fp16.onnx` | `10,519,654` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | **PASS** |
| `results/actual_validation_keys.json` (`66` holdout keys) | `1,056` | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | **PASS** |
| `data/patient_registry.json` (`326` TCGA-COAD cases) | `13,158,992` | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | **PASS** |
| `android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json` (`66` labels) | `12,639` | `21073937c768d87bcf0126af11de1ca44475a290ac572d9cd7bceaa25f2b264c` | **PASS** |
