# Phase 0 — Baseline Audit Before Editing
## Enterprise-Grade UI/UX Redesign — CRC Multimodal Edge AI Android Application

- **Audit Date**: 2026-10-21
- **Branch**: `edge-deployment`
- **Active Commit**: `62e4226b192b1b7895107493da6dfa803f7011ef`
- **Commit Message**: `chore(release): add automation scripts for v1.0.1 GitHub release and asset upload`
- **Working Tree Status**: Clean (no uncommitted changes)

---

## 1. Verified Protected Artifact Inventory & Baseline Hashes

All protected model and evaluation artifacts were checked and verified before making any modifications.

| Artifact Path | Size (Bytes) | SHA-256 Checksum | Baseline Integrity Status |
|---|---|---|---|
| `models/fusion_model_v1_1.pth` | 21,007,347 | `E9A7BBCDBE1713A0ED14B6ECD511AC65D046A52E2F03C69443F5BD3DCF12D453` | VERIFIED_PASS |
| `models/fusion_model_fp32.onnx` | 21,015,452 | `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` | VERIFIED_PASS |
| `models/fusion_model_fp16.onnx` | 10,519,654 | `EA3F5FEE8ED321B5E9AF5E1294A8527093C440FB71837FD2AC54ED01DDD84916` | VERIFIED_PASS |
| `data/patient_registry.json` | 13,158,992 | `7D1C92C51741E080787607733A8B34CB07252C804F527CD567937345A5B22CE2` | VERIFIED_PASS |
| `results/actual_validation_keys.json` | 1,056 | `61D7F8D98388EFBE9FC5E000D695B2F41022720A14A8308A6317BC9441930AA9` | VERIFIED_PASS |
| `reports/mobile_deployment/protocol1_patch_selection_manifest.json` | 138,689 | `E592F3E218169741FCEAE27DEB75E305CF7549739CFD1FBDBB0087B05863E668` | VERIFIED_PASS |
| `android/app/src/main/assets/models/fusion_model_fp32.onnx` | 21,015,452 | `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` | VERIFIED_PASS (Bit-for-bit identical to root FP32 ONNX) |

---

## 2. Baseline Test Execution Results

Prior to editing, all existing test suites were executed in the local environment:

1. **Android Unit Test Suite (`.\gradlew.bat testDebugUnitTest`)**:
   - Total tests: **32**
   - Passed: **32** (100% success rate)
   - Failures: **0**
   - Ignored: **0**
   - Execution time: ~30.89s
   - Status: `VERIFIED_PASS`

2. **Mobile Inference Contract Test Suite (`pytest tests/test_mobile_inference_contract.py`)**:
   - Total tests: **13**
   - Passed: **13** (100% success rate)
   - Failures: **0**
   - Status: `VERIFIED_PASS`

3. **Full Python Regression Test Suite (`PYTHONPATH=. pytest tests/`)**:
   - Total tests: **66**
   - Passed: **66**
   - Failures: **0**
   - Warnings: 3 (HuggingFace Hub deprecation regarding `resume_download`)
   - Status: `VERIFIED_PASS`

4. **Baseline Android APK**:
   - Path: `android/app/build/outputs/apk/debug/app-debug.apk`
   - File size: 75,357,763 bytes (~71.86 MB)
   - SHA-256: `E126D788700019F660A02D643BFC64240BCA813793D8FF7C034A8399EA087917`

---

## 3. Device Availability Audit

- Command: `C:\Android\Sdk\platform-tools\adb.exe devices -l`
- Output: `List of devices attached` (empty)
- Physical Samsung Galaxy S24 status: **Not attached / unavailable via USB/TCP**
- Verification Status: `BLOCKED_DEVICE_UNAVAILABLE`

---

## 4. Current Architecture and UI Audit

### 4.1 Activity & Threading
- Single activity: `org.crc.multimodal.ui.MainActivity`.
- Threading: Uses a background `ExecutorService` (`Executors.newSingleThreadExecutor()`) for model parsing and inference execution.
- Lifecycle: Checks `!isFinishing && !isDestroyed` before invoking `runOnUiThread`. On `onDestroy()`, calls `shutdownNow()` and closes the classifier session.

### 4.2 Inference Integration
- Classifier: `CrcMultimodalClassifier.fromByteArray(assetBytes)`.
- Input validation: `InputValidator.validateVision(...)` and `InputValidator.validateText(...)`.
- Decision threshold: `ThresholdLogic.OPERATING_THRESHOLD = 0.83f`.
- Class mapping: `0 = MSS`, `1 = MSI`.
- Protocols: `PROTOCOL_1_CAPPED` (capped at 1,000 patches) and `PROTOCOL_2_UNCAPPED`.
- Result contract: `InferenceResult` containing `predictedClass`, `predictedLabel`, `probability`, `logit`, `threshold`, `numPatches`, `protocol`, `attentionWeights`, `latencyMs`.

### 4.3 Existing UI Limitations Identified
- Visual hierarchy: Very basic developer-style linear vertical scroll view with generic text cards.
- Result display: Dumps formatted multiline text into a raw monospace `TextView` (`tvInferenceResult`).
- Visualization: No graphical representation of predicted probability relative to the critical 0.83 decision boundary.
- Input configuration: Hardcoded to 10 random patches with no user inspection of patch count, protocol options, or input embedding dimensions.
- Status feedback: Basic linear horizontal progress bar without clear multi-state transitions (initializing, ready, running, complete, error).
- Aesthetic styling: Lacks enterprise-grade healthcare polish, elevation hierarchy, refined typography tokens, and accessible semantic badging.

---

## 5. Scope of UI/UX Redesign Implementation

1. **Design System & Resources**:
   - Clean, medical-grade color palette tokenized in `colors.xml` (Midnight Navy `#0F172A`, Deep Slate `#1E293B`, Medical Teal `#0D9488`, Emerald `#10B981`, Amber Warning `#B45309`, Crimson Error `#B91C1C`, Canvas `#F8FAFC`).
   - Cohesive Day/Night support in `themes.xml` and `values-night/themes.xml`.
   - Reusable shape drawables, card styles, and button variants.
   - Vector drawables for clinical status, neural network, threshold, and badge iconography.

2. **Workspace & Header**:
   - Refined brand identity with clinical research badge and live runtime pill indicator.
   - Prominent primary action button ("Run Research Inference") with clear visual states (disabled, ready, executing).
   - Prominently styled research-use disclaimer card.

3. **Input Configuration & Inspection Panel**:
   - Interactive patch count selector (e.g. Standard 10, Cohort 50, Manifest Max 1,000) allowing inspection of Protocol 1 validation boundary.
   - Protocol selector (Protocol 1 Capped vs Protocol 2 Uncapped).
   - Clear input summary card indicating vision tensor `[1, N, 1024]` and clinical text tensor `[1, 768]`.

4. **Inference Progress & States**:
   - Real indeterminate loader with active execution state labels during background execution.
   - Safe button locking to prevent duplicate concurrent submissions.

5. **Results & Research Interpretation**:
   - High-impact diagnostic result banner with prominent MSI / MSS classification badge.
   - Custom graphical horizontal probability gauge with an authoritative indicator at the 0.83 decision boundary.
   - Clear probability score readout, signed margin to threshold (`Δ = P - 0.83`), and measured inference latency.
   - Expandable technical diagnostic card showing logit, attention weight dimensions, runtime provider, and timestamp.

6. **Model & Architecture Specifications**:
   - Expandable panel showing verified model specs: FP32 ONNX, single-patient sequential execution, 100% on-device local execution, zero network access.
