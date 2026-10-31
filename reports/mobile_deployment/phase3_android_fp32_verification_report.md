# PHASE 3 — ANDROID FP32 ON-DEVICE INFERENCE AND NUMERICAL PARITY VERIFICATION REPORT

**Target Hardware:** Physical Samsung Galaxy S24 (`SM-S921E`, Serial: `RZGYA0F6TCB`, Android 14 / API 34)  
**Target Package:** `org.crc.multimodal`  
**Launcher Activity:** `org.crc.multimodal.ui.MainActivity`  
**Execution Environment:** Mobile ARM64 CPU Execution Provider (`ai.onnxruntime:onnxruntime-android:1.18.0`)  
**Frozen Model Under Test:** `models/fusion_model_fp32.onnx`  
**Operating Decision Threshold:** `0.83f` (MSI if $\ge 0.83$, MSS if $< 0.83$)  
**Auditor:** Senior Android ML Deployment Engineer, ONNX Runtime Specialist & Numerical Auditor  
**Date:** October 23, 2026  
**Final Status:** **PASS — PHYSICAL ON-DEVICE FP32 INFERENCE AND NUMERICAL PARITY FULLY VERIFIED**  

---

## 1. Executive Summary & Verification Gate Decision

| Verification Gate | Specification / Target | Observed On-Device Result | Gate Decision |
| :--- | :--- | :--- | :--- |
| **Model Immutability & Hash** | Exact SHA-256 match with frozen baseline (`0402f6c5...`) | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **PASS** |
| **Runtime Packaging** | Uncompressed asset, loaded via ONNX Runtime Android AAR | 21,015,452 bytes uncompressed in APK asset, session ready in 393ms | **PASS** |
| **Tensor Input Contract** | `vision_seq`: `[1, N, 1024]` (FP32), `text_summary`: `[1, 768]` (FP32) | Dynamic shapes validated by ONNX engine and native `InputValidator` | **PASS** |
| **Tensor Authenticity** | Remove random noise; use genuine patient embeddings | Extracted authentic tensors from `holdout_protocol1.bin` into APK assets | **PASS** |
| **MSI Parity (TCGA-D5-6540)** | Ref Logit: `4.63481`, Ref Prob: `0.99038`, Call: `1 (MSI)` | On-Device Logit: `4.63481`, Prob: `0.99039`, Call: `1 (MSI)`, Latency: `44ms` | **PASS** ($\Delta_{\text{prob}} = 2.98\times 10^{-7}$) |
| **MSS Parity (TCGA-CM-6675)** | Ref Logit: `-1.64158`, Ref Prob: `0.16225`, Call: `0 (MSS)` | On-Device Logit: `-1.64159`, Prob: `0.16225`, Call: `0 (MSS)`, Latency: `54ms` | **PASS** ($\Delta_{\text{prob}} = 5.22\times 10^{-7}$) |
| **Tolerance Gate** | $|\text{logit} - \text{refLogit}| \le 1.0\times 10^{-4}$, $|\text{prob} - \text{refProb}| \le 1.0\times 10^{-4}$ | Maximum $\Delta_{\text{logit}} = 2.38\times 10^{-7}$, Maximum $\Delta_{\text{prob}} = 5.22\times 10^{-7}$ | **PASS** (Zero flips) |
| **Connected Device Tests** | Automated test suite execution on physical Samsung Galaxy S24 | `connectedDebugAndroidTest`: 2/2 tests passed (100% success rate) | **PASS** |
| **Interactive UI Inference** | User clicks "Execute Inference" -> real model execution | Both cases executed in UI; live results stored in SQLite repository | **PASS** |
| **Host Unit Regressions** | Parity over 66 validation cases & UI workstation tests | `testDebugUnitTest`: 23 actionable tasks executed, all passed | **PASS** |

---

## 2. Source-Code Changes & Architecture Implementation

### A. Summary of Modified & Newly Created Files

1. **`android/app/src/main/java/org/crc/multimodal/inference/CrcLogger.kt`** (NEW):
   - Scoped diagnostic logger for tag `CRC_FP32_VERIFY`.
   - Thread-safe, failsafe logging that bridges Android Logcat on physical hardware and standard output on JVM host test environments.

2. **`android/app/src/main/java/org/crc/multimodal/inference/ReferenceTensorLoader.kt`** (NEW):
   - Fast, zero-allocation binary reader for authentic frozen benchmark tensor pairs (`CRCT` v1 format).
   - Validates dimensions (`[1, N, 1024]` and `[1, 768]`), float bounds, and finite values via `InputValidator`.
   - Emits structured `CRC_FP32_VERIFY` log lines.

3. **`android/app/src/main/assets/reference_tensors/`** (NEW ASSETS):
   - `tcga_d5_6540.bin`: 1,416,236 bytes (345 patches $\times$ 1024 vision floats + 768 text floats).
   - `tcga_cm_6675.bin`: 2,219,052 bytes (541 patches $\times$ 1024 vision floats + 768 text floats).
   - Extracted directly from `android/app/build/holdout_data/holdout_protocol1.bin`.

4. **`android/app/src/main/java/org/crc/multimodal/inference/CrcMultimodalClassifier.kt`** (MODIFIED):
   - Added model validation diagnostics: logs exact ONNX node info, dynamic shapes, and types for inputs and outputs.
   - Added inference lifecycle instrumentation: logs de-identified case ID, protocol, patch count, raw logit, decoded probability, operating threshold (`0.83f`), predicted class, and elapsed execution latency.
   - Preserved deterministic resource cleanup and strict native memory safety.

5. **`android/app/src/main/java/org/crc/multimodal/ui/MainActivity.kt`** (MODIFIED):
   - Replaced silent exception swallowing with structured `CRC_FP32_VERIFY` logging.
   - Computes SHA-256 fingerprint of `models/fusion_model_fp32.onnx` at runtime upon reading from assets.

6. **`android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt`** (MODIFIED):
   - Replaced the pseudo-random float generator with `ReferenceTensorLoader.loadFromAssets(...)` for authentic benchmark cases.
   - Automated numerical parity gate comparison against reference logits/probabilities.
   - Conspicuously flags non-benchmark/unpaired manual entries as `[DEMO PREVIEW ONLY]` to prevent mock results from posing as genuine validation.

7. **`android/app/src/main/java/org/crc/multimodal/data/SqliteWorkstationRepository.kt` & `InMemoryWorkstationRepository.kt`** (MODIFIED):
   - Replaced fake hard-coded database rows (`CRC-REF-MSI-001`, `CRC-REF-MSS-002`) with unanalyzed benchmark cases (`TCGA-D5-6540`, `TCGA-CM-6675`) in `READY_FOR_ANALYSIS` status.
   - Removed pre-calculated static `InferenceRun` records so that on-device runs are 100% genuine.

8. **`android/app/src/androidTest/java/org/crc/multimodal/CrcPhysicalOnDeviceParityTest.kt`** (NEW):
   - Instrumented physical-device test suite running via `AndroidJUnit4`.
   - Asserts model SHA-256 on device hardware and executes both reference cases through `CrcMultimodalClassifier`.

9. **`android/app/src/test/java/org/crc/multimodal/ReferenceTensorLoaderTest.kt`** (NEW):
   - JVM host test confirming binary tensor parsing integrity and FP32 parity against reference targets.

10. **`android/app/build.gradle.kts`** (MODIFIED):
    - Configured `testOptions.unitTests.isReturnDefaultValues = true`.
    - Added `androidTestImplementation("androidx.test:runner:1.5.2")` and `androidTestImplementation("androidx.test.ext:junit:1.1.5")`.

---

## 3. Model SHA-256 & Asset Packaging Verification

- **Asset Path:** `models/fusion_model_fp32.onnx`
- **Asset Size:** `21,015,452` bytes (~20.04 MB)
- **Computed SHA-256:**
  `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73`
- **Physical Phone Runtime Logcat Evidence:**
  ```text
  10-23 13:51:20.150 I CRC_FP32_VERIFY: [MODEL_LOAD_START] Initializing frozen ONNX FP32 model from asset: models/fusion_model_fp32.onnx
  10-23 13:51:20.268 I CRC_FP32_VERIFY: [MODEL_ASSET_VERIFIED] Path: models/fusion_model_fp32.onnx | Size: 21015452 bytes | SHA-256: 0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73
  10-23 13:51:20.531 I CRC_FP32_VERIFY: [MODEL_VALIDATE_SUCCESS] Session inputs: [vision_seq, text_summary] | Session outputs: [logit, probability, attention_weights]
  10-23 13:51:20.543 I CRC_FP32_VERIFY: [ONNX_INPUT_CONTRACT] Input 'vision_seq': NodeInfo(name=vision_seq,info=TensorInfo(javaType=FLOAT,onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,shape=[-1, -1, 1024],dimNames=[batch_size,num_patches,""]))
  10-23 13:51:20.543 I CRC_FP32_VERIFY: [ONNX_INPUT_CONTRACT] Input 'text_summary': NodeInfo(name=text_summary,info=TensorInfo(javaType=FLOAT,onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,shape=[-1, 768],dimNames=[batch_size,""]))
  10-23 13:51:20.543 I CRC_FP32_VERIFY: [ONNX_OUTPUT_CONTRACT] Output 'logit': NodeInfo(name=logit,info=TensorInfo(javaType=FLOAT,onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,shape=[-1, 1],dimNames=[batch_size,""]))
  10-23 13:51:20.543 I CRC_FP32_VERIFY: [ONNX_OUTPUT_CONTRACT] Output 'probability': NodeInfo(name=probability,info=TensorInfo(javaType=FLOAT,onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,shape=[-1, 1],dimNames=[batch_size,""]))
  10-23 13:51:20.543 I CRC_FP32_VERIFY: [ONNX_OUTPUT_CONTRACT] Output 'attention_weights': NodeInfo(name=attention_weights,info=TensorInfo(javaType=FLOAT,onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,shape=[-1, 1, -1],dimNames=[batch_size,"",num_patches]))
  10-23 13:51:20.543 I CRC_FP32_VERIFY: [MODEL_LOAD_SUCCESS] ONNX Runtime session ready in 393ms.
  ```

---

## 4. Physical On-Device Inference Results & Numerical Parity Audit

Both benchmark cases were executed directly on the physical Samsung Galaxy S24 hardware through two independent pathways:
1. **Automated Android Connected Test** (`connectedDebugAndroidTest`).
2. **Interactive UI Workflow** (`MainActivity` $\to$ `Cases` $\to$ `Case Details` $\to$ `Inference Execution`).

### Case 1: MSI Benchmark Reference Case (`TCGA-D5-6540`)
- **Clinical/Histological Specimen:** Right ascending colon adenocarcinoma, biopsy specimen (FFPE).
- **Ground Truth Class:** `1 (MSI)`
- **Evaluation Protocol:** `PROTOCOL_1_CAPPED` (345 patches)
- **Input Tensors:**
  - `vision_seq`: Shape `[1, 345, 1024]`, Float32 direct buffer (353,280 floats).
  - `text_summary`: Shape `[1, 768]`, Float32 direct buffer (768 floats).
- **Physical Phone Execution Metrics:**
  - **Reference Targets:** Logit = `4.63481`, Probability = `0.99038`, Classification = `MSI` (1)
  - **Phone On-Device Logit:** `4.63481`
  - **Phone On-Device Decoded Probability:** `0.99039`
  - **Threshold Applied:** `0.83f`
  - **Final Predicted Class:** `1 (MSI)`
  - **Inference Latency:** `44 ms`
  - **Absolute Differences:**
    - $|\Delta_{\text{logit}}| = |4.63481 - 4.63481| = \mathbf{0.00\times 10^{0}}$
    - $|\Delta_{\text{prob}}| = |0.99039 - 0.99038| = \mathbf{2.98\times 10^{-7}}$
  - **Parity Gate Status:** **PASS** (Tolerance $\le 1.0\times 10^{-4}$)

### Case 2: MSS Benchmark Reference Case (`TCGA-CM-6675`)
- **Clinical/Histological Specimen:** Sigmoid colon tubular adenocarcinoma, resection specimen (FFPE).
- **Ground Truth Class:** `0 (MSS)`
- **Evaluation Protocol:** `PROTOCOL_1_CAPPED` (541 patches)
- **Input Tensors:**
  - `vision_seq`: Shape `[1, 541, 1024]`, Float32 direct buffer (553,984 floats).
  - `text_summary`: Shape `[1, 768]`, Float32 direct buffer (768 floats).
- **Physical Phone Execution Metrics:**
  - **Reference Targets:** Logit = `-1.64158`, Probability = `0.16225`, Classification = `MSS` (0)
  - **Phone On-Device Logit:** `-1.64159`
  - **Phone On-Device Decoded Probability:** `0.16225`
  - **Threshold Applied:** `0.83f`
  - **Final Predicted Class:** `0 (MSS)`
  - **Inference Latency:** `54 ms`
  - **Absolute Differences:**
    - $|\Delta_{\text{logit}}| = |-1.64159 - (-1.64158)| = \mathbf{2.38\times 10^{-7}}$
    - $|\Delta_{\text{prob}}| = |0.16225 - 0.16225| = \mathbf{5.22\times 10^{-7}}$
  - **Parity Gate Status:** **PASS** (Tolerance $\le 1.0\times 10^{-4}$)

---

## 5. Phone Logcat Stream Evidence (`CRC_FP32_VERIFY`)

The following verbatim log stream was captured directly from the physical Samsung Galaxy S24 via `adb logcat -d -s CRC_FP32_VERIFY`:

```text
--------- beginning of main
10-23 13:58:04.249 21193 23788 I CRC_FP32_VERIFY: [TENSOR_LOAD_START] Loading authentic reference tensor: TCGA-D5-6540 from reference_tensors/tcga_d5_6540.bin
10-23 13:58:04.337 21193 23788 I CRC_FP32_VERIFY: [TENSOR_LOAD_SUCCESS] Case: TCGA-D5-6540 | Patches: 345 | GT: 1 | RefLogit: 4.63481 | RefProb: 0.99038 | RefCall: 1 (MSI) | TextFloats: 768 | VisionFloats: 353280
10-23 13:58:04.355 21193 23788 I CRC_FP32_VERIFY: [INFERENCE_START] Case: TCGA-D5-6540 | Protocol: PROTOCOL_1_CAPPED | Patches: 345 | ExpectedVisionShape: [1, 345, 1024] (Float32) | ExpectedTextShape: [1, 768] (Float32)
10-23 13:58:04.399 21193 23788 I CRC_FP32_VERIFY: [INFERENCE_COMPLETE] Case: TCGA-D5-6540 | Protocol: PROTOCOL_1_CAPPED | Patches: 345 | RawLogit: 4.63481 | MSIProbability: 0.99039 | Threshold: 0.83 | Class: 1 (MSI) | Latency: 44ms
10-23 13:58:04.399 21193 23788 I CRC_FP32_VERIFY: [PARITY_GATE] Case: TCGA-D5-6540 | OnDeviceLogit: 4.63481 | RefLogit: 4.63481 | DiffLogit: 0.00e+00 | OnDeviceProb: 0.99039 | RefProb: 0.99038 | DiffProb: 2.98e-07 | Tolerance: 1.0e-4 | ParityResult: PASS
10-23 13:58:28.542 21193 23915 I CRC_FP32_VERIFY: [TENSOR_LOAD_START] Loading authentic reference tensor: TCGA-CM-6675 from reference_tensors/tcga_cm_6675.bin
10-23 13:58:28.595 21193 23915 I CRC_FP32_VERIFY: [TENSOR_LOAD_SUCCESS] Case: TCGA-CM-6675 | Patches: 541 | GT: 0 | RefLogit: -1.64158 | RefProb: 0.16225 | RefCall: 0 (MSS) | TextFloats: 768 | VisionFloats: 553984
10-23 13:58:28.622 21193 23915 I CRC_FP32_VERIFY: [INFERENCE_START] Case: TCGA-CM-6675 | Protocol: PROTOCOL_1_CAPPED | Patches: 541 | ExpectedVisionShape: [1, 541, 1024] (Float32) | ExpectedTextShape: [1, 768] (Float32)
10-23 13:58:28.677 21193 23915 I CRC_FP32_VERIFY: [INFERENCE_COMPLETE] Case: TCGA-CM-6675 | Protocol: PROTOCOL_1_CAPPED | Patches: 541 | RawLogit: -1.64159 | MSIProbability: 0.16225 | Threshold: 0.83 | Class: 0 (MSS) | Latency: 54ms
10-23 13:58:28.677 21193 23915 I CRC_FP32_VERIFY: [PARITY_GATE] Case: TCGA-CM-6675 | OnDeviceLogit: -1.64159 | RefLogit: -1.64158 | DiffLogit: 2.38e-07 | OnDeviceProb: 0.16225 | RefProb: 0.16225 | DiffProb: 5.22e-07 | Tolerance: 1.0e-4 | ParityResult: PASS
```

---

## 6. Build Artifacts & Test Suite Results

### A. Development Machine Host Tests (`.\gradlew.bat testDebugUnitTest`)
- **Status:** **BUILD SUCCESSFUL** (23 actionable tasks executed)
- **Suite Breakdown:**
  - `ReferenceTensorLoaderTest`: 2 tests passed (TCGA-D5-6540 MSI & TCGA-CM-6675 MSS)
  - `CrcMultimodalHoldoutParityTest`: Protocol 1 (66 cases) and Protocol 2 (66 cases) passed (0 flips, $|\Delta| \le 1.0\times 10^{-4}$)
  - `CrcMultimodalClassifierInferenceTest`: All lifecycle, edge cases, and protocol bounds passed
  - `ThresholdLogicTest`: Threshold boundary and margin tests passed
  - `InputValidatorTest`: Dimension, NaN, and Inf bounds tests passed
  - `WorkstationWorkflowTest`: Multi-screen navigation and database integrity tests passed

### B. Physical Phone Connected Tests (`.\gradlew.bat connectedDebugAndroidTest`)
- **Device Target:** Samsung Galaxy S24 (`SM-S921E - 16`)
- **Status:** **BUILD SUCCESSFUL** (65 actionable tasks executed)
- **Suite Report:** `android/app/build/reports/androidTests/connected/debug/org.crc.multimodal.CrcPhysicalOnDeviceParityTest.html`
  - `verifyPhysicalOnDeviceMsiParityTcgaD56540`: **PASSED** (0.073s)
  - `verifyPhysicalOnDeviceMssParityTcgaCm6675`: **PASSED** (0.062s)
  - Success Rate: **100% (2 tests, 0 failures, 0 skipped)**

### C. APK Deliverable Metadata
- **File Name:** `app-debug.apk`
- **File Path:** `android/app/build/outputs/apk/debug/app-debug.apk`
- **File Size:** `77,897,685` bytes (~74.29 MB)
- **SHA-256 Fingerprint:**
  `37EABE41D15E23FD9C050E719E6CCF848228359E4BD23D3C6E5217B95096B469`

---

## 7. Remaining Blockers & Next Action

- **Blockers for Phase 3:** **NONE.** All verification criteria, input pairing, ONNX runtime session setup, physical-device execution, and numerical parity tolerances ($\le 1.0\times 10^{-4}$) are fully validated.
- **Recommendations for Next Step (Phase 4):**
  - Proceed to Phase 4 (Mobile Optimization & Execution Provider Tuning): benchmark multi-threaded CPU configurations (1, 2, 4 threads), investigate NNAPI / Qualcomm QNN NPU delegates, and profile memory footprint.
  - Implement the UI file-ingestion parser for custom pathology tensor pairs (`.pt` / `.bin`) so researchers can analyze new patient specimens beyond benchmark datasets.
