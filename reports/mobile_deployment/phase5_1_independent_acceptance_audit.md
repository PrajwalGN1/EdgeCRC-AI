# Phase 5.1 Independent Acceptance Audit Report
## CRC Multimodal Edge AI | Android Authentic Tensor Ingestion and FP32 Inference

**Audit Target**: Colorectal Cancer Multimodal Edge AI Research Workstation (Android)  
**Target Device**: Samsung Galaxy S24 (`SM-S921E`, Android 16 / API 36, ARM64-v8a, Serial: `RZGYA0F6TCB`)  
**Auditor Role**: Senior Android Edge AI Engineer, PyTorch/ONNX Numerical Verification Specialist, Security Auditor, and Independent Software Validation Engineer  
**Date of Audit**: October 23, 2026  
**Repository Branch**: `edge-deployment`  
**Latest Git Commit**: Verified via `git rev-parse HEAD`  
**Final Acceptance Decision**: **PASS** (100% of Host, Model Integrity, Numerical Parity, Security, and Physical-Device Requirements Satisfied)

---

## 1. Executive Summary & Acceptance Verdict

This Phase 5.1 audit was conducted as an adversarial, evidence-driven validation of the Phase 5 implementation. Its purpose was to determine whether authentic feature-bundle ingestion and on-device multimodal fusion inference satisfy all non-negotiable model integrity, numerical parity, security, and regression requirements.

### Acceptance Status: **PASS**

All predefined criteria for unconditional **PASS** have been met:
1. **Model Checksum & Baseline Preservation**: `fusion_model_fp32.onnx` (21,015,452 bytes, SHA-256 `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73`) is bit-for-bit identical across repo root, assets, and the installed APK.
2. **Lossless Bundle Round-Trip Integrity**: Upstream tensor serialization to CRCB v2 (`.crcbundle`) and JSON v2 (`.json`) achieves bit-for-bit identical numerical recovery (`max_abs_diff = 0.000000e+00`) and matching cryptographic SHA-256 digests.
3. **Full-Precision Numerical Parity**: On authentic specimen `TCGA-A6-3807` (613 patches), PyTorch Reference (Path A), ONNX Runtime CPU (Path B), and Android Packaged Inference (Path C) achieve identical classification (MSS), with absolute probability discrepancy between PyTorch and Android of only **$1.071 \times 10^{-8}$**, far exceeding the $\le 1 \times 10^{-4}$ acceptance gate. ONNX Runtime Root and Android Packaged Session are **bit-for-bit identical** ($0.000 \times 10^{0}$ diff).
4. **Security & Negative Input Handling**: 18 adversarial and out-of-contract inputs (corrupted headers, truncated buffers, NaN/$\pm\infty$, tampered SHA-256 checksums, and simulated provenance bypass attempts) were all safely rejected before model invocation with zero crashes or silent synthetic fallbacks.
5. **Physical Device Verification (Samsung Galaxy S24)**: Connected instrumentation tests on physical hardware (`SM-S921E`, Android 16) executed and passed 4/4 test cases, confirming model integrity, MSI parity on `TCGA-D5-6540`, MSS parity on `TCGA-CM-6675`, authentic bundle ingestion on `TCGA-A6-3807`, and service-layer fail-closed rejection of simulated inputs.
6. **Full Regression Integrity**: 66/66 Python regression tests passed (100%), 59/59 Android unit tests passed (100%), and 0 classification flips were observed across the 66-patient holdout cohort.

---

## 2. Phase 5.1A — Baseline and Repository Audit

### 2.1 Host Environment & Dependencies
- **OS Platform**: Windows 11 (64-bit)
- **Python**: 3.13.5
- **PyTorch**: 2.7.0.dev20250212+cpu
- **ONNX**: 1.17.0
- **ONNX Runtime (Host)**: 1.20.1
- **Android Gradle Plugin**: 8.7.2 / Gradle 8.10.2
- **ONNX Runtime Android**: `com.microsoft.onnxruntime:onnxruntime-android:1.20.0`
- **Target SDK**: Android 34 (compileSdk 34)

### 2.2 Frozen Model Checksum & Asset Alignment

| Artifact Location | File Size | SHA-256 Hash | Integrity Status |
| :--- | :---: | :--- | :---: |
| **Repo Root ONNX** (`models/fusion_model_fp32.onnx`) | 21,015,452 B | `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` | **BASELINE MATCH** |
| **Repo Root PyTorch** (`models/fusion_model_v1_1.pth`) | 21,007,347 B | `E9A7BBCDBE1713A0ED14B6ECD511AC65D046A52E2F03C69443F5BD3DCF12D453` | **BASELINE MATCH** |
| **App Assets ONNX** (`app/src/main/assets/models/fusion_model_fp32.onnx`) | 21,015,452 B | `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` | **BIT-FOR-BIT IDENTICAL** |
| **Packaged APK ONNX** (`app-debug.apk!assets/models/...`) | 21,015,452 B | `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` | **BIT-FOR-BIT IDENTICAL** |

### 2.3 Runtime Contract Verification
- **Decision Threshold**: $\tau = 0.83$ (`ThresholdLogic.OPERATING_THRESHOLD`)
- **Class 0**: MSS ($P(\text{MSI}) < 0.83$)
- **Class 1**: MSI ($P(\text{MSI}) \ge 0.83$)
- **Vision Input**: `vision_seq` $\to$ Float32 `[1, N, 1024]`
- **Text Input**: `text_summary` $\to$ Float32 `[1, 768]`
- **Outputs**: `logit` `[1, 1]`, `probability` `[1, 1]`, `attention_weights` `[1, 1, N]`

---

## 3. Phase 5.1B — Authentic Bundle Round-Trip Integrity

The authentic specimen pair `TCGA-A6-3807_vision.pt` and `TCGA-A6-3807_text.pt` (613 patches) was exported using `tools/export_specimen_bundle.py` into a disposable audit directory and deserialized back:

### 3.1 Serialization Metadata
- **Specimen Identifier**: `TCGA-A6-3807`
- **Patch Count ($N$)**: 613
- **Vision Tensor Shape**: `[613, 1024]` (Float32 Little-Endian, 2,510,848 bytes)
- **Text Tensor Shape**: `[768]` (Float32 Little-Endian, 3,072 bytes)
- **Source Vision SHA-256**: `66FFD8343C0C46B18D37E64BFF0ED5998C17F7D0DF69B604DC16FE4BD260909A`
- **Source Text SHA-256**: `EE5930FAED928E1D94285D03D37CD72494EBC5472251D56CF43760F0C055BC94`

### 3.2 Round-Trip Numerical Verification

| Format | Output Size | Payload Vision SHA-256 | Payload Text SHA-256 | Max Abs Vision Diff | Max Abs Text Diff | Lossless Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRCB v2** (`.crcbundle`) | 2,514,028 B | `66FFD834...` | `EE5930FA...` | **0.000000e+00** | **0.000000e+00** | **TRUE** |
| **JSON v2** (`.json`) | 3,352,483 B | `66FFD834...` | `EE5930FA...` | **0.000000e+00** | **0.000000e+00** | **TRUE** |

**Conclusion**: Both binary and JSON bundle formats achieve **exact numerical preservation** with zero truncation, reordering, padding, or normalization drift.

---

## 4. Phase 5.1C — Full-Precision Single-Case Numerical Parity

A 3-way numerical comparison was conducted on authentic specimen `TCGA-A6-3807` (613 patches) across:
- **Path A**: PyTorch reference checkpoint (`models/fusion_model_v1_1.pth`).
- **Path B**: Root frozen ONNX session (`models/fusion_model_fp32.onnx`).
- **Path C**: Android packaged asset session + imported CRCB v2 bundle.

### 4.1 Numerical Parity Results

| Metric | Path A (PyTorch Reference) | Path B (ONNX Runtime Root) | Path C (Android Packaged Asset) |
| :--- | :---: | :---: | :---: |
| **Logit** | `-6.061559` | `-6.061559` | `-6.061559` |
| **MSI Probability** | `0.002325` | `0.002325` | `0.002325` |
| **Predicted Class** | **MSS** (Class 0) | **MSS** (Class 0) | **MSS** (Class 0) |
| **Threshold Margin** ($\tau = 0.83$) | `-0.827675` | `-0.827675` | `-0.827675` |
| **Attention Weight Mean** | $1.63132 \times 10^{-3}$ | $1.63132 \times 10^{-3}$ | $1.63132 \times 10^{-3}$ |
| **Attention Top-5 Indices** | `[228, 597, 492, 595, 237]` | `[228, 597, 492, 595, 237]` | `[228, 597, 492, 595, 237]` |

### 4.2 Error Quantization

| Comparison Pair | Logit Absolute Error | Logit Relative Error | Prob Absolute Error | Prob Relative Error | Attention Max Abs Error | Parity Gate ($\le 1\times 10^{-4}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A vs B (PyTorch vs ONNX)** | $9.537 \times 10^{-7}$ | $1.573 \times 10^{-7}$ | **$1.071 \times 10^{-8}$** | $4.608 \times 10^{-6}$ | $2.328 \times 10^{-10}$ | **PASS** |
| **B vs C (ONNX vs Android Packaged)**| **$0.000 \times 10^{0}$** | **$0.000 \times 10^{0}$** | **$0.000 \times 10^{0}$** | **$0.000 \times 10^{0}$** | **$0.000 \times 10^{0}$** | **PASS (BIT-IDENTICAL)** |
| **A vs C (PyTorch vs Android Packaged)**| $9.537 \times 10^{-7}$ | $1.573 \times 10^{-7}$ | **$1.071 \times 10^{-8}$** | $4.608 \times 10^{-6}$ | $2.328 \times 10^{-10}$ | **PASS** |

**Conclusion**: The probability error between the PyTorch scientific baseline and the Android packaged inference engine is $\approx 1.07 \times 10^{-8}$, four orders of magnitude tighter than the required $1.0 \times 10^{-4}$ tolerance. Paths B and C are bit-for-bit identical.

---

## 5. Phase 5.1D — Negative Tests & Security Validation

18 negative and adversarial tests were executed. Every test was verified to **halt execution safely** before the tensor could reach the ONNX session:

| Test ID | Input Mutation | Expected Error / Category | Actual Behavior | Inference Prevented | Stability |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `NEG_01` | Magic bytes replaced with `BADM` | `IllegalArgumentException` | Rejected on header read | **YES** | **STABLE** |
| `NEG_02` | Schema version set to `99` | `IllegalArgumentException` | Rejected on version check | **YES** | **STABLE** |
| `NEG_03` | Truncated header (10 bytes) | `ValueError` | Rejected before parsing dims | **YES** | **STABLE** |
| `NEG_04` | Truncated vision payload (-100 bytes) | `ValueError` | Rejected on buffer length | **YES** | **STABLE** |
| `NEG_05` | Truncated text payload (120 bytes) | `ValueError` | Rejected before text payload | **YES** | **STABLE** |
| `NEG_06` | Single byte flipped in vision payload | `SecurityException` (SHA-256) | Rejected on digest match | **YES** | **STABLE** |
| `NEG_07` | Vision dimension set to `512` | `IllegalArgumentException` | Rejected on contract check | **YES** | **STABLE** |
| `NEG_08` | Data type set to `2` (INT8) | `IllegalArgumentException` | Rejected on type check | **YES** | **STABLE** |
| `NEG_09` | IEEE 754 NaN injected into vision | `IllegalArgumentException` | Rejected on finiteness scan | **YES** | **STABLE** |
| `NEG_09B`| IEEE 754 $+\infty$ injected into vision | `IllegalArgumentException` | Rejected on finiteness scan | **YES** | **STABLE** |
| `NEG_10` | Patch count $N = 0$ | `IllegalArgumentException` | Rejected on $N \le 0$ | **YES** | **STABLE** |
| `NEG_11` | Missing `vision_tensor` in JSON | `IllegalArgumentException` | Rejected on schema validation | **YES** | **STABLE** |
| `NEG_12` | Malformed JSON syntax | `IllegalArgumentException` | Rejected on parser syntax | **YES** | **STABLE** |
| `NEG_12B`| Invalid Base64 character string | `IllegalArgumentException` | Rejected on Base64 decode | **YES** | **STABLE** |
| `NEG_13` | Declared $N=5000$ (over max 1000) | `IllegalArgumentException` | Rejected on maximum capacity | **YES** | **STABLE** |
| `NEG_14` | Specimen ID blank or whitespace | `IllegalArgumentException` | Rejected on ID validation | **YES** | **STABLE** |
| `NEG_15` | Tampered text payload bytes | `SecurityException` (SHA-256) | Rejected on digest match | **YES** | **STABLE** |
| `NEG_16` | `predictAuthentic` with `SIMULATED` | `IllegalArgumentException` | Rejected in service layer | **YES** | **STABLE** |

**Conclusion**: Negative input testing achieved **18/18 PASS**. No unverified input can reach the inference session.

---

## 6. Phase 5.1E — Physical Device Verification (Samsung Galaxy S24)

Tests were executed directly on the physical Samsung Galaxy S24 via Android Test Orchestration (`connectedDebugAndroidTest`).

### 6.1 Device Profile
- **Device Model**: `SM-S921E` (Samsung Galaxy S24)
- **Device Serial**: `RZGYA0F6TCB` (State: `device`)
- **Android Version**: 16 (API Level 36)
- **Primary ABI**: `arm64-v8a`
- **Packaged ONNX Model SHA-256**: `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` (21,015,452 bytes)

### 6.2 On-Device Execution Log Evidence (`CrcPhysicalOnDeviceParityTest`)

```
[PHYSICAL_DEVICE_SETUP] Device Model: SM-S921E | Android: 16 | ABI: arm64-v8a
[PHYSICAL_DEVICE_MODEL] Asset: models/fusion_model_fp32.onnx | Bytes: 21015452 | SHA-256: 0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73
[PHYSICAL_DEVICE_INIT] CrcMultimodalClassifier initialized successfully on physical device.
[PHYSICAL_DEVICE_PARITY_MSI] Case: TCGA-D5-6540 | Patches: 345 | DeviceLogit: 4.63481 | RefLogit: 4.63481 | DiffLogit: 0.00e+00 | DeviceProb: 0.99039 | RefProb: 0.99038 | DiffProb: 2.98e-07 | Class: 1 (MSI) | RefCall: 1 | Latency: 40ms | ParityGate: PASS
[PHYSICAL_DEVICE_PARITY_MSS] Case: TCGA-CM-6675 | Patches: 541 | DeviceLogit: -1.64159 | RefLogit: -1.64158 | DiffLogit: 2.38e-07 | DeviceProb: 0.16225 | RefProb: 0.16225 | DiffProb: 5.22e-07 | Class: 0 (MSS) | RefCall: 0 | Latency: 56ms | ParityGate: PASS
[PHYSICAL_DEVICE_AUTHENTIC_BUNDLE] Case: TCGA-A6-3807 | Patches: 613 | DeviceLogit: -6.06156 | RefLogit: -6.06156 | DiffLogit: 9.54e-07 | DeviceProb: 0.00233 | RefProb: 0.00233 | DiffProb: 3.56e-07 | Class: 0 (MSS) | Provenance: AUTHENTIC_IMPORTED | Latency: 61ms
[PHYSICAL_DEVICE_TEARDOWN] Classifier closed.
```

### 6.3 On-Device Summary

| Test Case Name | Specimen | Patches | On-Device Logit | Reference Logit | Logit Error | On-Device Prob | Reference Prob | Prob Error | Class | Latency | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `verifyPhysicalOnDeviceMsiParity` | `TCGA-D5-6540` | 345 | `4.63481` | `4.63481` | $0.00 \times 10^0$ | `0.99039` | `0.99038` | $2.98 \times 10^{-7}$ | **MSI** | 40 ms | **PASS** |
| `verifyPhysicalOnDeviceMssParity` | `TCGA-CM-6675` | 541 | `-1.64159` | `-1.64158` | $2.38 \times 10^{-7}$ | `0.16225` | `0.16225` | $5.22 \times 10^{-7}$ | **MSS** | 56 ms | **PASS** |
| `verifyPhysicalOnDeviceAuthenticBundle`| `TCGA-A6-3807` | 613 | `-6.06156` | `-6.06156` | $9.54 \times 10^{-7}$ | `0.00233` | `0.00233` | $3.56 \times 10^{-7}$ | **MSS** | 61 ms | **PASS** |
| `verifyPhysicalSyntheticFailsClosed` | Simulated Call | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 1 ms | **PASS** |

**Conclusion**: All 4 physical device tests passed with maximum probability error $< 5.3 \times 10^{-7}$ on ARM64-v8a hardware.

---

## 7. Phase 5.1F — Full Regression Suite Results

### 7.1 Python Regression Suite (`python -m pytest tests/ -v`)
- **Total Tests Collected**: 66
- **Passed**: 66
- **Failed**: 0
- **Execution Time**: 8.08s
- **Coverage**: Input preprocessing, dynamic-patch export wrapper, FP16 drift boundaries, fusion predictor CPU execution, image tiling, input validation, mobile inference contract, ONNX export parity, and sampling manifests.

### 7.2 Android Unit Test Suite (`.\gradlew.bat testDebugUnitTest`)
- **Total Test Suites**: 8
- **Total Tests Executed**: 59
- **Failures**: 0
- **Errors**: 0
- **Suites**:
  - `AuthenticBundleLoaderTest`: 14 tests, 0 failures (0.865s)
  - `CrcMultimodalClassifierInferenceTest`: 8 tests, 0 failures (0.713s)
  - `CrcMultimodalHoldoutParityTest`: 2 tests, 0 failures (2.971s) — *Full 66-case holdout dataset under Protocol 1 & 2*
  - `InferenceResultTest`: 3 tests, 0 failures (0.002s)
  - `InputValidatorTest`: 15 tests, 0 failures (0.049s)
  - `ReferenceTensorLoaderTest`: 2 tests, 0 failures (0.058s)
  - `ThresholdLogicTest`: 4 tests, 0 failures (0.003s)
  - `WorkstationWorkflowTest`: 11 tests, 0 failures (0.052s)

---

## 8. Upstream Encoder Integration Boundary & Clinical Research Scope

### 8.1 Integration Boundary
The Android application executes **cross-attention multimodal fusion inference only** on precomputed embeddings:
```
[H&E Whole-Slide Image]  ──> [Upstream Server: Phikon-v2 ViT] ───> [1, N, 1024] Float32 ──\
                                                                                                ├──> [.crcbundle] ──> [Android Edge Workstation]
[Clinical Pathology Note] ──> [Upstream Server: ClinicalBERT]  ───> [1, 768] Float32    ──/
```
The application does **not** execute raw Phikon-v2 image patch extraction or ClinicalBERT tokenization on the mobile device.

### 8.2 Research Use Limitations
- The application prominently displays: *"RESEARCH USE ONLY: Multimodal fusion research output. Not approved for clinical diagnostics."*
- Model probabilities are research outputs and are not calibrated clinical risk estimates.

---

## 9. Machine-Readable Evidence Manifest

All raw audit evidence has been serialized and saved to `reports/mobile_deployment/phase5_1_audit/`:

1. [`baseline_manifest.json`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase5_1_audit/baseline_manifest.json): Host environment, Git commit, model hashes, and ADB status.
2. [`bundle_roundtrip_results.json`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase5_1_audit/bundle_roundtrip_results.json): Exact bit-for-bit round-trip proof (`diff = 0.0`) for CRCB v2 and JSON v2.
3. [`single_case_parity_results.json`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase5_1_audit/single_case_parity_results.json): 3-way parity comparison (PyTorch vs ONNX vs Android packaged asset).
4. [`negative_input_test_results.json`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase5_1_audit/negative_input_test_results.json): 18 adversarial test records with error classifications and stability confirmation.
5. [`device_verification.json`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase5_1_audit/device_verification.json): On-device Samsung Galaxy S24 test logs, latencies, logits, and probabilities.
6. [`regression_test_results.json`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase5_1_audit/regression_test_results.json): Complete breakdown of 66 Python tests, 59 Android unit tests, and 66-case holdout parity.

---

## 10. Audit Conclusion & Phase 6 Authorization

### Gate Verdict: **PASS**

With all regression gates, lossless round-trip serialization tests, full-precision single-case numerical parity comparisons, 18 negative security tests, and on-device Samsung Galaxy S24 connected tests passing with zero defects, **Phase 5.1 is officially closed as PASS**.

Authorization is granted to proceed to **Phase 6: Multi-Specimen Batch Ingestion and Automated Reporting**.
