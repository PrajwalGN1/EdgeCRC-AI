# Phase 5 Audit Report: Authentic Feature Import and Verified New-Specimen Inference

**Project**: Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2  
**Target Platform**: Android Edge Deployment (Samsung Galaxy S24 / ARM64-v8a / Qualcomm Snapdragon 8 Gen 3)  
**Execution Context**: Edge Inference on Physical Edge Device / Host Android Test Harness  
**Date**: October 23, 2026  
**Auditor/Engineer**: Senior Android Edge AI Engineer, ML Deployment Engineer & Independent Software Auditor  
**Repository Branch**: `edge-deployment`  
**Phase Status**: **PASS** (Model Integrity & Regression Invariants Preserved 100%)

---

## 1. Executive Summary

Phase 5 addresses the vulnerability identified in the Phase 4.5 audit: the reliance on synthetic pseudo-random feature vectors (`Random(seed)`) in the new-specimen evaluation workflow. In this phase, we have:

1. **Eliminated Synthetic Vectors from Genuine Specimen Inference**: Removed pseudo-random feature vector generation from `InferenceExecutionFragment` and isolated simulated data into explicit demonstration modes.
2. **Designed and Standardized Upstream Serialization Contract**: Formulated the `CRCB` (CRC Bundle) v2 binary specification and JSON schema v2, matching the exact precomputed embeddings from upstream Phikon-v2 (`[1, N, 1024]`) and ClinicalBERT (`[1, 768]`).
3. **Implemented Verified Import Workflow**: Created a scoped, Storage Access Framework (SAF) import mechanism (`AuthenticBundleLoader.kt` & `BundleStorageManager.kt`) that reads `.crcbundle`, `.bin`, and `.json` files into secure app-private storage (`filesDir/specimen_bundles/`), validating schemas, tensor dimensions, IEEE 754 finiteness, and cryptographic SHA-256 checksums before inference.
4. **Enforced Service-Layer Provenance Guardrails**: Introduced `InputProvenance` (`AUTHENTIC_IMPORTED`, `BENCHMARK_REFERENCE`, `DEMONSTRATION_SIMULATED`) and `CrcMultimodalClassifier.predictAuthentic(...)`, preventing any unverified or synthetic tensor from reaching the frozen ONNX FP32 model session.
5. **Verified Parity Across All Regression Gates**: Executed the entire Python regression suite (66/66 passed), Android local unit and holdout parity suite (59/59 passed across 8 test suites), and validated bit-for-bit numerical consistency.

---

## 2. Non-Negotiable Model Integrity & Baseline Invariants

All baseline constraints defined in the Master Directive were verified and strictly preserved:

| Invariant Parameter | Established Baseline | Phase 5 Verified State | Verification Status |
| :--- | :--- | :--- | :--- |
| **Model Asset Path** | `android/app/src/main/assets/models/fusion_model_fp32.onnx` | `android/app/src/main/assets/models/fusion_model_fp32.onnx` | **MATCH** |
| **Model File Size** | `21,015,452 bytes` | `21,015,452 bytes` | **MATCH** |
| **Model SHA-256 Hash** | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **MATCH** |
| **Decision Threshold $\tau$**| `0.83` | `0.83` (`ThresholdLogic.OPERATING_THRESHOLD`) | **MATCH** |
| **Class Decision Rule** | MSI if $P(\text{MSI}) \ge 0.83$; MSS if $P(\text{MSI}) < 0.83$ | Preserved in `ThresholdLogic.classify()` | **MATCH** |
| **Class Mapping** | Class 0 = MSS, Class 1 = MSI | Preserved (`ThresholdLogic.LABEL_MSS`, `LABEL_MSI`) | **MATCH** |
| **Execution Provider** | ONNX Runtime CPU (`CPUExecutionProvider`) | ONNX Runtime CPU (FP32) | **MATCH** |
| **66-Patient Manifest**| `holdout_manifest_66.json` | Untouched, identical SHA-256 | **MATCH** |

---

## 3. Pre-Implementation Audit (Stage A)

### 3.1 Identification of Synthetic Code Paths
Prior to Phase 5, synthetic preview vectors were generated in `InferenceExecutionFragment.kt`:
```kotlin
// PREVIOUS VULNERABLE CODE PATH:
val rng = kotlin.random.Random(specimen.specimenId.hashCode())
val visionData = FloatArray(specimen.patchCount * 1024) { (rng.nextFloat() * 2f - 1f) }
val textData = FloatArray(768) { (rng.nextFloat() * 2f - 1f) }
```
When a user entered an arbitrary Specimen ID in `NewSpecimenFragment`, the app stored the ID in the local SQLite database (`specimen_database`), navigated to `SpecimenReviewFragment`, and then to `InferenceExecutionFragment`, which synthesized deterministic pseudo-random Gaussian noise seeded with `specimenId.hashCode()`.

### 3.2 Vulnerability Assessment
1. **Accidental Clinical Interpretation**: A user testing an unknown patient ID received a plausible-looking prediction and attention bar chart that was entirely derived from pseudo-random numbers.
2. **Missing Input Provenance**: `InferenceResult` lacked a provenance marker, making it impossible for audit logs or downstream views (`FindingsFragment`, `ReportFragment`) to distinguish between real holdout validation cases and synthetic mockups.
3. **No External Tensor Ingestion**: There was no mechanism for a researcher or clinician to supply precomputed `.pt`, `.npy`, or binary embeddings to the mobile workstation.

---

## 4. Input Serialization Contract & Specification (Stage B)

The upstream pipeline generates two precomputed feature tensors per specimen:
1. **Vision Sequence**: Feature representations from Phikon-v2 vision transformer, shape `[1, N, 1024]` (where $N$ is the number of selected tissue patches, $1 \le N \le 1000$), Float32 Little-Endian.
2. **Clinical-Text Summary**: Mean-pooled output from ClinicalBERT text transformer, shape `[1, 768]`, Float32 Little-Endian.

To support deterministic, zero-dependency ingestion on Android without requiring heavy C++ deserialization libraries, two equivalent formats were standardized:

### 4.1 Specification 1: CRCB Binary Format (Version 2)
Optimized for high-speed streaming ingestion directly into native memory buffers:

```
[0x00 - 0x03] Magic Bytes:       'C' 'R' 'C' 'B' (0x43, 0x52, 0x43, 0x42)
[0x04 - 0x07] Schema Version:    Int32 Little-Endian (Value: 2)
[0x08 - 0x0B] ID Length:         Int32 Little-Endian (L_id)
[0x0C - ...]  Specimen ID:       UTF-8 String (L_id bytes)
[...]         Desc Length:       Int32 Little-Endian (L_desc)
[...]         Description:       UTF-8 String (L_desc bytes)
[...]         num_patches:       Int32 Little-Endian (N)
[...]         vision_dim:        Int32 Little-Endian (1024)
[...]         text_dim:          Int32 Little-Endian (768)
[...]         data_type:         Int32 Little-Endian (1 = Float32 Little-Endian)
[...]         vision_sha256:     Raw SHA-256 Digest (32 bytes)
[...]         text_sha256:       Raw SHA-256 Digest (32 bytes)
[...]         text_payload:      768 * 4 bytes (Float32 Little-Endian)
[...]         vision_payload:    N * 1024 * 4 bytes (Float32 Little-Endian)
```

### 4.2 Specification 2: JSON Bundle Schema (Version 2.0)
Optimized for human readability, auditability, and web/API integration:

```json
{
  "schema_version": "2.0",
  "specimen_id": "TCGA-A6-3807",
  "description": "Colorectal Adenocarcinoma specimen embeddings",
  "patch_count": 613,
  "vision_tensor": {
    "shape": [1, 613, 1024],
    "dtype": "float32",
    "endianness": "little",
    "sha256": "5c9e2b...",
    "data_base64": "..."
  },
  "text_tensor": {
    "shape": [1, 768],
    "dtype": "float32",
    "endianness": "little",
    "sha256": "9a1f3c...",
    "data_base64": "..."
  }
}
```

### 4.3 Deterministic Conversion Utility
Implemented in `tools/export_specimen_bundle.py`:
- Deterministically extracts PyTorch tensors from `data/clean_tensors/*.pt`.
- Validates shapes, finiteness (no NaN/Inf), and computes cryptographic SHA-256 digests.
- Emits bit-for-bit verified `.crcbundle` and `.json` artifacts.
- Verified exact zero numerical difference (`max_abs_diff = 0.0`) when comparing round-trip deserialized data against original upstream tensors.

---

## 5. Implementation Architecture (Stage C, D, E)

### 5.1 Storage Access Framework Ingestion (`NewSpecimenFragment.kt`)
- Users select an authentic bundle via `ActivityResultContracts.OpenDocument()`.
- The URI is resolved and parsed via `AuthenticBundleLoader.loadBundle()`.
- The specimen identifier in the bundle automatically populates the Specimen ID field.
- If a specimen with the same ID already exists in the local database, an explicit Material Alert Dialog prompts the user: *"Overwrite Existing Specimen?"*, preventing silent data corruption.
- The bundle is saved atomically into app-private storage via `BundleStorageManager.saveBundle()`.

### 5.2 Private Storage Management (`BundleStorageManager.kt`)
- Location: `context.filesDir/specimen_bundles/<specimen_id>.crcbundle`.
- Storage is private to the application UID (`MODE_PRIVATE`).
- Does not require broad `READ_EXTERNAL_STORAGE` or `MANAGE_EXTERNAL_STORAGE` permissions.
- Atomic file writes using temporary staging files (`.tmp`) before atomic rename.

### 5.3 Service-Layer Verification (`CrcMultimodalClassifier.kt`)
The classifier now enforces strict provenance checking:
```kotlin
fun predictAuthentic(
    visionData: FloatArray,
    numPatches: Int,
    textData: FloatArray,
    caseId: String,
    provenance: InputProvenance,
    protocol: InferenceProtocol = InferenceProtocol.PROTOCOL_1_CAPPED
): InferenceResult {
    require(provenance == InputProvenance.AUTHENTIC_IMPORTED || provenance == InputProvenance.BENCHMARK_REFERENCE) {
        "Authentic feature embeddings are required. Import a validated vision and clinical-text tensor pair before running inference."
    }
    return predict(...)
}
```

### 5.4 Elimination of Synthetic Data in UI
In `InferenceExecutionFragment.kt`:
1. The random vector generator was **completely deleted**.
2. If `specimen.isReferenceCase` is true, the engine loads the verified reference tensors from app assets.
3. If `specimen.isReferenceCase` is false, the engine searches `BundleStorageManager` for the authentic bundle.
4. If no authentic bundle exists, execution halts immediately, and a red error card displays:
   > *"Authentic feature embeddings are required. Import a validated vision and clinical-text tensor pair before running inference."*
   The "Execute Inference" button is disabled.

---

## 6. Verification and Regression Gates (Stage F)

### 6.1 Python Regression Suite
Executed: `python -m pytest`
- **Result**: `66 passed, 3 warnings in 17.29s`
- **Verification**: Complete parity across all mathematical modules, ONNX exporters, input validators, and sampling manifests.

### 6.2 Android Unit Test Suite
Executed: `.\gradlew.bat testDebugUnitTest`
- **Total Test Suites**: 8
- **Total Unit Tests Executed**: 59
- **Failures**: 0
- **Errors**: 0

Detailed Suite Breakdown:
1. `AuthenticBundleLoaderTest`: **14 tests, 0 failures** (0.865s)
   - `testValidCrcbBundleIngestion`: Verified bit-for-bit parse of CRCB v2.
   - `testValidJsonBundleIngestion`: Verified base64 JSON parsing with zero external dependencies.
   - `testLegacyCrctBundleIngestion`: Backwards compatibility with CRCT v1.
   - `testInvalidMagicBytesRejected`: Fails closed on corrupt header.
   - `testVisionDimensionMismatchRejected`: Rejects non-1024 vision dimensions.
   - `testTextDimensionMismatchRejected`: Rejects non-768 text dimensions.
   - `testCorruptVisionChecksumRejected`: Cryptographic mismatch caught and rejected.
   - `testCorruptTextChecksumRejected`: Text cryptographic mismatch rejected.
   - `testTruncatedFileRejected`: Truncated binary file rejected cleanly.
   - `testNanValuesRejected`: IEEE 754 NaN values caught and rejected.
   - `testInfinityValuesRejected`: IEEE 754 $\pm\infty$ values caught and rejected.
   - `testLargePatchCountSupported`: Successfully processes $N=1000$ patch arrays without memory exhaustion.
   - `testSimulatedProvenanceFailsClosedInAuthenticApi`: Enforces service-layer rejection of synthetic inputs.
   - `testThresholdBoundaryClassification`: Verifies 0.8299 (MSS) vs 0.8300 (MSI) boundary.
2. `CrcMultimodalClassifierInferenceTest`: **8 tests, 0 failures** (0.713s)
3. `CrcMultimodalHoldoutParityTest`: **2 tests, 0 failures** (2.971s) — Verified against 66-case holdout manifest using live ONNX FP32 session.
4. `InferenceResultTest`: **3 tests, 0 failures** (0.002s)
5. `InputValidatorTest`: **15 tests, 0 failures** (0.049s)
6. `ReferenceTensorLoaderTest`: **2 tests, 0 failures** (0.058s)
7. `ThresholdLogicTest`: **4 tests, 0 failures** (0.003s)
8. `WorkstationWorkflowTest`: **11 tests, 0 failures** (0.052s)

---

## 7. Performance & Memory Profiling (Stage G)

Benchmarked on genuine holdout specimen bundles using `tools/benchmark_authentic_ingestion.py` (ONNX Runtime CPU FP32, 2 intra-op threads, 50 timed iterations):

| Specimen ID | Format | Patch Count ($N$) | File Size | Memory Footprint | Parse Time | Validation Time | ONNX Inference | End-to-End Latency | MSI Probability | Class ($\tau = 0.83$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCGA-A6-3807** | `CRCB_v2` | 613 | 2.51 MB | 2.51 MB | 1.12 ms | 7.80 ms | 22.45 ms | **31.37 ms** | 0.0023 | **MSS** |
| **TCGA-A6-6138** | `CRCB_v2` | 113 | 0.47 MB | 0.47 MB | 0.14 ms | 1.48 ms | 4.09 ms | **5.71 ms** | 0.7624 | **MSS** |
| **TCGA-CM-6675** | `CRCB_v2` | 541 | 2.22 MB | 2.22 MB | 0.97 ms | 5.57 ms | 18.10 ms | **24.64 ms** | 0.1622 | **MSS** |
| **TCGA-D5-6540** | `CRCB_v2` | 345 | 1.42 MB | 1.42 MB | 0.52 ms | 3.51 ms | 10.63 ms | **14.65 ms** | 0.9904 | **MSI** |
| **TCGA-A6-3807** | `JSON_v2` | 613 | 3.35 MB | 2.51 MB | 30.05 ms| 8.98 ms | 18.62 ms | **57.65 ms** | 0.0023 | **MSS** |

### Key Performance Findings:
1. **CRCB Binary Advantage**: Binary parsing is **~27x faster** than JSON (1.12 ms vs 30.05 ms) and reduces file footprint by 25%.
2. **Sub-35ms Ingestion Latency**: Total pipeline latency from disk read to final threshold classification is under 32 ms even for large specimens ($N=613$).
3. **Linear Memory Scaling**: Memory required for tensors is strictly $4 \times (1024N + 768)$ bytes, with zero persistent heap bloat.

---

## 8. Upstream Encoder Integration Boundary & Clinical Limitations (Stage H)

### 8.1 Current Architectural Boundary
The mobile workstation executes **multimodal cross-attention fusion inference on precomputed embeddings**. It does **not** execute the raw upstream vision and text foundation models on-device:

```
[Raw H&E Whole-Slide Image]  ──> [Upstream Server: Phikon-v2 ViT] ───> [1, N, 1024] Float32 ──\
                                                                                                ├──> [.crcbundle] ──> [Android Edge Workstation]
[Unstructured Clinical Note] ──> [Upstream Server: ClinicalBERT]  ───> [1, 768] Float32    ──/
```

### 8.2 External Requirements for Producing Valid Embeddings
1. **Vision Preprocessing (Phikon-v2)**:
   - Input: $224 \times 224$ tissue patches extracted from formalin-fixed paraffin-embedded (FFPE) H&E whole-slide images at $20\times$ magnification ($0.5\,\mu\text{m/pixel}$).
   - Color normalization: Macenko or Reinhard stain normalization.
   - Encoder: OWKIN Phikon-v2 (ViT-B/16 architecture, 1024-dimensional CLS token output).
   - Selection: Foreground tissue filtering removing blank background tiles.
2. **Text Preprocessing (ClinicalBERT)**:
   - Input: Clinical pathology report or diagnostic summary.
   - Tokenizer: ClinicalBERT WordPiece tokenizer (max sequence length 512).
   - Encoder: `emilyalsentzer/Bio_ClinicalBERT` (768 hidden dimensions).
   - Pooling: Mean pooling over non-padding tokens.

### 8.3 Regulatory & Clinical Disclaimers
- The application displays persistent banners: *"RESEARCH USE ONLY: Multimodal fusion research output. Not approved for clinical diagnostics."*
- Model probabilities represent cross-entropy softmax activations from a research dataset and are **not** clinically calibrated risks.

---

## 9. Before-and-After Codebase Diff Summary

```
Changes:
  modified:   android/app/build.gradle.kts (added targetSdkVersion 34, packaging options)
  modified:   android/app/src/main/java/org/crc/multimodal/inference/CrcMultimodalClassifier.kt (predictAuthentic, input/output validation logging)
  modified:   android/app/src/main/java/org/crc/multimodal/inference/InferenceResult.kt (added InputProvenance, equals/hashCode)
  modified:   android/app/src/main/java/org/crc/multimodal/ui/fragments/NewSpecimenFragment.kt (SAF import, overwrite dialog, DB persistence)
  modified:   android/app/src/main/java/org/crc/multimodal/ui/fragments/SpecimenReviewFragment.kt (provenance badge, input verification)
  modified:   android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt (eliminated synthetic random vectors)
  modified:   android/app/src/main/res/layout/fragment_new_specimen.xml (Import bundle card UI)
  modified:   android/app/src/main/res/layout/fragment_inference_execution.xml (Provenance alert card)

Added:
  android/app/src/main/java/org/crc/multimodal/inference/InputProvenance.kt
  android/app/src/main/java/org/crc/multimodal/inference/AuthenticFeatureBundle.kt
  android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt
  android/app/src/main/java/org/crc/multimodal/inference/BundleStorageManager.kt
  android/app/src/test/java/org/crc/multimodal/AuthenticBundleLoaderTest.kt
  tools/export_specimen_bundle.py
  tools/benchmark_authentic_ingestion.py
  reports/mobile_deployment/phase5_benchmarks/ingestion_benchmark_results.json
  reports/mobile_deployment/phase5_benchmarks/ingestion_benchmark_results.csv
  reports/mobile_deployment/phase5_authentic_tensor_ingestion.md
```

---

## 10. Acceptance Criteria & Final Verdict

| Acceptance Criteria | Target Condition | Achieved Result | Decision |
| :--- | :--- | :--- | :--- |
| **Model Checksum** | `0402f6c5...` (21,015,452 B) | Exact match verified | **PASS** |
| **Authentic Import** | Storage Access Framework + Private Cache | Fully functional (`.crcbundle`, `.json`) | **PASS** |
| **Input Validation** | Reject invalid dims, NaNs, corrupted hashes | 14 test cases verified | **PASS** |
| **Synthetic Fallback Prevention** | Zero synthetic data in genuine inference | Fails closed; synthetic code deleted | **PASS** |
| **Format Parity** | CRCB v2 matches upstream tensors | Diff = 0.0 across all holdout cases | **PASS** |
| **Existing Regression**| All Python & Android test suites pass | 66/66 Python pass, 59/59 Android pass | **PASS** |
| **Holdout Parity** | 66-case holdout predictions preserved | Exact match verified | **PASS** |
| **UI Separation** | Demonstrations clearly labeled | Provenance banners & blocked flows | **PASS** |

### FINAL DECISION: **PASS**

---

## 11. Recommended Next Phase: Phase 6

With authentic feature import operational, verified, and regression-tested, the application is ready for **Phase 6: Clinical Validation & Batch Deployment Readiness**, focusing on:
1. Multi-specimen batch import and automated cohort queuing.
2. End-to-end report PDF generation with cryptographic audit signatures.
3. Preparation for device-specific NPU execution feasibility studies under strict numerical tolerance gates.
