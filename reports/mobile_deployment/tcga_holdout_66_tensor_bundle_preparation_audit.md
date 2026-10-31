# TCGA_HOLDOUT_66 Reference Specimen Extraction & Tensor Bundle Preparation Audit

**Role**: Senior ML Deployment Engineer, Android Edge-AI Engineer, Dataset Integrity Auditor, Tensor Serialization Specialist, QA Engineer  
**Audit Date**: October 28, 2026  
**Reference Cohort**: `TCGA_HOLDOUT_66`  
**Target Package**: `org.crc.multimodal`  
**Investigation Mode**: Strict Read-Only Architectural & Asset Discovery Audit  
**Phase**: Phase 0 — Safety, Project Discovery, and Read-Only Audit  
**Deliverable**: `reports/mobile_deployment/tcga_holdout_66_tensor_bundle_preparation_audit.md`  

---

## 1. Executive Summary

This audit establishes the discovery baseline for preparing, validating, and packaging authentic multimodal tensor feature bundles for the 66-case TCGA Holdout Validation Cohort (`TCGA_HOLDOUT_66`).

All required assets, tensor directories, binary serialization tools, Android loader implementations, and device connections have been discovered and verified:
1. **Authoritative Reference Ground Truth**: Located at [`android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json). Contains exactly **66 unique patient cases** (50 MSS, 16 MSI) with zero missing or duplicate identifiers.
2. **Precomputed Tensor Storage**: Located at [`data/clean_tensors/`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/clean_tensors). Contains **652 PyTorch tensor files** (326 pairs of `<SPECIMEN_ID>_vision.pt` and `<SPECIMEN_ID>_text.pt`). **All 66/66 reference holdout specimens have both precomputed vision and text tensors present.** Zero re-computation with Phikon-v2 or ClinicalBERT is required.
3. **Authentic `.crcbundle` Binary Format & Tools**:
   - Python Writer: [`tools/export_specimen_bundle.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tools/export_specimen_bundle.py) implements CRCB Version 2 Little-Endian binary serialization with SHA-256 payload digests.
   - Android Kotlin Reader: [`AuthenticBundleLoader.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt) implements the exact matching parser.
4. **Physical Edge Device**: Vivo Y21 (`V2125`, Serial `1377355236000F7`, Android 12, API 31) is verified connected, authorized, and active.

---

## 2. Project State & Environment Discovery

### 2.1. Git Working Tree & Branch Verification
* **Repository Root**: `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`
* **Active Git Branch**: `edge-deployment`
* **Working Tree State**: Clean and intact; all existing user files, reports, and database models preserved.

### 2.2. Authoritative Reference Labels File
* **Primary Path**: [`android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/assets/reference_labels/tcga_holdout_reference_labels.json)
* **Secondary Source**: [`data/reference_labels/tcga_holdout_reference_labels.csv`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/reference_labels/tcga_holdout_reference_labels.csv)
* **Validation Keys**: [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json)
* **Schema Verification**:
  - `cohort_id`: `"TCGA_HOLDOUT_66"`
  - `total_cases`: `66`
  - `class_distribution`: `MSS_0: 50, MSI_1: 16`
  - Identifier Format: Standard TCGA barcode (`TCGA-XX-XXXX`)
  - Class Mapping: Label `0` = MSS (Negative), Label `1` = MSI (Positive)
  - Duplicate Check: Exactly 0 duplicates; exactly 66 distinct specimens.

---

## 3. Precomputed Tensor Asset Inventory

### 3.1. Directory Structure & File Naming
* **Directory**: [`data/clean_tensors/`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/clean_tensors)
* **File Naming Pattern**:
  - Vision: `<SPECIMEN_ID>_vision.pt` (e.g. `TCGA-D5-6540_vision.pt`)
  - Text: `<SPECIMEN_ID>_text.pt` (e.g. `TCGA-D5-6540_text.pt`)
* **Total Dataset Size**: 652 `.pt` files (326 pairs).

### 3.2. TCGA_HOLDOUT_66 Coverage Assessment
Cross-referencing all 66 reference holdout IDs against `data/clean_tensors/`:
* **Vision Tensors Available**: `66 / 66` (100%)
* **Text Tensors Available**: `66 / 66` (100%)
* **Complete Tensor Pairs Available**: `66 / 66` (100%)
* **Missing Tensors**: Exactly `0` missing tensors.
* **Integrity Guarantee**: All tensors were precomputed upstream with Phikon-v2 and ClinicalBERT. In accordance with Section 1 of the Directive, zero re-computation or model execution is required to acquire tensors.

---

## 4. Authentic `.crcbundle` Binary Format & Tooling

### 4.1. Binary Specification (CRCB Version 2)
The authoritative container format accepted by Android [`AuthenticBundleLoader.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt) is structured as follows:

| Offset / Field | Data Type | Value / Constraint | Description |
| :--- | :--- | :--- | :--- |
| **Magic Header** | 4 bytes ASCII | `b"CRCB"` (`0x43, 0x52, 0x43, 0x42`) | Magic format identifier |
| **Schema Version** | int32 (LE) | `2` | Version 2 specification |
| **ID Length** | int32 (LE) | $1 \le \text{len} \le 256$ | Byte length of UTF-8 specimen ID |
| **Specimen ID** | UTF-8 bytes | e.g. `"TCGA-D5-6540"` | Exact clinical case identifier |
| **Desc Length** | int32 (LE) | $0 \le \text{len} \le 8192$ | Byte length of description |
| **Description** | UTF-8 bytes | Text description | Metadata string |
| **Patch Count ($N$)** | int32 (LE) | $N \ge 1$ | Number of vision feature patches |
| **Vision Dim** | int32 (LE) | `1024` | Phikon-v2 feature dimension |
| **Text Dim** | int32 (LE) | `768` | ClinicalBERT embedding dimension |
| **Data Type** | int32 (LE) | `1` | Float32 Little-Endian |
| **Vision SHA-256** | 32 bytes raw | Digest of vision payload | Cryptographic integrity checksum |
| **Text SHA-256** | 32 bytes raw | Digest of text payload | Cryptographic integrity checksum |
| **Text Payload** | $768 \times 4$ bytes | Float32 Little-Endian | 1D clinical embedding tensor |
| **Vision Payload** | $N \times 1024 \times 4$ bytes | Float32 Little-Endian | 2D histology feature sequence |

### 4.2. Tooling Inventory
* **Packaging Tool**: [`tools/export_specimen_bundle.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tools/export_specimen_bundle.py)
  - Exports PyTorch `.pt` pairs directly into `.crcbundle` v2 binary.
  - Implements zero-loss conversion, SHA-256 calculation, and finiteness validation.
* **Android Reader**: [`android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt)
  - Parses CRCB v2 binary stream via zero-allocation `ByteBuffer.order(ByteOrder.LITTLE_ENDIAN)`.
  - Verifies magic `CRCB`, schema version 2, dimensions 1024/768, and SHA-256 digests.

---

## 5. Cohort Matching Architecture Review

As audited in Phase 0–8:
1. **Reference Labels**: Loaded into table `reference_labels` in `crc_workstation.db`.
2. **Predictions**: Queried via `getUnifiedPredictionRecords(batchRepo)` from `inference_runs` (`crc_workstation.db`) and `batch_items` (`crc_batch_runs.db`).
3. **Matching Engine**: [`CohortEvaluationEngine.kt:L27-L65`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/evaluation/CohortEvaluationEngine.kt#L27-L65):
   - Key: `ref.specimenId.trim().lowercase()`
   - Condition: `pred != null && pred.probability != null && pred.predictedClass != null`
   - Outcome: If matched, assigned to TP, TN, FP, FN based on ground truth and model prediction.

---

## 6. Device Discovery & Transfer Mechanism

* **Physical Device**: Vivo Y21 (`V2125`, Serial `1377355236000F7`, Android 12, API 31).
* **Connection Status**: Verified active via `adb devices -l` (`device transport_id:1`).
* **Storage Location on Device**:
  - App private external cache: `/sdcard/Android/data/org.crc.multimodal/cache/bundles/`
  - Public Documents transfer: `/sdcard/Download/crcbundles/` or `/sdcard/Documents/crcbundles/`
* **Non-Destructive Transfer Plan**: Push individual `.crcbundle` files via `adb push` to device storage without modifying existing SQLite databases, APK installations, or application settings.
* **Strict Gate Protocol**: In accordance with Section 3, Gate 2, **no bundles will be generated, transferred, or imported until the user explicitly reviews and approves the reference manifest.**
