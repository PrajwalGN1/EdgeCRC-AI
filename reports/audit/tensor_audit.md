# Precomputed Tensor Store Audit

**Audit Date:** 2026-10-12  
**Target Directory:** [`data/clean_tensors/`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/clean_tensors)  
**Status:** **PASSED (100% PAIRING & NUMERICAL INTEGRITY)**  

---

## 1. Aggregate Inventory & File Pairing

Every tensor in `data/clean_tensors/` was verified via read-only safe loading using PyTorch `weights_only=True`:

| Category | Count | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Total Files** | **652** | Verified | Exactly 326 vision + 326 text tensors. |
| **Vision Tensors (`*_vision.pt`)** | **326** | Verified | Maps to 326 unique patient identifiers. |
| **Text Tensors (`*_text.pt`)** | **326** | Verified | Maps to 326 unique patient identifiers. |
| **Paired Patients** | **326** | **100.0%** | Every single patient has both a vision and text tensor. |
| **Orphan Vision Tensors** | **0** | Clean | No uncoupled vision files. |
| **Orphan Text Tensors** | **0** | Clean | No uncoupled text files. |

---

## 2. Dimensional & Precision Profiles

### Vision Tensors (`*_vision.pt`)
* **Observed Shape:** $[N, 1024]$, where $N$ denotes the number of extracted histology patches per patient.
* **Feature Dimension:** Strictly **1024** across all 326 files. This exactly matches the output embedding dimension of **Phikon-v2** (`owkin/phikon-v2`, DINOv2 ViT-L/16).
* **Patch Count ($N$) Distribution:**
  * **Minimum:** 6 patches (e.g., small biopsy fragment).
  * **Maximum:** 4,305 patches.
  * **Mean:** **531.89 patches** per whole slide image.
* **Data Type:** `torch.float32` (FP32 single-precision float) exclusively.
* **Storage Device:** CPU.

### Text Tensors (`*_text.pt`)
* **Observed Shape:** $[1, 768]$ uniformly across all 326 files.
* **Feature Dimension:** Strictly **768**. This exactly matches the hidden embedding dimension of **ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`, BERT-base `[CLS]` token summary).
* **Data Type:** `torch.float32` (FP32 single-precision float) exclusively.
* **Storage Device:** CPU.

---

## 3. Numerical Health & Safety Verification

1. **Safe Deserialization:**
   * 100% of all 652 `.pt` files loaded cleanly with `torch.load(..., weights_only=True)`. No arbitrary Python code execution or unpickling vulnerabilities were triggered.
2. **NaN and Infinity Scan:**
   * **0 NaNs** detected across all scanned features.
   * **0 Infs** detected across all scanned features.
3. **Empty Tensor Detection:**
   * 0 empty or zero-byte files found.

---

## 4. Architectural Compatibility Findings

* Both [SummaryToPatchCrossAttention](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/cross_attention_fusion.py#L4-L63) (`vision_dim=1024`, `text_dim=768`) and [MultimodalFusionNetwork](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/late_fusion_model.py#L5-L93) accept these tensor dimensions directly.
* In [multimodal_dataset.py](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/multimodal_dataset.py#L76-L80), when $N > 1000$, dynamic random sampling cap (`torch.randperm(num_patches)[:1000]`) is applied on-the-fly.
