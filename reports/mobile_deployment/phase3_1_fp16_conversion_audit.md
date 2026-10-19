# Phase 3.1 — FP16 Model Conversion & Graph Architecture Audit

> **Document ID:** `AUDIT-PHASE3.1-FP16-CONVERSION`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Source Model (FP32):** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **Converted Model (FP16):** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Audit Status:** **CONVERTED, VERIFIED & OPERATIONAL (49.94% SIZE REDUCTION)**  

---

## 1. Executive Summary

Under Phase 3 authorization, the validated FP32 ONNX model ([`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)) was successfully converted to a standalone uniform FP16 ONNX candidate ([`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)).

Key achievements:
1. **Zero New Package Installations:** Utilized the built-in `onnxruntime.transformers.float16.convert_float_to_float16` engine available in the authorized `onnxruntime==1.31.0` environment.
2. **Exact Storage Reduction:** Model size decreased from **21,015,452 bytes (20.04 MiB)** to **10,519,654 bytes (10.03 MiB)**, representing a **49.94% reduction** in on-disk storage footprint (10,495,798 bytes saved).
3. **Graph Integrity & Validation:** Passed `onnx.checker.check_model` with 0 errors and 0 warnings.
4. **Dynamic Sequence Dimensions:** Fully retained dynamic sequence dimension scaling on the patch-sequence axis (`num_patches`), empirically verified on ONNX Runtime across $N \in \{1, 6, 50, 100, 385, 1000, 4305\}$.
5. **Preserved Output Semantics:** Retained all three model output streams: pre-sigmoid `logit`, post-sigmoid `probability`, and `attention_weights`.

---

## 2. Pre-Conversion Integrity & Asset Hashing

Before initiating conversion, all baseline assets were inspected and confirmed bit-intact:

* **PyTorch Checkpoint (`models/fusion_model_v1_1.pth`):**
  * SHA-256: `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` (Verified intact)
* **FP32 ONNX Reference (`models/fusion_model_fp32.onnx`):**
  * SHA-256: `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` (Preserved without modification)
* **Patient Registry (`data/patient_registry.json`):**
  * SHA-256: `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` (Verified intact)
* **Holdout Cohort Keys (`results/actual_validation_keys.json`):**
  * SHA-256: `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` (Verified intact)
* **Patch Selection Manifest (`protocol1_patch_selection_manifest.json`):**
  * SHA-256: `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` (Verified intact)

---

## 3. Conversion Methodology & Configuration

### 3.1 Conversion Engine
* **Module:** `onnxruntime.transformers.float16.convert_float_to_float16`
* **Source:** Native `onnxruntime==1.31.0` library.
* **Conversion Parameters:**
  * `keep_io_types`: `False` (Converts inputs and outputs to IEEE-754 `float16` for uniform half-precision execution).
  * `min_positive_val`: `1.0e-7` (Prevents denormal underflow).
  * `max_finite_val`: `65504.0` (Standard IEEE-754 half-precision max bound).
  * `disable_shape_infer`: `False` (Validates dynamic dimension shapes during translation).

### 3.2 Resulting Artifact Properties
* **Artifact Path:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)
* **SHA-256 Digest:** `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916`
* **File Size:** 10,519,654 bytes (~10.03 MiB)
* **Opset Version:** 17
* **IR Version:** 8

---

## 4. Graph Architecture & Operator Inventory

The converted FP16 graph consists of 109 nodes and 16 initializers (5,250,561 float16 parameters):

| Operator Type | Count | ONNX Opset 17 FP16 Support | Description in Fusion Pipeline |
| :--- | :---: | :---: | :--- |
| `Constant` | 30 | Full FP16 | Static weights, bias terms, and scalar reshape constants |
| `Unsqueeze` | 18 | Full FP16 | Dimension expansion for multi-head broadcasting |
| `Transpose` | 9 | Full FP16 | Head splitting permutations (`[B, N, H, D] -> [B, H, N, D]`) |
| `Shape` | 7 | Integer | Extracts dynamic sequence length $N$ and batch size |
| `Gather` | 7 | Integer | Slices dimension sizes for dynamic concat vectors |
| `MatMul` | 6 | Full FP16 | Scaled dot-product attention ($Q K^T$ and $A V$) |
| `Concat` | 6 | Integer / Dynamic | Reconstructs dynamic shape vectors |
| `Reshape` | 6 | Dynamic | Sequence length flattening and multi-head merging |
| `Add` | 5 | Full FP16 | Linear layer biases and residual fusion ($Q + \text{Attn}$) |
| `Mul` | 3 | Full FP16 | Scaling factor $1/\sqrt{d_k}$ |
| `Gemm` | 3 | Full FP16 | Text projection and classifier MLP dense layers |
| `Cast` | 2 | Integer | Shape index casting |
| `Div` | 1 | Full FP16 | Layer normalization variance normalization |
| `Softmax` | 1 | Full FP16 | Attention probability distribution across patches |
| `ReduceMean` | 1 | Full FP16 | Layer normalization mean calculation |
| `LayerNormalization` | 1 | Full FP16 | Post-residual LayerNorm |
| `Squeeze` | 1 | Full FP16 | Sequence pooling before classification MLP |
| `Relu` | 1 | Full FP16 | Activation in classifier MLP |
| `Sigmoid` | 1 | Full FP16 | S-curve squashing for MSI probability output |
| **Total Nodes** | **109** | **Full FP16** | **100% Executable on ONNX Runtime CPUExecutionProvider** |

### Tensor I/O Signatures:

| Name | Direction | Element Type | Shape Definition | Dynamic Dimensions |
| :--- | :---: | :---: | :--- | :--- |
| `vision_seq` | Input | `FLOAT16` | `['batch_size', 'num_patches', 1024]` | Dim 0 (`batch_size`), Dim 1 (`num_patches`) |
| `text_summary` | Input | `FLOAT16` | `['batch_size', 768]` | Dim 0 (`batch_size`) |
| `logit` | Output | `FLOAT16` | `['batch_size', 1]` | Dim 0 (`batch_size`) |
| `probability` | Output | `FLOAT16` | `['batch_size', 1]` | Dim 0 (`batch_size`) |
| `attention_weights` | Output | `FLOAT16` | `['batch_size', 1, 'num_patches']` | Dim 0 (`batch_size`), Dim 2 (`num_patches`) |

---

## 5. Dynamic Sequence Dimension Validation

The model was tested across variable patch counts on ONNX Runtime 1.31.0:

| Sequence Length ($N$) | Context / Pathological Meaning | FP32 Prob | FP16 Prob | Absolute Prob Drift | Absolute Logit Drift | Runtime Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | Single patch (Extreme min) | 0.106197 | 0.106140 | $5.68 \times 10^{-5}$ | $6.70 \times 10^{-4}$ | **PASS** |
| **$N = 6$** | Very small sequence | 0.083323 | 0.083374 | $5.06 \times 10^{-5}$ | $4.12 \times 10^{-4}$ | **PASS** |
| **$N = 50$** | Low tissue density | 0.464399 | 0.464355 | $4.39 \times 10^{-5}$ | $1.78 \times 10^{-4}$ | **PASS** |
| **$N = 100$** | Standard small biopsy | 0.457745 | 0.457764 | $1.90 \times 10^{-5}$ | $8.33 \times 10^{-6}$ | **PASS** |
| **$N = 385$** | Cohort median benchmark | 0.252927 | 0.252930 | $2.92 \times 10^{-6}$ | $5.56 \times 10^{-5}$ | **PASS** |
| **$N = 1,000$** | Protocol 1 cap limit | 0.552131 | 0.552246 | $1.15 \times 10^{-4}$ | $6.69 \times 10^{-5}$ | **PASS** |
| **$N = 4,305$** | Extreme upper sequence bound | 0.385068 | 0.385010 | $5.82 \times 10^{-5}$ | $7.45 \times 10^{-5}$ | **PASS** |

Zero shape errors or runtime exceptions occurred across any sequence length.

---

## 6. Audit Sign-off

The FP16 conversion satisfies all structural, topological, operator, and dynamic sequence requirements. Storage reduction of **49.94%** was achieved while strictly preserving the model's tripartite output contract.
