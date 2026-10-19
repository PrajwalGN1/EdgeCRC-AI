# Phase 1.5.2 — Golden Reference & Parity Tolerance Contract Audit

> **Document ID:** `AUDIT-PHASE1.5.2-GOLDEN-REFERENCE`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Audit Status:** **VERIFIED & MATHEMATICALLY CONGRUENT**  
> **Golden Reference Dataset:** [`reports/mobile_deployment/phase1_fp32_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase1_fp32_benchmark_data.json)  

---

## 1. Executive Summary

This audit performs an independent verification of the frozen FP32 golden reference dataset against a fresh, strictly executed PyTorch forward pass across all 66 holdout validation cases.

Both Protocol 1 (Capped at 1,000 patches via manifest) and Protocol 2 (Uncapped full sequences) were audited. The results confirm:
1. **Zero Classification Flips (66/66 Agreement):** 100.0% diagnostic classification agreement between fresh PyTorch execution and the frozen reference across both protocols.
2. **Sub-Microsecond Parity:** Maximum absolute logit difference is $4.91 \times 10^{-7}$ and maximum absolute probability difference is $4.97 \times 10^{-7}$ (exclusively attributable to standard 6-decimal JSON serialization rounding, $|e| < 0.5 \times 10^{-6}$).
3. **Formal Tolerance Contract Defined:** Numerical acceptance criteria for Phase 2 ONNX conversion are formally specified and justified.

---

## 2. Integrity Verification of Upstream Assets

All inputs, parameters, and metadata were verified against the immutable asset baseline:

* **Model Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
  - SHA-256: `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` (21,007,347 bytes)
  - Loaded strictly with `weights_only=True` and `strict=True` into [`SummaryToPatchCrossAttention`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/cross_attention_fusion.py).
* **Patient Registry:** [`data/patient_registry.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/patient_registry.json)  
  - SHA-256: `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` (326 registered patients)
* **Holdout Validation Keys:** [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json)  
  - SHA-256: `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` (66 holdout keys)
* **Precomputed Tensor Store:** [`data/clean_tensors/`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/clean_tensors)  
  - 652 files (326 vision, 326 text), Aggregate SHA-256: `3a57ef2f7424e010d27f327a701c8d3042a82c3cc60bbb029313cef1428312a1`.
* **Class Mapping:** Class 0 = MSS, Class 1 = MSI (Operating Decision Threshold $\tau = 0.83$).

---

## 3. Fresh PyTorch vs. Golden Reference Consistency Audit

A fresh evaluation was executed loading raw tensors and weights from disk, calculating pre-sigmoid logits, post-sigmoid probabilities, and cross-attention maps.

### 3.1 Aggregate Consistency Metrics

| Metric | Protocol 1 (Capped, Manifest-Locked) | Protocol 2 (Uncapped, Dynamic Sequence) | Audit Evaluation |
| :--- | :---: | :---: | :--- |
| **Cases Evaluated** | 66 / 66 | 66 / 66 | 100% cohort coverage |
| **Classification Agreement** | **100.0% (66 / 66)** | **100.0% (66 / 66)** | Zero diagnostic flips |
| **Max Absolute Logit Error ($L_\infty$)** | $\mathbf{4.91 \times 10^{-7}}$ | $\mathbf{4.91 \times 10^{-7}}$ | Serialization rounding ($< 0.5 \times 10^{-6}$) |
| **Max Absolute Prob Error ($L_\infty$)** | $\mathbf{4.97 \times 10^{-7}}$ | $\mathbf{4.97 \times 10^{-7}}$ | Serialization rounding ($< 0.5 \times 10^{-6}$) |
| **Mean Absolute Prob Error (MAE)** | $\mathbf{2.20 \times 10^{-7}}$ | $\mathbf{2.24 \times 10^{-7}}$ | Micro-scale congruence |
| **Attention Map Entropy Agreement** | $\Delta < 1.0 \times 10^{-6}$ | $\Delta < 1.0 \times 10^{-6}$ | Verified |

### 3.2 Metric Definitions & Calculations

Every comparison metric is calculated according to the following mathematical specifications:

$$\text{Logit Error}_i = |\text{logit}_{\text{fresh}, i} - \text{logit}_{\text{ref}, i}|$$

$$\text{Prob Error}_i = |P_{\text{fresh}, i} - P_{\text{ref}, i}|$$

$$\text{MAE} = \frac{1}{66} \sum_{i=1}^{66} |P_{\text{fresh}, i} - P_{\text{ref}, i}|$$

$$\hat{y}_i = \begin{cases} 1 \ (\text{MSI}) & \text{if } P_i \ge 0.83 \\ 0 \ (\text{MSS}) & \text{if } P_i < 0.83 \end{cases}$$

$$\text{Distance to Threshold}_i = P_i - 0.83$$

$$\text{Diagnostic Flip}_i = \mathbb{I}(\hat{y}_{\text{fresh}, i} \ne \hat{y}_{\text{ref}, i})$$

---

## 4. Per-Case Audit Distribution (De-Identified)

Below is the verified distribution for sample representative rows across both protocols (de-identified IDs `CASE_01` through `CASE_66`):

| Row ID | Ground Truth | Patches ($N$) | Protocol 1 Logit | Protocol 1 Prob | Protocol 2 Logit | Protocol 2 Prob | Dist to $\tau=0.83$ (P1) | Dist to $\tau=0.83$ (P2) | Parity Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CASE_01` | MSS (0) | 370 | -2.671342 | 0.064716 | -2.671342 | 0.064716 | -0.765284 | -0.765284 | PASS |
| `CASE_10` | MSS (0) | 267 | -2.593678 | 0.069472 | -2.593678 | 0.069472 | -0.760528 | -0.760528 | PASS |
| `CASE_19` | MSI (1) | 1,690 | +2.051184 | 0.886053 | +2.247812 | 0.904470 | +0.056053 | +0.074470 | PASS |
| `CASE_20` | MSS (0) | 3,914 | +0.984952 | 0.728124 | +1.663588 | 0.840750 | -0.101876 | +0.010750 | PASS* |
| `CASE_35` | MSI (1) | 361 | +1.820067 | 0.860587 | +1.820067 | 0.860587 | +0.030587 | +0.030587 | PASS |
| `CASE_36` | MSI (1) | 80 | +1.586948 | 0.830106 | +1.586948 | 0.830106 | +0.000106 | +0.000106 | PASS |
| `CASE_59` | MSS (0) | 166 | +1.503461 | 0.817998 | +1.503461 | 0.817998 | -0.012002 | -0.012002 | PASS |
| `CASE_66` | MSI (1) | 2,090 | +1.984120 | 0.879105 | +2.189422 | 0.899298 | +0.049105 | +0.069298 | PASS |

*\*Note on `CASE_20`: As documented in Phase 1.5.1, `CASE_20` has $N=3,914$ patches. Under Protocol 1 (capped to 1,000 via manifest), $P=0.728124$ (MSS call). Under Protocol 2 (all 3,914 patches), $P=0.840750$ (MSI call). Both are 100% faithful to their respective input tensors.*

---

## 5. Formal FP32 ONNX Acceptance Criteria Specification

For Phase 2 ONNX model conversion, we establish explicit, mathematically grounded acceptance criteria:

| Acceptance Criterion | Proposed FP32 Limit | Mathematical Definition | Technical Justification |
| :--- | :---: | :--- | :--- |
| **Max Absolute Logit Error** | $\mathbf{\le 1.0 \times 10^{-4}}$ | $\max_i \|\text{logit}_{\text{ONNX}, i} - \text{logit}_{\text{PyTorch}, i}\|$ | Accounts for floating-point reassociation in BLAS matrix multiplications between PyTorch ATen and ONNX Runtime CPU. |
| **Max Absolute Prob Error** | $\mathbf{\le 1.0 \times 10^{-4}}$ | $\max_i \|P_{\text{ONNX}, i} - P_{\text{PyTorch}, i}\|$ | $\sigma'(z) \le 0.25$, so probability error is bounded by $0.25 \times \Delta\text{logit}$. A limit of $10^{-4}$ provides ample safety. |
| **Cohort Mean Absolute Error** | $\mathbf{\le 1.0 \times 10^{-5}}$ | $\frac{1}{66} \sum_i \|P_{\text{ONNX}, i} - P_{\text{PyTorch}, i}\|$ | Verifies that overall model calibration does not shift systematically. |
| **Non-Boundary Classification Agreement** | **100.0% (62 / 62)** | $\hat{y}_{\text{ONNX}, i} == \hat{y}_{\text{PyTorch}, i}$ | Required for all cases outside the boundary sensitivity zone $[0.78, 0.88]$. |
| **Boundary Classification Monitoring** | **Zero Unjustified Flips** | Audit of the 4 boundary cases | Any flip in boundary cases must be accompanied by exact numerical error accounting. |

> [!WARNING]
> **FP32 vs. Future FP16 Separation:**  
> These acceptance limits apply **strictly to FP32 ONNX conversion**. FP16 quantization introduces mantissa truncation ($23 \to 10$ bits) with machine epsilon $\epsilon \approx 9.77 \times 10^{-4}$, which inevitably increases numerical divergence. A separate, independently authorized acceptance contract will govern Phase 3 FP16 evaluation.
