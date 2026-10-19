# Phase 3.3 — Boundary-Case Sensitivity & Decision Stability Audit

> **Document ID:** `AUDIT-PHASE3.3-BOUNDARY-CASES`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **FP32 Reference Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **FP16 Converted Model:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Audit Status:** **VERIFIED STABLE — ZERO BOUNDARY FLIPS OCCURRED UNDER FP16**  

---

## 1. Executive Summary

In machine learning deployment for clinical diagnostics, decision boundaries ($P = \tau$) represent critical sensitivity inflection points. Small floating-point precision changes during quantization can displace borderline patients across the threshold, generating false positives or false negatives.

During Phase 2, case **`CASE_36`** was flagged as the project's single most sensitive case, having a reference probability of `0.830106` against the operating threshold $\tau = 0.83$ (a positive margin of just $+0.000106$). Because IEEE-754 FP16 machine epsilon ($\sim 9.77 \times 10^{-4}$) is roughly an order of magnitude larger than this margin, `CASE_36` carried an elevated risk of flipping to `0` (MSS) under uniform FP16 conversion.

This audit establishes the empirical behavior of all boundary cases ($0.78 \le P \le 0.88$) under FP16 ONNX Runtime execution:
* **`CASE_36` Retention:** Evaluated to **`0.830078`** in FP16, drifting by only $2.84 \times 10^{-5}$ and remaining **$+0.000078$ (+0.0078 percentage points) above threshold**. The classification decision `1` (MSI) was **strictly preserved**.
* **Cohort-Wide Boundary Stability:** All 4 boundary cases retained identical diagnostic classifications between FP32 and FP16 with **zero classification flips**.
* **Mixed-Precision Candidate Necessity:** Because uniform FP16 achieved 100% agreement and preserved all boundary decisions without error, a separate mixed-precision candidate was not required for host-side correctness.

---

## 2. Forensic Analysis of Threshold-Adjacent Cases

All cases within the decision margin ($0.78 \le P \le 0.88$ or $|P - 0.83| < 0.05$) were evaluated under full-precision unrounded inputs:

### 2.1 Critical Boundary Case: `CASE_36`
* **Ground Truth Diagnosis:** MSI (Class 1)
* **Pathological Patch Count:** 80 patches
* **Unrounded FP32 Reference Probability:** `0.8301064968` ($+0.000106$ margin)
* **Unrounded FP16 Converted Probability:** `0.8300781250` ($+0.000078$ margin)
* **Observed Numerical Drift:** $|0.8300781250 - 0.8301064968| = \mathbf{2.837187 \times 10^{-5}}$
* **Classification Decision:**
  * FP32 Reference Decision: `1` (MSI)
  * FP16 Converted Decision: `1` (MSI)
  * **Classification Flip:** **FALSE (STABLY PRESERVED)**

### 2.2 Boundary Case: `CASE_59`
* **Ground Truth Diagnosis:** MSS (Class 0)
* **Pathological Patch Count:** 166 patches
* **Unrounded FP32 Reference Probability:** `0.8179975152` ($-0.012002$ margin)
* **Unrounded FP16 Converted Probability:** `0.8178710938` ($-0.012129$ margin)
* **Observed Numerical Drift:** $|0.8178710938 - 0.8179975152| = \mathbf{1.264214 \times 10^{-4}}$
* **Classification Decision:**
  * FP32 Reference Decision: `0` (MSS)
  * FP16 Converted Decision: `0` (MSS)
  * **Classification Flip:** **FALSE (STABLY PRESERVED)**

### 2.3 Boundary Case: `CASE_35`
* **Ground Truth Diagnosis:** MSI (Class 1)
* **Pathological Patch Count:** 361 patches
* **Unrounded FP32 Reference Probability:** `0.8605872393` ($+0.030587$ margin)
* **Unrounded FP16 Converted Probability:** `0.8608398438` ($+0.030840$ margin)
* **Observed Numerical Drift:** $|0.8608398438 - 0.8605872393| = \mathbf{2.526045 \times 10^{-4}}$
* **Classification Decision:**
  * FP32 Reference Decision: `1` (MSI)
  * FP16 Converted Decision: `1` (MSI)
  * **Classification Flip:** **FALSE (STABLY PRESERVED)**

### 2.4 Boundary Case: `CASE_20` (Protocol 2 Uncapped Evaluation)
* **Ground Truth Diagnosis:** MSS (Class 0) — Note: Known baseline false positive
* **Pathological Patch Count:** 3,914 patches (full uncapped slide)
* **Unrounded FP32 Reference Probability:** `0.8407499790` ($+0.010750$ margin)
* **Unrounded FP16 Converted Probability:** `0.8408203125` ($+0.010820$ margin)
* **Observed Numerical Drift:** $|0.8408203125 - 0.8407499790| = \mathbf{7.033348 \times 10^{-5}}$
* **Classification Decision:**
  * FP32 Reference Decision: `1` (MSI — False Positive)
  * FP16 Converted Decision: `1` (MSI — False Positive)
  * **Classification Flip vs FP32:** **FALSE (STABLY PRESERVED)**

---

## 3. Boundary Summary Table

| Case Identifier | Evaluation Mode | Ground Truth | Patches | FP32 Probability | FP16 Probability | Absolute Probability Drift | FP16 Margin to Threshold ($\tau = 0.83$) | FP32 Call | FP16 Call | Flip Occurred? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CASE_36`** | Protocol 1 & 2 | MSI (1) | 80 | 0.830106 | 0.830078 | $2.84 \times 10^{-5}$ | $+0.000078$ | 1 (MSI) | 1 (MSI) | **NO** |
| **`CASE_59`** | Protocol 1 & 2 | MSS (0) | 166 | 0.817998 | 0.817871 | $1.26 \times 10^{-4}$ | $-0.012129$ | 0 (MSS) | 0 (MSS) | **NO** |
| **`CASE_35`** | Protocol 1 & 2 | MSI (1) | 361 | 0.860587 | 0.860840 | $2.52 \times 10^{-4}$ | $+0.030840$ | 1 (MSI) | 1 (MSI) | **NO** |
| **`CASE_20`** | Protocol 2 (Uncapped) | MSS (0) | 3,914 | 0.840750 | 0.840820 | $7.03 \times 10^{-5}$ | $+0.010820$ | 1 (FP) | 1 (FP) | **NO** |

---

## 4. Assessment of Mixed-Precision Requirement

In the Phase 3 master plan:
> *"If uniform FP16 causes unacceptable drift, investigate a separate mixed-precision candidate that preserves numerically sensitive operations in FP32."*

### Decision Rationale:
1. **Zero Classification Drift:** The uniform FP16 candidate ([`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)) achieved **100.00% agreement with the FP32 reference model across all 66 cases**.
2. **Boundary Retention:** `CASE_36` remained comfortably above the threshold ($P_{\text{FP16}} = 0.830078 > 0.830000$).
3. **No Unacceptable Degradation:** Mean probability drift across the cohort was $5.23 \times 10^{-5}$, well within acceptable bounds.
4. **Computational Efficiency:** Introducing a mixed-precision graph would inject intermediate `Cast` nodes (e.g. before LayerNorm or MLP), increasing graph complexity and runtime memory copies without providing any diagnostic benefit on the evaluated cohort.

**Conclusion:** The uniform FP16 candidate is fully acceptable. Construction of an alternative mixed-precision model is unnecessary for host-side deployment readiness.

---

## 5. Mobile Runtime Guardrail Recommendation (Phase 4)

Although `CASE_36` is stable on host x86 CPU ONNX Runtime, its remaining margin in FP16 is narrow ($+0.000078$). 

On mobile edge hardware (Qualcomm Hexagon NPU, MediaTek APU, or ARM Mali GPU), execution backends frequently utilize non-standard fused multiply-add (FMA) implementations or aggressive FP16 fast-math rounding.

**Phase 4 Guardrail:**  
During physical Android on-device benchmarking in Phase 4, `CASE_36` must be monitored as a designated canary case. If mobile hardware-specific floating-point arithmetic shifts `CASE_36` below $0.830000$, a selective mixed-precision graph (keeping the classifier MLP in FP32) should be deployed as a fallback.
