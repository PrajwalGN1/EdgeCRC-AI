# Phase 2.3 — ONNX Runtime FP32 Numerical Parity Report

> **Document ID:** `REPORT-PHASE2.3-NUMERICAL-PARITY`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
> **Evaluated ONNX Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **Benchmark Artifact:** [`reports/mobile_deployment/phase2_parity_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_parity_benchmark_data.json)  
> **Evaluation Status:** **PASSED — ZERO FLIPS ACROSS ALL 66 CASES UNDER BOTH PROTOCOLS**  

---

## 1. Executive Summary

This report establishes the host-side numerical parity verification between the reference PyTorch FP32 multimodal fusion model ([`SummaryToPatchCrossAttention`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/cross_attention_fusion.py)) and the exported ONNX FP32 model ([`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)) executed via ONNX Runtime 1.31.0 (`CPUExecutionProvider`).

In accordance with Phase 2 requirements, all 66 holdout cohort cases were evaluated independently under both frozen evaluation protocols:
1. **Protocol 1 (Manifest-Controlled Capped Inputs):** Sequences capped at 1,000 patches via deterministic indices in [`protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json).
2. **Protocol 2 (Uncapped Candidate Inputs):** Full variable patch sequences processed without subsampling (lengths up to 3,914 patches).

### Key Parity Verification Results:
* **Classification Agreement:** **100.00% (66 / 66 cases agreed)** under both Protocol 1 and Protocol 2 at decision threshold $\tau = 0.83$.
* **Classification Flips:** **Exactly 0 flips** across the entire cohort.
* **Numerical Tolerances:** Actual observed errors are **20 to 200 times smaller** than the approved tolerance limits across all cases.
* **Boundary Case Stability:** Critical boundary case `CASE_36` (reference probability 0.830106, just $+0.000106$ above threshold) evaluated on ONNX Runtime to 0.830106 with error $4.97 \times 10^{-7}$, strictly preserving the positive classification call.

---

## 2. Approved Numerical Tolerances & Aggregate Performance

The approved parity contract specifies three numerical tolerance limits and a zero-flip requirement on non-boundary cases:

| Metric / Requirement | Approved Gate Threshold | Protocol 1 Observed | Protocol 2 Observed | Gate Compliance |
| :--- | :---: | :---: | :---: | :---: |
| **Max Absolute Logit Error ($L_\infty$)** | $\le 1.00 \times 10^{-4}$ | **$2.05 \times 10^{-6}$** | **$2.88 \times 10^{-6}$** | **PASS (48x margin)** |
| **Max Absolute Probability Error ($L_\infty$)** | $\le 1.00 \times 10^{-4}$ | **$4.97 \times 10^{-7}$** | **$4.97 \times 10^{-7}$** | **PASS (201x margin)** |
| **Aggregate Mean Probability Error (MAE)** | $\le 1.00 \times 10^{-5}$ | **$2.31 \times 10^{-7}$** | **$2.32 \times 10^{-7}$** | **PASS (43x margin)** |
| **Classification Agreement ($\tau = 0.83$)** | 100% (0 flips) | **100.00% (66 / 66)** | **100.00% (66 / 66)** | **PASS (0 flips)** |
| **Boundary Case Agreement** | Zero flip on $\tau \pm 0.05$ | **100.00% (3 / 3)** | **100.00% (4 / 4)** | **PASS (0 flips)** |
| **Dynamic Shape Support ($N \in [1, 4305]$)** | All test counts pass | **7 / 7 counts pass** | **7 / 7 counts pass** | **PASS** |

---

## 3. Dynamic Sequence Dimension Parity Verification

Synthetic test tensors covering the full operational range of patch sequence lengths were evaluated against the PyTorch reference model:

| Tested Patch Count ($N$) | Pathological / Clinical Context | PyTorch Ref Prob | ONNX Prob | Absolute Prob Error | Absolute Logit Error | Tolerance Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | Single isolated patch (Extreme min) | 0.923603 | 0.923603 | $5.96 \times 10^{-8}$ | $4.77 \times 10^{-7}$ | **PASS** |
| **$N = 6$** | Very small sequence | 0.869295 | 0.869295 | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |
| **$N = 50$** | Low tissue density | 0.392143 | 0.392143 | $2.98 \times 10^{-8}$ | $1.49 \times 10^{-7}$ | **PASS** |
| **$N = 100$** | Standard small biopsy | 0.542573 | 0.542573 | $0.00 \times 10^0$ | $8.94 \times 10^{-8}$ | **PASS** |
| **$N = 385$** | Cohort median benchmark | 0.471722 | 0.471722 | $0.00 \times 10^0$ | $6.71 \times 10^{-8}$ | **PASS** |
| **$N = 1,000$** | Protocol 1 cap limit | 0.653745 | 0.653745 | $0.00 \times 10^0$ | $5.96 \times 10^{-8}$ | **PASS** |
| **$N = 4,305$** | Extreme upper sequence bound | 0.605037 | 0.605037 | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |

---

## 4. Boundary Case Stability Analysis

Four cases in the holdout cohort lie within the decision margin ($0.78 \le P \le 0.88$ or $|\text{Prob} - \tau| < 0.05$):

### Protocol 1 Boundary Cases:
1. **`CASE_36` (Ground Truth: MSI / Class 1):**
   * *Reference Probability:* `0.830106` | *ONNX Probability:* `0.830106`
   * *Probability Error:* $4.97 \times 10^{-7}$ | *Logit Error:* $3.89 \times 10^{-7}$
   * *Distance to Threshold ($\tau = 0.83$):* **$+0.000106$ (+0.0106 percentage points)**
   * *Reference Decision:* 1 (MSI) | *ONNX Decision:* 1 (MSI) $\implies$ **NO FLIP (MATCH)**
2. **`CASE_59` (Ground Truth: MSS / Class 0):**
   * *Reference Probability:* `0.817998` | *ONNX Probability:* `0.817998`
   * *Probability Error:* $4.85 \times 10^{-7}$ | *Logit Error:* $1.50 \times 10^{-7}$
   * *Distance to Threshold:* **$-0.012002$ (-1.2002 percentage points)**
   * *Reference Decision:* 0 (MSS) | *ONNX Decision:* 0 (MSS) $\implies$ **NO FLIP (MATCH)**
3. **`CASE_35` (Ground Truth: MSI / Class 1):**
   * *Reference Probability:* `0.860587` | *ONNX Probability:* `0.860587`
   * *Probability Error:* $2.39 \times 10^{-7}$ | *Logit Error:* $2.44 \times 10^{-7}$
   * *Distance to Threshold:* **$+0.030587$ (+3.0587 percentage points)**
   * *Reference Decision:* 1 (MSI) | *ONNX Decision:* 1 (MSI) $\implies$ **NO FLIP (MATCH)**

### Protocol 2 Additional Boundary Case:
4. **`CASE_20` (Ground Truth: MSS / Class 0, Uncapped $N=3,914$ Patches):**
   * *Reference Probability:* `0.840750` | *ONNX Probability:* `0.840750`
   * *Probability Error:* $2.10 \times 10^{-8}$ | *Logit Error:* $4.46 \times 10^{-8}$
   * *Distance to Threshold:* **$+0.010750$ (+1.0750 percentage points)**
   * *Reference Decision:* 1 (False Positive) | *ONNX Decision:* 1 (False Positive) $\implies$ **NO FLIP (MATCH)**

---

## 5. Audit of Capped Cases under Protocol 1

Exactly 8 cases in the holdout cohort have $N > 1,000$ patches and were subsampled using the immutable indices in [`protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json):

| De-identified ID | Original Patches | Evaluated Patches | Ground Truth | Ref Probability | ONNX Probability | Probability Error | Decision Match? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CASE_19` | 1,690 | 1,000 | MSI (1) | 0.985367 | 0.985367 | $5.97 \times 10^{-8}$ | **MATCH (1)** |
| `CASE_20` | 3,914 | 1,000 | MSS (0) | 0.728124 | 0.728124 | $3.80 \times 10^{-7}$ | **MATCH (0)** |
| `CASE_26` | 1,403 | 1,000 | MSS (0) | 0.989148 | 0.989148 | $9.85 \times 10^{-8}$ | **MATCH (1)** |
| `CASE_43` | 1,433 | 1,000 | MSS (0) | 0.332312 | 0.332312 | $4.89 \times 10^{-7}$ | **MATCH (0)** |
| `CASE_49` | 1,921 | 1,000 | MSI (1) | 0.130220 | 0.130220 | $9.34 \times 10^{-8}$ | **MATCH (0)** |
| `CASE_51` | 2,504 | 1,000 | MSS (0) | 0.130048 | 0.130048 | $6.77 \times 10^{-9}$ | **MATCH (0)** |
| `CASE_65` | 1,413 | 1,000 | MSS (0) | 0.118750 | 0.118750 | $2.38 \times 10^{-8}$ | **MATCH (0)** |
| `CASE_66` | 2,090 | 1,000 | MSI (1) | 0.992791 | 0.992791 | $6.26 \times 10^{-8}$ | **MATCH (1)** |

All 8 capped cases exhibited errors under $4.9 \times 10^{-7}$ with zero classification discrepancy between PyTorch and ONNX Runtime.

---

## 6. Complete 66-Case Protocol 1 Numerical Parity Table

| De-identified ID | Patches | Ground Truth | Ref Prob | ONNX Prob | Prob Error | Logit Error | Dist to $\tau$ | Ref Call | ONNX Call | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CASE_01` | 345 | MSI (1) | 0.990385 | 0.990385 | 2.94e-07 | 2.02e-07 | +0.1604 | 1 | 1 | PASS |
| `CASE_02` | 541 | MSS (0) | 0.162250 | 0.162250 | 4.65e-07 | 1.27e-07 | -0.6678 | 0 | 0 | PASS |
| `CASE_03` | 452 | MSI (1) | 0.937404 | 0.937404 | 3.35e-07 | 7.53e-07 | +0.1074 | 1 | 1 | PASS |
| `CASE_04` | 61 | MSS (0) | 0.013114 | 0.013114 | 1.25e-07 | 1.32e-06 | -0.8169 | 0 | 0 | PASS |
| `CASE_05` | 331 | MSS (0) | 0.011473 | 0.011473 | 3.87e-07 | 9.02e-07 | -0.8185 | 0 | 0 | PASS |
| `CASE_06` | 308 | MSS (0) | 0.003447 | 0.003447 | 3.32e-07 | 2.05e-06 | -0.8266 | 0 | 0 | PASS |
| `CASE_07` | 516 | MSS (0) | 0.091230 | 0.091230 | 6.46e-08 | 7.53e-07 | -0.7388 | 0 | 0 | PASS |
| `CASE_08` | 299 | MSS (0) | 0.017761 | 0.017761 | 2.76e-07 | 9.14e-07 | -0.8122 | 0 | 0 | PASS |
| `CASE_09` | 94 | MSS (0) | 0.065060 | 0.065060 | 2.19e-07 | 9.15e-07 | -0.7649 | 0 | 0 | PASS |
| `CASE_10` | 52 | MSS (0) | 0.014129 | 0.014129 | 4.00e-07 | 3.91e-07 | -0.8159 | 0 | 0 | PASS |
| `CASE_11` | 279 | MSS (0) | 0.021346 | 0.021346 | 1.76e-07 | 6.57e-07 | -0.8087 | 0 | 0 | PASS |
| `CASE_12` | 190 | MSS (0) | 0.006736 | 0.006736 | 1.59e-07 | 6.15e-07 | -0.8233 | 0 | 0 | PASS |
| `CASE_13` | 271 | MSI (1) | 0.994391 | 0.994391 | 3.93e-07 | 6.33e-07 | +0.1644 | 1 | 1 | PASS |
| `CASE_14` | 370 | MSS (0) | 0.015399 | 0.015399 | 4.56e-07 | 3.75e-07 | -0.8146 | 0 | 0 | PASS |
| `CASE_15` | 40 | MSS (0) | 0.006374 | 0.006374 | 2.97e-07 | 8.30e-07 | -0.8236 | 0 | 0 | PASS |
| `CASE_16` | 60 | MSS (0) | 0.002760 | 0.002760 | 2.61e-07 | 4.10e-07 | -0.8272 | 0 | 0 | PASS |
| `CASE_17` | 162 | MSS (0) | 0.041518 | 0.041518 | 5.69e-08 | 2.45e-07 | -0.7885 | 0 | 0 | PASS |
| `CASE_18` | 52 | MSS (0) | 0.558614 | 0.558614 | 2.54e-07 | 3.24e-08 | -0.2714 | 0 | 0 | PASS |
| `CASE_19` | 1000 | MSI (1) | 0.985367 | 0.985367 | 5.97e-08 | 2.22e-07 | +0.1554 | 1 | 1 | PASS |
| `CASE_20` | 1000 | MSS (0) | 0.728124 | 0.728124 | 3.80e-07 | 2.70e-07 | -0.1019 | 0 | 0 | PASS |
| `CASE_21` | 406 | MSS (0) | 0.002370 | 0.002370 | 2.39e-07 | 8.03e-07 | -0.8276 | 0 | 0 | PASS |
| `CASE_22` | 70 | MSS (0) | 0.001834 | 0.001834 | 3.49e-08 | 4.66e-07 | -0.8282 | 0 | 0 | PASS |
| `CASE_23` | 220 | MSS (0) | 0.295826 | 0.295826 | 4.17e-08 | 4.36e-07 | -0.5342 | 0 | 0 | PASS |
| `CASE_24` | 10 | MSS (0) | 0.031855 | 0.031855 | 4.04e-07 | 6.82e-07 | -0.7981 | 0 | 0 | PASS |
| `CASE_25` | 370 | MSI (1) | 0.975944 | 0.975944 | 2.81e-07 | 1.30e-06 | +0.1459 | 1 | 1 | PASS |
| `CASE_26` | 1000 | MSS (0) | 0.989148 | 0.989148 | 9.85e-08 | 5.50e-07 | +0.1591 | 1 | 1 | PASS |
| `CASE_27` | 913 | MSI (1) | 0.146424 | 0.146424 | 3.92e-07 | 2.28e-07 | -0.6836 | 0 | 0 | PASS |
| `CASE_28` | 35 | MSS (0) | 0.051696 | 0.051696 | 2.36e-07 | 5.80e-07 | -0.7783 | 0 | 0 | PASS |
| `CASE_29` | 290 | MSS (0) | 0.035441 | 0.035441 | 4.88e-07 | 5.30e-07 | -0.7946 | 0 | 0 | PASS |
| `CASE_30` | 98 | MSS (0) | 0.069602 | 0.069602 | 1.66e-07 | 3.60e-07 | -0.7604 | 0 | 0 | PASS |
| `CASE_31` | 139 | MSS (0) | 0.065912 | 0.065912 | 3.49e-07 | 6.08e-07 | -0.7641 | 0 | 0 | PASS |
| `CASE_32` | 262 | MSS (0) | 0.070781 | 0.070781 | 7.49e-09 | 7.61e-07 | -0.7592 | 0 | 0 | PASS |
| `CASE_33` | 858 | MSI (1) | 0.992237 | 0.992237 | 3.29e-07 | 7.27e-07 | +0.1622 | 1 | 1 | PASS |
| `CASE_34` | 232 | MSI (1) | 0.989086 | 0.989086 | 3.26e-07 | 5.76e-07 | +0.1591 | 1 | 1 | PASS |
| `CASE_35` | 361 | MSI (1) | 0.860587 | 0.860587 | 2.39e-07 | 2.44e-07 | +0.0306 | 1 | 1 | PASS |
| `CASE_36` | 80 | MSI (1) | 0.830106 | 0.830106 | 4.97e-07 | 3.89e-07 | +0.0001 | 1 | 1 | PASS |
| `CASE_37` | 35 | MSS (0) | 0.070763 | 0.070763 | 4.55e-07 | 1.42e-07 | -0.7592 | 0 | 0 | PASS |
| `CASE_38` | 190 | MSS (0) | 0.178422 | 0.178422 | 9.34e-08 | 3.05e-07 | -0.6516 | 0 | 0 | PASS |
| `CASE_39` | 121 | MSS (0) | 0.002019 | 0.002019 | 2.80e-07 | 3.43e-07 | -0.8280 | 0 | 0 | PASS |
| `CASE_40` | 542 | MSS (0) | 0.377595 | 0.377595 | 6.70e-08 | 4.75e-07 | -0.4524 | 0 | 0 | PASS |
| `CASE_41` | 619 | MSI (1) | 0.319604 | 0.319604 | 9.39e-09 | 2.27e-07 | -0.5104 | 0 | 0 | PASS |
| `CASE_42` | 169 | MSS (0) | 0.002958 | 0.002958 | 2.94e-10 | 7.70e-07 | -0.8270 | 0 | 0 | PASS |
| `CASE_43` | 1000 | MSS (0) | 0.332312 | 0.332312 | 4.89e-07 | 1.14e-06 | -0.4977 | 0 | 0 | PASS |
| `CASE_44` | 283 | MSS (0) | 0.911503 | 0.911503 | 1.02e-07 | 2.90e-07 | +0.0815 | 1 | 1 | PASS |
| `CASE_45` | 155 | MSS (0) | 0.003557 | 0.003557 | 2.12e-07 | 4.62e-07 | -0.8264 | 0 | 0 | PASS |
| `CASE_46` | 613 | MSS (0) | 0.002325 | 0.002325 | 3.56e-07 | 2.77e-07 | -0.8277 | 0 | 0 | PASS |
| `CASE_47` | 703 | MSI (1) | 0.991042 | 0.991042 | 1.01e-07 | 2.62e-07 | +0.1610 | 1 | 1 | PASS |
| `CASE_48` | 220 | MSS (0) | 0.006376 | 0.006376 | 2.37e-07 | 7.44e-07 | -0.8236 | 0 | 0 | PASS |
| `CASE_49` | 1000 | MSI (1) | 0.130220 | 0.130220 | 9.34e-08 | 2.16e-07 | -0.6998 | 0 | 0 | PASS |
| `CASE_50` | 356 | MSS (0) | 0.012500 | 0.012500 | 4.77e-08 | 6.02e-08 | -0.8175 | 0 | 0 | PASS |
| `CASE_51` | 1000 | MSS (0) | 0.130048 | 0.130048 | 6.77e-09 | 7.28e-07 | -0.7000 | 0 | 0 | PASS |
| `CASE_52` | 113 | MSS (0) | 0.762421 | 0.762421 | 1.07e-07 | 6.60e-07 | -0.0676 | 0 | 0 | PASS |
| `CASE_53` | 67 | MSS (0) | 0.090774 | 0.090774 | 2.98e-07 | 1.38e-07 | -0.7392 | 0 | 0 | PASS |
| `CASE_54` | 447 | MSS (0) | 0.586794 | 0.586794 | 1.00e-07 | 2.90e-07 | -0.2432 | 0 | 0 | PASS |
| `CASE_55` | 444 | MSS (0) | 0.002443 | 0.002443 | 4.54e-08 | 1.65e-06 | -0.8276 | 0 | 0 | PASS |
| `CASE_56` | 23 | MSS (0) | 0.938753 | 0.938753 | 2.47e-07 | 3.93e-08 | +0.1088 | 1 | 1 | PASS |
| `CASE_57` | 402 | MSS (0) | 0.014741 | 0.014741 | 2.05e-07 | 1.63e-06 | -0.8153 | 0 | 0 | PASS |
| `CASE_58` | 315 | MSI (1) | 0.187694 | 0.187694 | 2.81e-07 | 8.59e-07 | -0.6423 | 0 | 0 | PASS |
| `CASE_59` | 166 | MSS (0) | 0.817998 | 0.817998 | 4.85e-07 | 1.50e-07 | -0.0120 | 0 | 0 | PASS |
| `CASE_60` | 62 | MSS (0) | 0.001732 | 0.001732 | 9.76e-08 | 5.69e-07 | -0.8283 | 0 | 0 | PASS |
| `CASE_61` | 349 | MSS (0) | 0.027655 | 0.027655 | 3.33e-07 | 7.01e-07 | -0.8023 | 0 | 0 | PASS |
| `CASE_62` | 327 | MSS (0) | 0.005316 | 0.005316 | 3.09e-07 | 5.44e-07 | -0.8247 | 0 | 0 | PASS |
| `CASE_63` | 502 | MSI (1) | 0.986665 | 0.986665 | 3.68e-07 | 4.78e-07 | +0.1567 | 1 | 1 | PASS |
| `CASE_64` | 99 | MSS (0) | 0.011387 | 0.011387 | 2.29e-07 | 2.18e-07 | -0.8186 | 0 | 0 | PASS |
| `CASE_65` | 1000 | MSS (0) | 0.118750 | 0.118750 | 2.38e-08 | 3.37e-07 | -0.7113 | 0 | 0 | PASS |
| `CASE_66` | 1000 | MSI (1) | 0.992791 | 0.992791 | 6.26e-08 | 9.58e-07 | +0.1628 | 1 | 1 | PASS |

---

## 7. Numerical Parity Audit Conclusion

The ONNX FP32 model achieves 100% numerical parity with the PyTorch golden baseline under both Protocol 1 and Protocol 2. There are 0 tolerance violations, 0 classification flips, and zero drift in borderline cases.
