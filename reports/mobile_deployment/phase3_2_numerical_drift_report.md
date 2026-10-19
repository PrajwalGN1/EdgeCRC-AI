# Phase 3.2 — FP16 Numerical Drift & Parity Report

> **Document ID:** `REPORT-PHASE3.2-NUMERICAL-DRIFT`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **FP32 Reference Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **FP16 Converted Model:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Benchmark Dataset:** [`reports/mobile_deployment/phase3_fp16_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_fp16_benchmark_data.json)  
> **Audit Status:** **PASSED — ZERO CLASSIFICATION FLIPS (100.00% AGREEMENT WITH FP32)**  

---

## 1. Executive Summary

This report evaluates numerical drift between the validated host-side FP32 ONNX model and the newly converted uniform FP16 ONNX candidate ([`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)) across all 66 holdout cohort cases under both evaluation protocols:
* **Protocol 1 (Manifest-Controlled Capped Inputs):** Sequences capped at 1,000 patches via deterministic indices.
* **Protocol 2 (Uncapped Candidate Inputs):** Full variable patch sequences processed without subsampling.

### Primary Audit Findings:
* **Classification Agreement:** **100.00% (66 / 66 cases agreed with FP32)** at decision threshold $\tau = 0.83$ under both Protocol 1 and Protocol 2.
* **Classification Flips:** **Exactly 0 flips** vs. the FP32 reference model.
* **Accuracy vs. Ground Truth:** Identical to the FP32 reference:
  * Protocol 1 Accuracy: **89.39% (59/66 cases correct)**
  * Protocol 2 Accuracy: **87.88% (58/66 cases correct)**
* **Numerical Drift Magnitude:**
  * Maximum probability drift observed across all 66 cases is $2.99 \times 10^{-4}$ (under Protocol 2) and $2.52 \times 10^{-4}$ (under Protocol 1).
  * Mean probability drift is $5.23 \times 10^{-5}$ (Protocol 1) and $5.73 \times 10^{-5}$ (Protocol 2).
* **Threshold Stability:** Critical boundary case `CASE_36` (reference probability 0.830106, margin $+0.000106$) evaluated on FP16 to **0.830078** ($+0.000078$ above threshold), safely avoiding a classification flip.

---

## 2. Proposed FP16 Numerical Tolerance Contract

Because IEEE-754 half-precision (FP16) operates with 10 mantissa bits ($\epsilon_{\text{FP16}} \approx 9.77 \times 10^{-4}$) compared to FP32's 23 mantissa bits ($\epsilon_{\text{FP32}} \approx 1.19 \times 10^{-7}$), the FP32 tolerance limits ($10^{-4}$ max error, $10^{-5}$ MAE) cannot be directly applied. 

We define the formal **FP16 Numerical Acceptance Contract** based on floating-point arithmetic bounds and clinical risk tolerances:

| Metric / Gate Criterion | FP32 Reference Gate | Proposed FP16 Gate Limit | Observed Protocol 1 | Observed Protocol 2 | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Max Absolute Logit Error ($L_\infty$)** | $\le 1.00 \times 10^{-4}$ | **$\le 5.00 \times 10^{-3}$** | **$1.99 \times 10^{-3}$** | **$1.99 \times 10^{-3}$** | **PASS** |
| **Max Absolute Probability Error ($L_\infty$)** | $\le 1.00 \times 10^{-4}$ | **$\le 1.00 \times 10^{-3}$** | **$2.52 \times 10^{-4}$** | **$2.99 \times 10^{-4}$** | **PASS** |
| **Aggregate Mean Probability Error (MAE)** | $\le 1.00 \times 10^{-5}$ | **$\le 1.00 \times 10^{-4}$** | **$5.23 \times 10^{-5}$** | **$5.73 \times 10^{-5}$** | **PASS** |
| **Classification Agreement vs FP32 ($\tau = 0.83$)** | 100% (0 flips) | **100.00% (0 flips)** | **100.00% (0 flips)** | **100.00% (0 flips)** | **PASS** |
| **Boundary Case Retention ($0.78 \le P \le 0.88$)** | 0 flips | **0 flips** | **0 flips (3 / 3 cases)** | **0 flips (4 / 4 cases)** | **PASS** |
| **Dynamic Shape Scaling ($N \in [1, 4305]$)** | 0 shape errors | **0 shape errors** | **7 / 7 counts pass** | **7 / 7 counts pass** | **PASS** |

---

## 3. Confusion Matrix & Diagnostic Metric Comparisons

To confirm that FP16 quantization introduces no diagnostic performance degradation, we compare the complete confusion matrices and derived diagnostic metrics between the FP32 baseline and the converted FP16 candidate:

### Protocol 1 (Capped via Manifest at 1,000 Patches):
* **Class Mapping:** 0 = MSS (50 cases), 1 = MSI (16 cases)
* **Decision Threshold:** $\tau = 0.83$

```
                   FP32 Reference Model                     FP16 Converted Model
                ┌─────────────────────────┐              ┌─────────────────────────┐
                │ Predicted 0 │ Predicted 1│              │ Predicted 0 │ Predicted 1│
 ┌──────────────┼─────────────┼───────────┤ ┌────────────┼─────────────┼───────────┤
 │ Actual MSS 0 │   47 (TN)   │   3 (FP)  │ │ Actual MSS │   47 (TN)   │   3 (FP)  │
 ├──────────────┼─────────────┼───────────┤ ├────────────┼─────────────┼───────────┤
 │ Actual MSI 1 │    4 (FN)   │  12 (TP)  │ │ Actual MSI │    4 (FN)   │  12 (TP)  │
 └──────────────┴─────────────┴───────────┘ └────────────┴─────────────┴───────────┘
```

| Metric | FP32 Baseline | FP16 Candidate | Difference / Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 89.39% (59/66) | 89.39% (59/66) | **0.00% (Identical)** |
| **Sensitivity (Recall)** | 75.00% (12/16) | 75.00% (12/16) | **0.00% (Identical)** |
| **Specificity** | 94.00% (47/50) | 94.00% (47/50) | **0.00% (Identical)** |
| **Positive Predictive Value (PPV)** | 80.00% (12/15) | 80.00% (12/15) | **0.00% (Identical)** |
| **Negative Predictive Value (NPV)** | 92.16% (47/51) | 92.16% (47/51) | **0.00% (Identical)** |
| **Balanced Accuracy** | 84.50% | 84.50% | **0.00% (Identical)** |
| **MSI F1-Score** | 0.7742 | 0.7742 | **0.00% (Identical)** |
| **MSS F1-Score** | 0.9307 | 0.9307 | **0.00% (Identical)** |

### Protocol 2 (Uncapped Candidate Evaluation):
* **Actual Cohort Distribution:** MSS: 50 cases, MSI: 16 cases

```
                   FP32 Reference Model                     FP16 Converted Model
                ┌─────────────────────────┐              ┌─────────────────────────┐
                │ Predicted 0 │ Predicted 1│              │ Predicted 0 │ Predicted 1│
 ┌──────────────┼─────────────┼───────────┤ ┌────────────┼─────────────┼───────────┤
 │ Actual MSS 0 │   46 (TN)   │   4 (FP)  │ │ Actual MSS │   46 (TN)   │   4 (FP)  │
 ├──────────────┼─────────────┼───────────┤ ├────────────┼─────────────┼───────────┤
 │ Actual MSI 1 │    4 (FN)   │  12 (TP)  │ │ Actual MSI │    4 (FN)   │  12 (TP)  │
 └──────────────┴─────────────┴───────────┘ └────────────┴─────────────┴───────────┘
```

| Metric | FP32 Baseline | FP16 Candidate | Difference / Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 87.88% (58/66) | 87.88% (58/66) | **0.00% (Identical)** |
| **Sensitivity (Recall)** | 75.00% (12/16) | 75.00% (12/16) | **0.00% (Identical)** |
| **Specificity** | 92.00% (46/50) | 92.00% (46/50) | **0.00% (Identical)** |
| **Positive Predictive Value (PPV)** | 75.00% (12/16) | 75.00% (12/16) | **0.00% (Identical)** |
| **Negative Predictive Value (NPV)** | 92.00% (46/50) | 92.00% (46/50) | **0.00% (Identical)** |
| **Balanced Accuracy** | 83.50% | 83.50% | **0.00% (Identical)** |
| **MSI F1-Score** | 0.7500 | 0.7500 | **0.00% (Identical)** |
| **MSS F1-Score** | 0.9200 | 0.9200 | **0.00% (Identical)** |

---

## 4. Complete 66-Case Protocol 1 Numerical Drift Table

| De-identified ID | Patches | Ground Truth | FP32 Prob | FP16 Prob | Prob Drift | Logit Drift | Dist to $\tau$ (FP16) | FP32 Call | FP16 Call | Agreement? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CASE_01` | 345 | MSI (1) | 0.990385 | 0.990234 | 1.51e-04 | 1.91e-03 | +0.1602 | 1 | 1 | PASS |
| `CASE_02` | 541 | MSS (0) | 0.162250 | 0.162231 | 1.81e-05 | 1.68e-05 | -0.6678 | 0 | 0 | PASS |
| `CASE_03` | 452 | MSI (1) | 0.937404 | 0.937500 | 9.57e-05 | 6.12e-04 | +0.1075 | 1 | 1 | PASS |
| `CASE_04` | 61 | MSS (0) | 0.013114 | 0.013115 | 8.05e-07 | 5.52e-04 | -0.8169 | 0 | 0 | PASS |
| `CASE_05` | 331 | MSS (0) | 0.011473 | 0.011475 | 2.00e-06 | 7.79e-04 | -0.8185 | 0 | 0 | PASS |
| `CASE_06` | 308 | MSS (0) | 0.003447 | 0.003447 | 1.49e-07 | 1.08e-03 | -0.8266 | 0 | 0 | PASS |
| `CASE_07` | 516 | MSS (0) | 0.091230 | 0.091248 | 1.75e-05 | 1.20e-04 | -0.7388 | 0 | 0 | PASS |
| `CASE_08` | 299 | MSS (0) | 0.017761 | 0.017761 | 5.36e-07 | 1.13e-03 | -0.8122 | 0 | 0 | PASS |
| `CASE_09` | 94 | MSS (0) | 0.065060 | 0.065002 | 5.73e-05 | 8.39e-04 | -0.7650 | 0 | 0 | PASS |
| `CASE_10` | 52 | MSS (0) | 0.014129 | 0.014122 | 7.39e-06 | 8.27e-04 | -0.8159 | 0 | 0 | PASS |
| `CASE_11` | 279 | MSS (0) | 0.021346 | 0.021347 | 1.22e-06 | 1.10e-03 | -0.8087 | 0 | 0 | PASS |
| `CASE_12` | 190 | MSS (0) | 0.006736 | 0.006737 | 5.96e-07 | 1.32e-03 | -0.8233 | 0 | 0 | PASS |
| `CASE_13` | 271 | MSI (1) | 0.994391 | 0.994629 | 2.38e-04 | 1.99e-03 | +0.1646 | 1 | 1 | PASS |
| `CASE_14` | 370 | MSS (0) | 0.015399 | 0.015411 | 1.19e-05 | 1.66e-03 | -0.8146 | 0 | 0 | PASS |
| `CASE_15` | 40 | MSS (0) | 0.006374 | 0.006374 | 6.56e-07 | 1.60e-03 | -0.8236 | 0 | 0 | PASS |
| `CASE_16` | 60 | MSS (0) | 0.002760 | 0.002760 | 3.28e-07 | 9.69e-04 | -0.8272 | 0 | 0 | PASS |
| `CASE_17` | 162 | MSS (0) | 0.041518 | 0.041534 | 1.65e-05 | 5.53e-04 | -0.7885 | 0 | 0 | PASS |
| `CASE_18` | 52 | MSS (0) | 0.558614 | 0.558594 | 2.04e-05 | 7.99e-04 | -0.2714 | 0 | 0 | PASS |
| `CASE_19` | 1000 | MSI (1) | 0.985367 | 0.985352 | 1.55e-05 | 1.20e-03 | +0.1554 | 1 | 1 | PASS |
| `CASE_20` | 1000 | MSS (0) | 0.728124 | 0.728027 | 9.70e-05 | 2.64e-04 | -0.1020 | 0 | 0 | PASS |
| `CASE_21` | 406 | MSS (0) | 0.002370 | 0.002369 | 8.34e-07 | 3.82e-04 | -0.8276 | 0 | 0 | PASS |
| `CASE_22` | 70 | MSS (0) | 0.001834 | 0.001834 | 1.79e-07 | 1.41e-03 | -0.8282 | 0 | 0 | PASS |
| `CASE_23` | 220 | MSS (0) | 0.295826 | 0.295898 | 7.26e-05 | 6.72e-05 | -0.5341 | 0 | 0 | PASS |
| `CASE_24` | 10 | MSS (0) | 0.031855 | 0.031891 | 3.55e-05 | 1.12e-04 | -0.7981 | 0 | 0 | PASS |
| `CASE_25` | 370 | MSI (1) | 0.975944 | 0.976074 | 1.30e-04 | 9.42e-05 | +0.1461 | 1 | 1 | PASS |
| `CASE_26` | 1000 | MSS (0) | 0.989148 | 0.989258 | 1.10e-04 | 7.66e-04 | +0.1593 | 1 | 1 | PASS |
| `CASE_27` | 913 | MSI (1) | 0.146424 | 0.146484 | 6.08e-05 | 1.21e-03 | -0.6835 | 0 | 0 | PASS |
| `CASE_28` | 35 | MSS (0) | 0.051696 | 0.051697 | 1.01e-06 | 8.58e-04 | -0.7783 | 0 | 0 | PASS |
| `CASE_29` | 290 | MSS (0) | 0.035441 | 0.035461 | 1.99e-05 | 1.05e-03 | -0.7945 | 0 | 0 | PASS |
| `CASE_30` | 98 | MSS (0) | 0.069602 | 0.069580 | 2.18e-05 | 9.29e-04 | -0.7604 | 0 | 0 | PASS |
| `CASE_31` | 139 | MSS (0) | 0.065912 | 0.065918 | 6.29e-06 | 8.65e-04 | -0.7641 | 0 | 0 | PASS |
| `CASE_32` | 262 | MSS (0) | 0.070781 | 0.070740 | 4.12e-05 | 5.35e-04 | -0.7593 | 0 | 0 | PASS |
| `CASE_33` | 858 | MSI (1) | 0.992237 | 0.992188 | 4.98e-05 | 9.20e-04 | +0.1622 | 1 | 1 | PASS |
| `CASE_34` | 232 | MSI (1) | 0.989086 | 0.989258 | 1.72e-04 | 1.10e-03 | +0.1593 | 1 | 1 | PASS |
| `CASE_35` | 361 | MSI (1) | 0.860587 | 0.860840 | 2.52e-04 | 1.36e-04 | +0.0308 | 1 | 1 | PASS |
| `CASE_36` | 80 | MSI (1) | 0.830106 | 0.830078 | 2.84e-05 | 4.45e-04 | +0.0001 | 1 | 1 | PASS |
| `CASE_37` | 35 | MSS (0) | 0.070763 | 0.070801 | 3.82e-05 | 8.15e-04 | -0.7592 | 0 | 0 | PASS |
| `CASE_38` | 190 | MSS (0) | 0.178422 | 0.178467 | 4.48e-05 | 7.08e-04 | -0.6515 | 0 | 0 | PASS |
| `CASE_39` | 121 | MSS (0) | 0.002019 | 0.002018 | 7.45e-07 | 1.47e-04 | -0.8280 | 0 | 0 | PASS |
| `CASE_40` | 542 | MSS (0) | 0.377595 | 0.377686 | 9.04e-05 | 1.25e-05 | -0.4523 | 0 | 0 | PASS |
| `CASE_41` | 619 | MSI (1) | 0.319604 | 0.319580 | 2.40e-05 | 2.21e-04 | -0.5104 | 0 | 0 | PASS |
| `CASE_42` | 169 | MSS (0) | 0.002958 | 0.002958 | 2.38e-07 | 4.72e-05 | -0.8270 | 0 | 0 | PASS |
| `CASE_43` | 1000 | MSS (0) | 0.332311 | 0.332275 | 3.61e-05 | 4.93e-04 | -0.4977 | 0 | 0 | PASS |
| `CASE_44` | 283 | MSS (0) | 0.911503 | 0.911621 | 1.18e-04 | 9.32e-05 | +0.0816 | 1 | 1 | PASS |
| `CASE_45` | 155 | MSS (0) | 0.003557 | 0.003557 | 4.17e-07 | 1.38e-03 | -0.8264 | 0 | 0 | PASS |
| `CASE_46` | 613 | MSS (0) | 0.002325 | 0.002325 | 2.98e-07 | 9.41e-04 | -0.8277 | 0 | 0 | PASS |
| `CASE_47` | 703 | MSI (1) | 0.991042 | 0.991211 | 1.69e-04 | 8.34e-04 | +0.1612 | 1 | 1 | PASS |
| `CASE_48` | 220 | MSS (0) | 0.006376 | 0.006378 | 1.94e-06 | 1.91e-03 | -0.8236 | 0 | 0 | PASS |
| `CASE_49` | 1000 | MSI (1) | 0.130220 | 0.130249 | 2.92e-05 | 3.98e-04 | -0.6998 | 0 | 0 | PASS |
| `CASE_50` | 356 | MSS (0) | 0.012500 | 0.012505 | 4.53e-06 | 1.65e-03 | -0.8175 | 0 | 0 | PASS |
| `CASE_51` | 1000 | MSS (0) | 0.130048 | 0.130127 | 7.90e-05 | 1.44e-04 | -0.6999 | 0 | 0 | PASS |
| `CASE_52` | 113 | MSS (0) | 0.762421 | 0.762207 | 2.14e-04 | 1.68e-05 | -0.0678 | 0 | 0 | PASS |
| `CASE_53` | 67 | MSS (0) | 0.090774 | 0.090759 | 1.50e-05 | 4.71e-04 | -0.7392 | 0 | 0 | PASS |
| `CASE_54` | 447 | MSS (0) | 0.586794 | 0.586914 | 1.20e-04 | 3.86e-04 | -0.2431 | 0 | 0 | PASS |
| `CASE_55` | 444 | MSS (0) | 0.002443 | 0.002443 | 2.68e-07 | 3.37e-04 | -0.8276 | 0 | 0 | PASS |
| `CASE_56` | 23 | MSS (0) | 0.938753 | 0.938965 | 2.12e-04 | 8.28e-04 | +0.1090 | 1 | 1 | PASS |
| `CASE_57` | 402 | MSS (0) | 0.014741 | 0.014748 | 6.82e-06 | 8.36e-04 | -0.8153 | 0 | 0 | PASS |
| `CASE_58` | 315 | MSI (1) | 0.187694 | 0.187744 | 4.99e-05 | 2.19e-04 | -0.6423 | 0 | 0 | PASS |
| `CASE_59` | 166 | MSS (0) | 0.817997 | 0.817871 | 1.26e-04 | 9.10e-05 | -0.0121 | 0 | 0 | PASS |
| `CASE_60` | 62 | MSS (0) | 0.001732 | 0.001732 | 2.98e-08 | 1.35e-03 | -0.8283 | 0 | 0 | PASS |
| `CASE_61` | 349 | MSS (0) | 0.027655 | 0.027664 | 8.79e-06 | 6.56e-04 | -0.8023 | 0 | 0 | PASS |
| `CASE_62` | 327 | MSS (0) | 0.005316 | 0.005318 | 2.00e-06 | 1.29e-03 | -0.8247 | 0 | 0 | PASS |
| `CASE_63` | 502 | MSI (1) | 0.986665 | 0.986816 | 1.51e-04 | 7.21e-04 | +0.1568 | 1 | 1 | PASS |
| `CASE_64` | 99 | MSS (0) | 0.011387 | 0.011398 | 1.11e-05 | 1.03e-03 | -0.8186 | 0 | 0 | PASS |
| `CASE_65` | 1000 | MSS (0) | 0.118750 | 0.118774 | 2.46e-05 | 4.16e-04 | -0.7112 | 0 | 0 | PASS |
| `CASE_66` | 1000 | MSI (1) | 0.992791 | 0.992676 | 1.15e-04 | 6.01e-04 | +0.1627 | 1 | 1 | PASS |

---

## 5. Drift Audit Conclusion

The converted uniform FP16 model ([`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)) complies with all defined FP16 numerical limits. It achieves **100.00% classification agreement with FP32 across all 66 cases under both evaluation protocols**, causing **zero classification flips**. Because uniform FP16 preserved all classifications without error, introducing a mixed-precision variant was not required.

---

## 6. Audit & Correction History (Phase 3.7 Reconciliation)

> [!NOTE]
> **Phase 3.7 Correction Record (2026-10-17):**  
> An independent cohort integrity audit revealed that Section 3 of this document originally contained a transcription error: the cohort distribution was stated as 46 MSS / 20 MSI with corresponding synthetic confusion matrices `[[44, 2], [5, 15]]` and `[[43, 3], [5, 15]]`.  
> Forensic verification confirmed that the underlying benchmark dataset (`reports/mobile_deployment/phase3_fp16_benchmark_data.json`), the holdout manifest (`results/actual_validation_keys.json`), and Table 4 of this report all preserved the authoritative 50 MSS / 16 MSI cohort with 100% agreement.  
> Section 3 has been corrected to reflect the true recalculated confusion matrices (`[[47, 3], [4, 12]]` for Protocol 1, `[[46, 4], [4, 12]]` for Protocol 2) and identical performance metrics.  
> The original pre-correction document is preserved at [`reports/mobile_deployment/phase3_2_numerical_drift_report_orig_backup.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_2_numerical_drift_report_orig_backup.md).
