# Phase 3.7 — Confusion Matrix and Cohort Integrity Reconciliation Audit

> **Document ID:** `AUDIT-PHASE3.7-COHORT-INTEGRITY-RECONCILIATION`  
> **Repository Root:** [`d:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2)  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
> **FP32 ONNX Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **FP16 ONNX Model:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Holdout Manifest:** [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json) (66 isolated validation cases)  
> **Patch Selection Manifest:** [`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json)  
> **Machine-Readable Dataset:** [`reports/mobile_deployment/phase3_7_metric_reconciliation.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_7_metric_reconciliation.json)  
> **Audit Status:** **COMPLETE — FORMAL GATE DECISION: GO**

---

## 1. Executive Summary & Gate Decision

Prior to authorizing Phase 4 Android Edge Integration, an independent reconciliation audit was executed to investigate an observed inconsistency between the frozen Phase 1 FP32 reference benchmark report and Section 3 of the Phase 3 FP16 numerical drift report:

1. **Observed Inconsistency:**  
   In [`reports/mobile_deployment/phase1_fp32_reference_benchmark.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase1_fp32_reference_benchmark.md), the 66-case holdout cohort is recorded as **50 MSS cases and 16 MSI cases**, with Protocol 1 confusion matrix `[[47, 3], [4, 12]]` and Protocol 2 confusion matrix `[[46, 4], [4, 12]]`. Conversely, in Section 3 of [`reports/mobile_deployment/phase3_2_numerical_drift_report.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_2_numerical_drift_report.md), the text reported an altered cohort distribution of **46 MSS cases and 20 MSI cases**, with confusion matrices `[[44, 2], [5, 15]]` and `[[43, 3], [5, 15]]`.

2. **Forensic Root Cause Determination:**  
   The discrepancy was **strictly an isolated documentation transcription error** in Section 3 of `phase3_2_numerical_drift_report.md`. The author erroneously assumed 20 MSI cases and back-calculated synthetic confusion matrix entries from the aggregate sensitivity percentage ($75.00\% = 15/20$).  
   Forensic audit proved that **the underlying data never changed**:
   - The authoritative patient registry ([`data/patient_registry.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/patient_registry.json)), holdout keys ([`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json)), patch selection manifest, Phase 1 benchmark dataset, Phase 2 parity dataset, and Phase 3 FP16 benchmark dataset ([`reports/mobile_deployment/phase3_fp16_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_fp16_benchmark_data.json)) are **100% unanimous**: exactly **50 MSS (Class 0) and 16 MSI (Class 1)** cases.
   - Table 4 in `phase3_2_numerical_drift_report.md` itself lists all 66 cases with the exact 50 MSS and 16 MSI labels, yielding the true confusion matrices `[[47, 3], [4, 12]]` and `[[46, 4], [4, 12]]`.

3. **Protection of Artifacts:**  
   No model weights, ONNX graphs, holdout manifests, or patch manifests were modified, reconverted, or retrained. All SHA-256 cryptographic fingerprints match Phase 1 and Phase 2 baselines with zero bit drift.

4. **Gate Recommendation:** **GO**.  
   The reporting error in `phase3_2_numerical_drift_report.md` has been corrected with full traceability, the original report has been preserved, all derived metrics are completely reconciled, and the codebase is 100% test-verified (53/53 tests pass).

---

## 2. Protected Artifact Integrity Verification

In accordance with strict deployment governance, all foundational model checkpoints and evaluation manifests were audited via cryptographic SHA-256 digests. None of these files were overwritten or modified.

### Hash Verification Log:

| Artifact Path | Expected / Authorized SHA-256 | Recomputed SHA-256 (Phase 3.7) | Status |
| :--- | :--- | :--- | :---: |
| [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth) | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **INTACT** |
| [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx) | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **INTACT** |
| [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx) | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | **INTACT** |
| [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json) | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | **INTACT** |
| [`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json) | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | **INTACT** |

*Verification Command:*
```powershell
python -c "import hashlib; files=['models/fusion_model_v1_1.pth','models/fusion_model_fp32.onnx','models/fusion_model_fp16.onnx','results/actual_validation_keys.json','reports/mobile_deployment/protocol1_patch_selection_manifest.json']; print({f: hashlib.sha256(open(f,'rb').read()).hexdigest() for f in files})"
```

---

## 3. Authoritative Frozen Evaluation Cohort Integrity

To establish ground-truth clinical cohort composition without inferring labels from model outputs, we audited the patient registry and manifests directly:

1. **Cohort Size:** Exactly **66 unique cases** (indices `CASE_01` to `CASE_66`).
2. **Authoritative Label Source:** Ground truth was extracted directly from [`data/patient_registry.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/patient_registry.json) for the keys defined in [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json).
3. **Class Breakdown:**
   - **MSS (Microsatellite Stable / Class 0):** **50 cases (75.76%)**
   - **MSI (Microsatellite Instability / Class 1):** **16 cases (24.24%)**
   - **Total Cohort:** **66 cases (100.00%)**
4. **Cross-Artifact Consensus:** Every authoritative file in the repository agrees on these labels without a single mismatch:
   - `data/patient_registry.json`: 50 MSS, 16 MSI (0 mismatches)
   - `reports/mobile_deployment/protocol1_patch_selection_manifest.json`: 50 MSS, 16 MSI (0 mismatches)
   - `reports/mobile_deployment/phase1_fp32_benchmark_data.json`: 50 MSS, 16 MSI (0 mismatches)
   - `reports/mobile_deployment/phase2_parity_benchmark_data.json`: 50 MSS, 16 MSI (0 mismatches)
   - `reports/mobile_deployment/phase3_fp16_benchmark_data.json`: 50 MSS, 16 MSI (0 mismatches)

---

## 4. Independent Recalculation of Confusion Matrices & Metrics

### 4.1 Documented Diagnostic Convention
To eliminate all ambiguity, calculations strictly adhere to the standard binary confusion matrix convention with decision threshold $\tau = 0.83$:

$$
\text{Prediction} = 
\begin{cases} 
1 \text{ (MSI)}, & \text{if } P \ge 0.83 \\ 
0 \text{ (MSS)}, & \text{if } P < 0.83 
\end{cases}
$$

* **True Negative (TN):** Actual MSS (Class 0) correctly predicted as MSS ($P < 0.83$)
* **False Positive (FP):** Actual MSS (Class 0) incorrectly predicted as MSI ($P \ge 0.83$)
* **False Negative (FN):** Actual MSI (Class 1) incorrectly predicted as MSS ($P < 0.83$)
* **True Positive (TP):** Actual MSI (Class 1) correctly predicted as MSI ($P \ge 0.83$)

### 4.2 Derived Diagnostic Metric Formulations
* $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
* $\text{Sensitivity (Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{16}$
* $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{\text{TN}}{50}$
* $\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$
* $\text{Positive Predictive Value (PPV / Precision)} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
* $\text{Negative Predictive Value (NPV)} = \frac{\text{TN}}{\text{TN} + \text{FN}}$
* $\text{MSI F1-Score} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$
* $\text{MSS F1-Score} = \frac{2 \cdot \text{TN}}{2 \cdot \text{TN} + \text{FP} + \text{FN}}$

### 4.3 Recalculated Results Across All 6 Target Configurations

All 6 target model configurations were recalculated independently from their per-case outputs:

| Evaluation Target | TN | FP | FN | TP | Total (MSS/MSI) | Accuracy | Sensitivity | Specificity | Balanced Acc | PPV (Precision) | NPV | MSI F1 | MSS F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 Protocol 1 Ref (PyTorch)** | **47** | **3** | **4** | **12** | 66 (50 / 16) | **89.39%** (59/66) | **75.00%** (12/16) | **94.00%** (47/50) | **84.50%** | **80.00%** (12/15) | **92.16%** (47/51) | **0.7742** | **0.9307** |
| **Phase 1 Protocol 2 Ref (PyTorch)** | **46** | **4** | **4** | **12** | 66 (50 / 16) | **87.88%** (58/66) | **75.00%** (12/16) | **92.00%** (46/50) | **83.50%** | **75.00%** (12/16) | **92.00%** (46/50) | **0.7500** | **0.9200** |
| **Phase 3 FP32 ONNX Protocol 1** | **47** | **3** | **4** | **12** | 66 (50 / 16) | **89.39%** (59/66) | **75.00%** (12/16) | **94.00%** (47/50) | **84.50%** | **80.00%** (12/15) | **92.16%** (47/51) | **0.7742** | **0.9307** |
| **Phase 3 FP32 ONNX Protocol 2** | **46** | **4** | **4** | **12** | 66 (50 / 16) | **87.88%** (58/66) | **75.00%** (12/16) | **92.00%** (46/50) | **83.50%** | **75.00%** (12/16) | **92.00%** (46/50) | **0.7500** | **0.9200** |
| **Phase 3 FP16 ONNX Protocol 1** | **47** | **3** | **4** | **12** | 66 (50 / 16) | **89.39%** (59/66) | **75.00%** (12/16) | **94.00%** (47/50) | **84.50%** | **80.00%** (12/15) | **92.16%** (47/51) | **0.7742** | **0.9307** |
| **Phase 3 FP16 ONNX Protocol 2** | **46** | **4** | **4** | **12** | 66 (50 / 16) | **87.88%** (58/66) | **75.00%** (12/16) | **92.00%** (46/50) | **83.50%** | **75.00%** (12/16) | **92.00%** (46/50) | **0.7500** | **0.9200** |

```
Protocol 1 (PyTorch / FP32 ONNX / FP16 ONNX):
                 Predicted MSS (0)   Predicted MSI (1)
 Actual MSS (0)       47 (TN)              3 (FP)        Total: 50
 Actual MSI (1)        4 (FN)             12 (TP)        Total: 16
 Total Predicted:     51                  15             Grand Total: 66

Protocol 2 (PyTorch / FP32 ONNX / FP16 ONNX):
                 Predicted MSS (0)   Predicted MSI (1)
 Actual MSS (0)       46 (TN)              4 (FP)        Total: 50
 Actual MSI (1)        4 (FN)             12 (TP)        Total: 16
 Total Predicted:     50                  16             Grand Total: 66
```

### 4.4 Case-by-Case FP32 vs. FP16 Agreement Verification
Rather than relying solely on aggregate metrics, all 66 cases were audited individually for classification flips:
* **Protocol 1 Case-by-Case Agreement:** **66 / 66 cases (100.00%)** — **0 flips** between FP32 and FP16.
* **Protocol 2 Case-by-Case Agreement:** **66 / 66 cases (100.00%)** — **0 flips** between FP32 and FP16.
* **Maximum Observed Probability Drift:** $2.52 \times 10^{-4}$ (Protocol 1), $2.99 \times 10^{-4}$ (Protocol 2) — well below the gate threshold of $1.00 \times 10^{-3}$.
* **Mean Observed Probability Drift:** $5.23 \times 10^{-5}$ (Protocol 1), $5.73 \times 10^{-5}$ (Protocol 2) — well below the gate threshold of $1.00 \times 10^{-4}$.

---

## 5. Discrepancy Investigation & Root Cause Analysis

### 5.1 What Occurred in the Phase 3 Report
In Section 3 of `phase3_2_numerical_drift_report.md`, the document stated:
```markdown
* Class Mapping: 0 = MSS (46 cases), 1 = MSI (20 cases)
[[44, 2], [5, 15]]  (Protocol 1)
[[43, 3], [5, 15]]  (Protocol 2)
```

### 5.2 Forensic Root Cause
1. **Source of the Erroneous Numbers:**  
   The true overall accuracy of Protocol 1 is $59/66 = 89.39\%$ and sensitivity is $12/16 = 75.00\%$.  
   During document authoring, the author mistakenly assumed an integer MSI cohort of 20 cases. Because $75\%$ of 20 equals 15, the author transcribed $\text{TP} = 15$ and $\text{FN} = 5$. To satisfy the known 59 correct cases, $\text{TN}$ was calculated as $59 - 15 = 44$. The remaining MSS cases were assumed to be $66 - 20 = 46$, leading to $\text{FP} = 46 - 44 = 2$.
2. **Proof that Underlying Data Was Not Changed:**  
   - In the exact same document, **Table 4 (Complete 66-Case Protocol 1 Numerical Drift Table)** lists every case with its actual label. Parsing Table 4 directly yields:
     $$\text{MSS Cases} = 50, \quad \text{MSI Cases} = 16, \quad \text{TN} = 47, \quad \text{FP} = 3, \quad \text{FN} = 4, \quad \text{TP} = 12$$
   - The JSON data file [`reports/mobile_deployment/phase3_fp16_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_fp16_benchmark_data.json) contains all 66 cases, with `"ground_truth_label": 0` appearing 50 times and `"ground_truth_label": 1` appearing 16 times.
   - The summary block of `phase3_fp16_benchmark_data.json` records `"flips_vs_gt": 7`, which corresponds exactly to the 7 errors in the true confusion matrix ($3\text{ FP} + 4\text{ FN} = 7$).
3. **Corrective Action Executed:**  
   - The original pre-correction report has been preserved at [`reports/mobile_deployment/phase3_2_numerical_drift_report_orig_backup.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_2_numerical_drift_report_orig_backup.md).
   - Section 3 of [`reports/mobile_deployment/phase3_2_numerical_drift_report.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_2_numerical_drift_report.md) has been updated with the verified recalculations.
   - Section 6 was appended to document the audit and revision history.

---

## 6. Complete 66-Case Audit and Metric Reconciliation Table

The table below reconciles all 66 holdout cohort cases across both protocols using de-identified identifiers (`CASE_01` to `CASE_66`):

| ID | Patches (P1/P2) | Ground Truth | P1 FP32 Prob | P1 FP16 Prob | P1 Outcome | P2 FP32 Prob | P2 FP16 Prob | P2 Outcome | Decision Agreement? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CASE_01` | 345 / 345 | MSI (1) | 0.990385 | 0.990234 | **TP** | 0.990385 | 0.990234 | **TP** | PASS (100%) |
| `CASE_02` | 541 / 541 | MSS (0) | 0.162250 | 0.162231 | **TN** | 0.162250 | 0.162231 | **TN** | PASS (100%) |
| `CASE_03` | 452 / 452 | MSI (1) | 0.937404 | 0.937500 | **TP** | 0.937404 | 0.937500 | **TP** | PASS (100%) |
| `CASE_04` | 61 / 61 | MSS (0) | 0.013114 | 0.013115 | **TN** | 0.013114 | 0.013115 | **TN** | PASS (100%) |
| `CASE_05` | 331 / 331 | MSS (0) | 0.011473 | 0.011475 | **TN** | 0.011473 | 0.011475 | **TN** | PASS (100%) |
| `CASE_06` | 308 / 308 | MSS (0) | 0.003447 | 0.003447 | **TN** | 0.003447 | 0.003447 | **TN** | PASS (100%) |
| `CASE_07` | 516 / 516 | MSS (0) | 0.091230 | 0.091248 | **TN** | 0.091230 | 0.091248 | **TN** | PASS (100%) |
| `CASE_08` | 299 / 299 | MSS (0) | 0.017761 | 0.017761 | **TN** | 0.017761 | 0.017761 | **TN** | PASS (100%) |
| `CASE_09` | 94 / 94 | MSS (0) | 0.065060 | 0.065002 | **TN** | 0.065060 | 0.065002 | **TN** | PASS (100%) |
| `CASE_10` | 52 / 52 | MSS (0) | 0.014129 | 0.014122 | **TN** | 0.014129 | 0.014122 | **TN** | PASS (100%) |
| `CASE_11` | 279 / 279 | MSS (0) | 0.021346 | 0.021347 | **TN** | 0.021346 | 0.021347 | **TN** | PASS (100%) |
| `CASE_12` | 190 / 190 | MSS (0) | 0.006736 | 0.006737 | **TN** | 0.006736 | 0.006737 | **TN** | PASS (100%) |
| `CASE_13` | 271 / 271 | MSI (1) | 0.994391 | 0.994629 | **TP** | 0.994391 | 0.994629 | **TP** | PASS (100%) |
| `CASE_14` | 370 / 370 | MSS (0) | 0.015399 | 0.015411 | **TN** | 0.015399 | 0.015411 | **TN** | PASS (100%) |
| `CASE_15` | 40 / 40 | MSS (0) | 0.006374 | 0.006374 | **TN** | 0.006374 | 0.006374 | **TN** | PASS (100%) |
| `CASE_16` | 60 / 60 | MSS (0) | 0.002760 | 0.002760 | **TN** | 0.002760 | 0.002760 | **TN** | PASS (100%) |
| `CASE_17` | 162 / 162 | MSS (0) | 0.041518 | 0.041534 | **TN** | 0.041518 | 0.041534 | **TN** | PASS (100%) |
| `CASE_18` | 52 / 52 | MSS (0) | 0.558614 | 0.558594 | **TN** | 0.558614 | 0.558594 | **TN** | PASS (100%) |
| `CASE_19` | 1000 / 1690 | MSI (1) | 0.985367 | 0.985352 | **TP** | 0.991278 | 0.991211 | **TP** | PASS (100%) |
| `CASE_20` | 1000 / 3914 | MSS (0) | 0.728124 | 0.728027 | **TN (P1) / FP (P2)** | 0.840750 | 0.840820 | **FP** | PASS (100%) |
| `CASE_21` | 406 / 406 | MSS (0) | 0.002370 | 0.002369 | **TN** | 0.002370 | 0.002369 | **TN** | PASS (100%) |
| `CASE_22` | 70 / 70 | MSS (0) | 0.001834 | 0.001834 | **TN** | 0.001834 | 0.001834 | **TN** | PASS (100%) |
| `CASE_23` | 220 / 220 | MSS (0) | 0.295826 | 0.295898 | **TN** | 0.295826 | 0.295898 | **TN** | PASS (100%) |
| `CASE_24` | 10 / 10 | MSS (0) | 0.031855 | 0.031891 | **TN** | 0.031855 | 0.031891 | **TN** | PASS (100%) |
| `CASE_25` | 370 / 370 | MSI (1) | 0.975944 | 0.976074 | **TP** | 0.975944 | 0.976074 | **TP** | PASS (100%) |
| `CASE_26` | 1000 / 1403 | MSS (0) | 0.989148 | 0.989258 | **FP** | 0.989063 | 0.989258 | **FP** | PASS (100%) |
| `CASE_27` | 913 / 913 | MSI (1) | 0.146424 | 0.146484 | **FN** | 0.146424 | 0.146484 | **FN** | PASS (100%) |
| `CASE_28` | 35 / 35 | MSS (0) | 0.051696 | 0.051697 | **TN** | 0.051696 | 0.051697 | **TN** | PASS (100%) |
| `CASE_29` | 290 / 290 | MSS (0) | 0.035441 | 0.035461 | **TN** | 0.035441 | 0.035461 | **TN** | PASS (100%) |
| `CASE_30` | 98 / 98 | MSS (0) | 0.069602 | 0.069580 | **TN** | 0.069602 | 0.069580 | **TN** | PASS (100%) |
| `CASE_31` | 139 / 139 | MSS (0) | 0.065912 | 0.065918 | **TN** | 0.065912 | 0.065918 | **TN** | PASS (100%) |
| `CASE_32` | 262 / 262 | MSS (0) | 0.070781 | 0.070740 | **TN** | 0.070781 | 0.070740 | **TN** | PASS (100%) |
| `CASE_33` | 858 / 858 | MSI (1) | 0.992237 | 0.992188 | **TP** | 0.992237 | 0.992188 | **TP** | PASS (100%) |
| `CASE_34` | 232 / 232 | MSI (1) | 0.989086 | 0.989258 | **TP** | 0.989086 | 0.989258 | **TP** | PASS (100%) |
| `CASE_35` | 361 / 361 | MSI (1) | 0.860587 | 0.860840 | **TP** | 0.860587 | 0.860840 | **TP** | PASS (100%) |
| `CASE_36` | 80 / 80 | MSI (1) | 0.830106 | 0.830078 | **TP** | 0.830106 | 0.830078 | **TP** | PASS (100%) |
| `CASE_37` | 35 / 35 | MSS (0) | 0.070763 | 0.070801 | **TN** | 0.070763 | 0.070801 | **TN** | PASS (100%) |
| `CASE_38` | 190 / 190 | MSS (0) | 0.178422 | 0.178467 | **TN** | 0.178422 | 0.178467 | **TN** | PASS (100%) |
| `CASE_39` | 121 / 121 | MSS (0) | 0.002019 | 0.002018 | **TN** | 0.002019 | 0.002018 | **TN** | PASS (100%) |
| `CASE_40` | 542 / 542 | MSS (0) | 0.377595 | 0.377686 | **TN** | 0.377595 | 0.377686 | **TN** | PASS (100%) |
| `CASE_41` | 619 / 619 | MSI (1) | 0.319604 | 0.319580 | **FN** | 0.319604 | 0.319580 | **FN** | PASS (100%) |
| `CASE_42` | 169 / 169 | MSS (0) | 0.002958 | 0.002958 | **TN** | 0.002958 | 0.002958 | **TN** | PASS (100%) |
| `CASE_43` | 1000 / 1433 | MSS (0) | 0.332311 | 0.332275 | **TN** | 0.198942 | 0.198975 | **TN** | PASS (100%) |
| `CASE_44` | 283 / 283 | MSS (0) | 0.911503 | 0.911621 | **FP** | 0.911503 | 0.911621 | **FP** | PASS (100%) |
| `CASE_45` | 155 / 155 | MSS (0) | 0.003557 | 0.003557 | **TN** | 0.003557 | 0.003557 | **TN** | PASS (100%) |
| `CASE_46` | 613 / 613 | MSS (0) | 0.002325 | 0.002325 | **TN** | 0.002325 | 0.002325 | **TN** | PASS (100%) |
| `CASE_47` | 703 / 703 | MSI (1) | 0.991042 | 0.991211 | **TP** | 0.991042 | 0.991211 | **TP** | PASS (100%) |
| `CASE_48` | 220 / 220 | MSS (0) | 0.006376 | 0.006378 | **TN** | 0.006376 | 0.006378 | **TN** | PASS (100%) |
| `CASE_49` | 1000 / 1921 | MSI (1) | 0.130220 | 0.130249 | **FN** | 0.471002 | 0.470703 | **FN** | PASS (100%) |
| `CASE_50` | 356 / 356 | MSS (0) | 0.012500 | 0.012505 | **TN** | 0.012500 | 0.012505 | **TN** | PASS (100%) |
| `CASE_51` | 1000 / 2504 | MSS (0) | 0.130048 | 0.130127 | **TN** | 0.231908 | 0.231934 | **TN** | PASS (100%) |
| `CASE_52` | 113 / 113 | MSS (0) | 0.762421 | 0.762207 | **TN** | 0.762421 | 0.762207 | **TN** | PASS (100%) |
| `CASE_53` | 67 / 67 | MSS (0) | 0.090774 | 0.090759 | **TN** | 0.090774 | 0.090759 | **TN** | PASS (100%) |
| `CASE_54` | 447 / 447 | MSS (0) | 0.586794 | 0.586914 | **TN** | 0.586794 | 0.586914 | **TN** | PASS (100%) |
| `CASE_55` | 444 / 444 | MSS (0) | 0.002443 | 0.002443 | **TN** | 0.002443 | 0.002443 | **TN** | PASS (100%) |
| `CASE_56` | 23 / 23 | MSS (0) | 0.938753 | 0.938965 | **FP** | 0.938753 | 0.938965 | **FP** | PASS (100%) |
| `CASE_57` | 402 / 402 | MSS (0) | 0.014741 | 0.014748 | **TN** | 0.014741 | 0.014748 | **TN** | PASS (100%) |
| `CASE_58` | 315 / 315 | MSI (1) | 0.187694 | 0.187744 | **FN** | 0.187694 | 0.187744 | **FN** | PASS (100%) |
| `CASE_59` | 166 / 166 | MSS (0) | 0.817997 | 0.817871 | **TN** | 0.817997 | 0.817871 | **TN** | PASS (100%) |
| `CASE_60` | 62 / 62 | MSS (0) | 0.001732 | 0.001732 | **TN** | 0.001732 | 0.001732 | **TN** | PASS (100%) |
| `CASE_61` | 349 / 349 | MSS (0) | 0.027655 | 0.027664 | **TN** | 0.027655 | 0.027664 | **TN** | PASS (100%) |
| `CASE_62` | 327 / 327 | MSS (0) | 0.005316 | 0.005318 | **TN** | 0.005316 | 0.005318 | **TN** | PASS (100%) |
| `CASE_63` | 502 / 502 | MSI (1) | 0.986665 | 0.986816 | **TP** | 0.986665 | 0.986816 | **TP** | PASS (100%) |
| `CASE_64` | 99 / 99 | MSS (0) | 0.011387 | 0.011398 | **TN** | 0.011387 | 0.011398 | **TN** | PASS (100%) |
| `CASE_65` | 1000 / 1210 | MSS (0) | 0.118750 | 0.118774 | **TN** | 0.054363 | 0.054382 | **TN** | PASS (100%) |
| `CASE_66` | 1000 / 1060 | MSI (1) | 0.992791 | 0.992676 | **TP** | 0.992388 | 0.992188 | **TP** | PASS (100%) |

---

## 7. Verified Facts vs. Assumptions Ledger

To ensure clinical governance and audit traceability, all findings are formally cataloged:

### Verified Facts:
1. **Model Weights & Checkpoints:** No model weights or checkpoints were altered, fine-tuned, or re-exported during this reconciliation. All SHA-256 hashes match bit-identically.
2. **Authoritative Cohort Composition:** The holdout set consists of exactly 66 cases (50 MSS and 16 MSI).
3. **Parity & Classification Agreement:** FP16 ONNX Runtime execution agrees 100.00% (66/66 cases, 0 flips) with FP32 across both Protocol 1 and Protocol 2.
4. **Boundary Stability:** Critical boundary case `CASE_36` evaluates to $P_{\text{FP16}} = 0.830078$, stably preserving the MSI positive diagnosis above threshold $\tau = 0.83$.
5. **Exact Confusion Matrices:**
   - Protocol 1: $\text{TN} = 47, \text{FP} = 3, \text{FN} = 4, \text{TP} = 12$ ($\text{Accuracy} = 89.39\%$).
   - Protocol 2: $\text{TN} = 46, \text{FP} = 4, \text{FN} = 4, \text{TP} = 12$ ($\text{Accuracy} = 87.88\%$).
6. **Regression Suite:** All 53 unit and regression tests pass cleanly (`python -m pytest`).

### Unsupported Assumptions (Rejected by Audit):
1. **"The cohort changed to 46 MSS / 20 MSI in Phase 3":** **REJECTED.** The underlying dataset, manifests, and per-case results never contained 46/20; it was a manual transcription error in the markdown narrative of Section 3.
2. **"Sensitivity was 75.00% because 15 out of 20 were correct":** **REJECTED.** Sensitivity was $75.00\%$ because $12$ out of $16$ were correct ($12/16 = 0.75$).

---

## 8. Final Gate Decision: GO

* **Cohort Integrity:** Confirmed (50 MSS, 16 MSI, 66 Total).
* **Ground Truth Labels:** 100% unanimous across all repository sources.
* **Confusion Matrices:** Reconciled and verified against Phase 1 golden reference.
* **FP32 vs FP16 Parity:** 100% agreement (0 flips on all 66 cases).
* **Protected Artifacts:** Intact and verified by SHA-256.
* **Reporting Error:** Documented, corrected, and original report preserved.

**GATE STATUS: GO.**  
Phase 3.7 is formally concluded. The project is fully unblocked and ready for Phase 4 Android Edge Integration.
