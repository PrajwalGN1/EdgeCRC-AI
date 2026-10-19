# Phase 3.9 — Final FP32 Reference Value Provenance & Precision Audit

> **Document ID:** `AUDIT-PHASE3.9-FP32-REFERENCE-PROVENANCE`  
> **Repository Root:** [`d:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2)  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
> **FP32 Reference Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **FP16 Converted Model:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Audit Date:** October 17, 2026  
> **Audit Status:** **COMPLETE — PROVENANCE FULLY RESOLVED — FINAL GATE DECISION: GO**

---

## 1. Executive Summary & Provenance Resolution

Phase 3.9 was initiated to rigorously audit and resolve the subtle numerical discrepancies (at the 7th and 8th decimal places, magnitude $\sim 10^{-7}$) between the full-precision FP32 probabilities cited in the Phase 3.3 boundary-case audit ([`reports/mobile_deployment/phase3_3_boundary_case_audit.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_3_boundary_case_audit.md)) and those audited in Phase 3.7 / 3.8 from the raw Phase 3 benchmark dataset ([`reports/mobile_deployment/phase3_fp16_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_fp16_benchmark_data.json)).

### Primary Forensic Conclusions:
1. **Provenance Traced to Distinct Historical Inference Snapshots:**
   - The 10-decimal values cited in `phase3_3_boundary_case_audit.md` were directly extracted from the **Phase 2 ONNX Runtime parity execution snapshot** recorded in [`reports/mobile_deployment/phase2_parity_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_parity_benchmark_data.json) and [`reports/mobile_deployment/phase2_recovery_and_phase3_readiness.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_recovery_and_phase3_readiness.md).
   - The values audited in Phase 3.7 and Phase 3.8 were extracted from the **Phase 3 comparative FP32/FP16 execution snapshot** recorded in [`reports/mobile_deployment/phase3_fp16_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_fp16_benchmark_data.json).
2. **Mathematical Nature of the Discrepancy (1 to 2 ULPs IEEE-754 FP32 Spacing):**
   - For probability values in the binade $[0.5, 1.0)$, IEEE-754 single-precision (FP32) floating point uses exponent $e = -1$. With 23 explicit mantissa bits, the unit in the last place (ULP) spacing is exactly $\text{ULP}_{[0.5, 1.0)} = 2^{-1} \times 2^{-23} = 2^{-24} \approx 5.960464477539063 \times 10^{-8}$. (Note: machine epsilon $\epsilon_{\text{mach}} = 2^{-23} \approx 1.192093 \times 10^{-7}$ defines the spacing in binade $[1.0, 2.0)$).
   - The delta between the Phase 2 and Phase 3 FP32 inference runs is:
     - `CASE_36`: $\Delta = 0.00 \times 10^0 = \mathbf{0\text{ ULPs}}$ (**bit-identical** across both snapshots).
     - `CASE_59`: $\Delta = 5.960464 \times 10^{-8} = 1.0 \times 2^{-24} \approx \mathbf{1\text{ ULP}}$.
     - `CASE_35`: $\Delta = 1.192093 \times 10^{-7} = 2.0 \times 2^{-24} \approx \mathbf{2\text{ ULPs}}$.
     - `CASE_20` (Protocol 2 Uncapped): $\Delta = 1.192093 \times 10^{-7} = 2.0 \times 2^{-24} \approx \mathbf{2\text{ ULPs}}$ (note: Protocol 1 is bit-identical, $\Delta = 0$).
   - A plausible technical explanation for this $1\text{--}2\text{ ULP}$ difference across distinct inference sessions is floating-point non-associativity in CPU multithreaded matrix multiplication or SIMD tile reductions (AVX2). Because internal hardware SIMD traces are not directly logged in the benchmark artifacts, this mechanism is recognized as a mathematically consistent and plausible engineering explanation rather than a directly verified instruction trace.
3. **Correction to Phase 3.8 Narrative:**
   - While both snapshots round to identical 6-decimal values (`0.817997`, `0.860587`, `0.840750`), attributing the difference between two 10-decimal numbers solely to "display rounding" was imprecise. Phase 3.9 formally clarifies that these represent two distinct, valid inference executions differing by $\le 2\text{ ULPs}$ ($\le 1.192 \times 10^{-7}$).
4. **Diagnostic Invariance & Threshold Margins Confirmed:**
   - For individual discrepant cases (`CASE_59`, `CASE_35`, `CASE_20`), their distances to the $\tau = 0.83$ decision threshold ($\sim 1.08 \times 10^{-2}$ to $3.06 \times 10^{-2}$) are approximately **$90,000$ to $256,000$ times larger** than their respective snapshot deltas ($\sim 5$ orders of magnitude).
   - Across the entire boundary cohort, the smallest relevant threshold margin belongs to `CASE_36` ($|0.8301064968 - 0.83| \approx 1.065 \times 10^{-4}$), where both snapshots are bit-identical ($\Delta = 0$). Comparing the maximum snapshot difference overall ($\Delta_{\max} \approx 1.192 \times 10^{-7}$, $\approx 2\text{ ULPs}$) against this critical margin yields $\frac{1.065 \times 10^{-4}}{1.192 \times 10^{-7}} \approx \mathbf{893\times}$ ($\approx 10^{2.95}$, approximately **3 orders of magnitude**).
   - All diagnostic classifications, confusion matrices, and the zero-flip claim ($66/66 = 100.00\%$ agreement) are 100% invariant under both snapshots.
5. **Gate Recommendation:** **GO**.

---

## 2. Forensic Traceability of the Discrepant Values

The table below traces the exact numerical values across all repository benchmark snapshots:

| Case Identifier | Pathological Context | Metric Field | Phase 1 Benchmark (`phase1_fp32_benchmark_data.json`) | Phase 2 Parity Dataset (`phase2_parity_benchmark_data.json`) | Phase 3.3 Audit Report (`phase3_3_boundary_case_audit.md`) | Phase 3 FP16 Dataset (`phase3_fp16_benchmark_data.json`) | Provenance Trace |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`CASE_36`** | Protocol 1 & 2<br>$N = 80$<br>GT: MSI (1) | FP32 Prob | `0.830106` | `0.83010649681091309` | `0.8301064968` | `0.83010649681091309` | Phase 2 and Phase 3 produced **bit-identical** float32 output ($0.83010649681...$). Phase 3.3 truncated this to 10 decimals. Delta = 0 ULPs. |
| **`CASE_59`** | Protocol 1 & 2<br>$N = 166$<br>GT: MSS (0) | FP32 Prob | `0.817998` | `0.8179975152015686` | `0.8179975152` | `0.81799745559692383` | Phase 3.3 extracted Phase 2 `onnx_prob` (`0.817997515201...`) truncated to 10 decimals. Phase 3 recorded a distinct CPU inference run differing by $5.96 \times 10^{-8}$ ($\approx 1\text{ ULP}$). |
| **`CASE_35`** | Protocol 1 & 2<br>$N = 361$<br>GT: MSI (1) | FP32 Prob | `0.860587` | `0.86058723926544189` | `0.8605872393` | `0.86058735847473145` | Phase 3.3 extracted Phase 2 `onnx_prob` (`0.860587239265...`) rounded to 10 decimals. Phase 3 recorded a distinct CPU inference run differing by $1.19 \times 10^{-7}$ ($\approx 2\text{ ULPs}$). |
| **`CASE_20`** | Protocol 2<br>$N = 3,914$<br>GT: MSS (0) | FP32 Prob | `0.840750` | `0.84074997901916504` | `0.8407499790` | `0.84075009822845459` | Phase 3.3 extracted Phase 2 `onnx_prob` (`0.840749979019...`) truncated to 10 decimals. Phase 3 recorded a distinct CPU inference run differing by $1.19 \times 10^{-7}$ ($\approx 2\text{ ULPs}$). (Protocol 1 is bit-identical: `0.7281243801116943`). |

---

## 3. High-Precision Delta & ULP Analysis

To distinguish floating-point machine precision limits from manual reporting errors, probability deltas were computed using exact IEEE-754 arithmetic:

### 3.1 Binade Definition and Delta Calculations:
For probabilities in $[0.5, 1.0) = [2^{-1}, 2^0)$, IEEE-754 single-precision (FP32) floating-point numbers have exponent $e = -1$. Because the mantissa has 23 explicit fractional bits, the distance between adjacent representable numbers (1 ULP) is:
$$\text{ULP}_{[0.5, 1.0)} = 2^{-1} \times 2^{-23} = 2^{-24} \approx 5.960464477539063 \times 10^{-8}$$

1. **`CASE_36`:**
   $$\Delta(\text{Phase 2}, \text{Phase 3}) = |0.83010649681091309 - 0.83010649681091309| = \mathbf{0.00 \times 10^0} \quad (\mathbf{0.00\text{ ULPs, bit-identical}})$$
   $$\Delta(\text{Phase 3.3 cited}, \text{Phase 3 raw}) = |0.8301064968 - 0.83010649681091309| = 1.09 \times 10^{-11} \text{ (10-decimal string truncation)}$$

2. **`CASE_59`:**
   $$\Delta(\text{Phase 2}, \text{Phase 3}) = |0.8179975152015686 - 0.81799745559692383| = \mathbf{5.960464 \times 10^{-8}}$$
   $$\text{Relative to FP32 ULP in }[0.5, 1.0) \text{ } (\text{ULP} = 2^{-24} \approx 5.960464 \times 10^{-8}): \quad \frac{5.960464 \times 10^{-8}}{5.960464 \times 10^{-8}} = \mathbf{1.00 \text{ ULP}} \quad (\approx \mathbf{1\text{ ULP}})$$

3. **`CASE_35`:**
   $$\Delta(\text{Phase 2}, \text{Phase 3}) = |0.86058723926544189 - 0.86058735847473145| = \mathbf{1.192093 \times 10^{-7}}$$
   $$\text{Relative to FP32 ULP in }[0.5, 1.0) \text{ } (\text{ULP} = 2^{-24} \approx 5.960464 \times 10^{-8}): \quad \frac{1.192093 \times 10^{-7}}{5.960464 \times 10^{-8}} = \mathbf{2.00 \text{ ULPs}} \quad (\approx \mathbf{2\text{ ULPs}})$$

4. **`CASE_20` (Protocol 2 Uncapped):**
   $$\Delta(\text{Phase 2}, \text{Phase 3}) = |0.84074997901916504 - 0.84075009822845459| = \mathbf{1.192093 \times 10^{-7}}$$
   $$\text{Relative to FP32 ULP in }[0.5, 1.0) \text{ } (\text{ULP} = 2^{-24} \approx 5.960464 \times 10^{-8}): \quad \frac{1.192093 \times 10^{-7}}{5.960464 \times 10^{-8}} = \mathbf{2.00 \text{ ULPs}} \quad (\approx \mathbf{2\text{ ULPs}})$$

### 3.2 Observed Evidence vs. Inferred Plausible Cause:
- **Observed Empirical Evidence:** Benchmark records in `phase2_parity_benchmark_data.json` and `phase3_fp16_benchmark_data.json` reveal probability differences between the two execution sessions of exactly 0 to 2 ULPs ($0$ to $1.192 \times 10^{-7}$) for these boundary cases.
- **Plausible Engineering Explanation:** When ONNX Runtime CPU execution provider executes dense matrix multiplications, floating-point operations are partitioned across thread blocks and SIMD vector registers (e.g., AVX2). Because floating-point addition is non-associative ($(a + b) + c \ne a + (b + c)$), slight variations in thread scheduling or SIMD tile reduction order across separate execution sessions plausibly account for shifts of 1 to 2 units in the least significant bit (bit 23 of the mantissa). Because benchmark logs capture outputs and execution latencies rather than internal SIMD instruction traces, this mechanism is described as a plausible, standard numerical explanation consistent with IEEE-754 arithmetic, rather than a directly verified hardware trace. This variance is orders of magnitude below the project's numerical parity tolerance ($|\Delta P| \le 1.00 \times 10^{-4}$).

---

## 4. Impact on Threshold Decisions, Drift Calculations & Parity

We evaluate whether the choice between Snapshot A (Phase 2 ONNX) and Snapshot B (Phase 3 ONNX) alters any clinical decision or validation gate:

### 4.1 Threshold-0.83 Decision Stability:

| Case ID | Distance to Threshold $\tau = 0.83$ (Snapshot A: Phase 2) | Distance to Threshold $\tau = 0.83$ (Snapshot B: Phase 3) | Decision (Snapshot A) | Decision (Snapshot B) | Clinical Decision Flip? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`CASE_36`** | $+0.0001064968$ | $+0.0001064968$ | **1 (MSI / TP)** | **1 (MSI / TP)** | **NO (IDENTICAL)** |
| **`CASE_59`** | $-0.0120024848$ | $-0.0120025444$ | **0 (MSS / TN)** | **0 (MSS / TN)** | **NO (IDENTICAL)** |
| **`CASE_35`** | $+0.0305872393$ | $+0.0305873585$ | **1 (MSI / TP)** | **1 (MSI / TP)** | **NO (IDENTICAL)** |
| **`CASE_20` (P2)** | $+0.0107499790$ | $+0.0107500982$ | **1 (MSI / FP)** | **1 (MSI / FP)** | **NO (IDENTICAL)** |

### 4.2 Distance Ratio & Margin Comparison:
1. **Individual Discrepant Cases vs. Their Own Margins:**
   - `CASE_59`: Self-margin $|0.817997 - 0.83| \approx 1.200 \times 10^{-2}$. Ratio to delta ($5.960 \times 10^{-8}$): $\mathbf{201,369\times}$ ($\approx 10^{5.30}$, $\sim 5$ orders of magnitude).
   - `CASE_35`: Self-margin $|0.860587 - 0.83| \approx 3.059 \times 10^{-2}$. Ratio to delta ($1.192 \times 10^{-7}$): $\mathbf{256,585\times}$ ($\approx 10^{5.41}$, $\sim 5$ orders of magnitude).
   - `CASE_20` (P2): Self-margin $|0.840750 - 0.83| \approx 1.075 \times 10^{-2}$. Ratio to delta ($1.192 \times 10^{-7}$): $\mathbf{90,178\times}$ ($\approx 10^{4.95}$, $\sim 5$ orders of magnitude).
2. **Maximum Snapshot Delta vs. Smallest Cohort Margin:**
   - Smallest threshold margin overall: `CASE_36` margin $= |0.8301064968 - 0.83| = 1.064968 \times 10^{-4} \approx \mathbf{1.065 \times 10^{-4}}$. (Note: `CASE_36` is bit-identical across snapshots, $\Delta = 0$).
   - Maximum snapshot difference overall: $\Delta_{\max} = 1.192093 \times 10^{-7}$ ($\approx 2\text{ ULPs}$).
   - Ratio of smallest cohort margin to maximum snapshot delta:
     $$\frac{1.064968 \times 10^{-4}}{1.192093 \times 10^{-7}} \approx \mathbf{893.36\times} \quad (\approx 10^{2.95}, \approx \mathbf{3\text{ orders of magnitude}})$$
   - *Audit Finding:* Whether evaluated against individual case margins ($\sim 5$ orders of magnitude) or against the tightest boundary margin in the cohort ($\approx 893\times$, $\sim 3$ orders of magnitude), the numerical difference between snapshots is orders of magnitude smaller than the threshold margin. Diagnostic classifications remain 100% stable and invariant.

### 4.3 Recalculated Numerical Drift to FP16:

The uniform FP16 model outputs are exact IEEE-754 half-precision values:
* `CASE_36`: $\text{Prob}_{\text{FP16}} = 0.830078125$
* `CASE_59`: $\text{Prob}_{\text{FP16}} = 0.81787109375$
* `CASE_35`: $\text{Prob}_{\text{FP16}} = 0.86083984375$
* `CASE_20` (P2): $\text{Prob}_{\text{FP16}} = 0.8408203125$

Recalculating absolute probability drift against both reference snapshots:

| Case ID | Drift vs Snapshot A (Phase 2 ONNX / Phase 3.3) | Drift vs Snapshot B (Phase 3 ONNX / Phase 3.7 / 3.8) | Delta between Drift Calculations | Gate Compliance ($|\Delta P| \le 1.00 \times 10^{-3}$) |
| :---: | :---: | :---: | :---: | :---: |
| **`CASE_36`** | $2.837181 \times 10^{-5}$ | $2.837181 \times 10^{-5}$ | $0.00 \times 10^0$ | **PASS (35x margin)** |
| **`CASE_59`** | $1.264215 \times 10^{-4}$ | $1.263618 \times 10^{-4}$ | $5.96 \times 10^{-8}$ | **PASS (7.9x margin)** |
| **`CASE_35`** | $2.526045 \times 10^{-4}$ | $2.524853 \times 10^{-4}$ | $1.19 \times 10^{-7}$ | **PASS (4.0x margin)** |
| **`CASE_20` (P2)** | $7.033348 \times 10^{-5}$ | $7.021427 \times 10^{-5}$ | $1.19 \times 10^{-7}$ | **PASS (14.2x margin)** |

* **Audit Finding:** Under both snapshots, the drift remains well below the $1.00 \times 10^{-3}$ acceptance limit. The FP32-to-FP16 zero-flip parity claim (**66/66 cases, 0 flips**) is strictly upheld under both snapshots.

---

## 5. Formal Correction to Phase 3.8 Audit Narrative

In [`reports/mobile_deployment/phase3_8_final_boundary_consistency_audit.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_8_final_boundary_consistency_audit.md), Section 4 stated:
> *"All differences across all reports are 100% explained by standard decimal rounding. No mathematical or numerical inconsistencies exist between the raw benchmark data, the machine-readable inventories, and the Markdown report presentations."*

### Phase 3.9 Clarification & Correction:
* **True for 6-decimal values:** In all narrative tables where probabilities were presented to 6 decimal places (`0.817997`, `0.860587`, `0.840750`), the difference was indeed solely display rounding.
* **Refinement for 10-decimal values:** In Section 2 of `phase3_3_boundary_case_audit.md`, the values displayed to 10 decimal digits (`0.8179975152`, `0.8605872393`, `0.8407499790`) were derived from the **Phase 2 ONNX benchmark snapshot**, whereas the raw data audited in Phase 3.8 was from the **Phase 3 ONNX benchmark snapshot**.
* The difference between these snapshots ($\le 1.192 \times 10^{-7}$, $\le 2\text{ ULPs}$) is consistent with single-precision floating-point non-associativity across execution runs, not a decimal rounding artifact.
* Both snapshots are mathematically valid representations of the frozen PyTorch FP32 model and yield bit-identical diagnostic classifications.

---

## 6. Cryptographic Asset Integrity Confirmation

All 5 protected artifacts were verified via SHA-256 during Phase 3.9 / 3.9.1. No files were modified, overwritten, or reconverted:

| Target Artifact Path | Approved Baseline SHA-256 Digest | Re-verified SHA-256 Digest (Phase 3.9.1 Recalculated) | Integrity Status |
| :--- | :--- | :--- | :---: |
| [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth) | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **INTACT** |
| [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx) | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **INTACT** |
| [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx) | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | **INTACT** |
| [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json) | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | **INTACT** |
| [`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json) | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | **INTACT** |

---

## 7. Confusion Matrices & Parity Reconfirmation

Both evaluation protocols maintain 100% classification agreement across snapshots:

### 7.1 Protocol 1 (Capped, $N \le 500$ patches):
- **Authoritative Confusion Matrix:** $\text{TN} = 47, \text{FP} = 3, \text{FN} = 4, \text{TP} = 12$
- **Total Cases:** 66 ($50\text{ MSS}, 16\text{ MSI}$)
- **Performance Metrics:** Accuracy = $89.39\%$ ($59/66$), Sensitivity = $75.00\%$ ($12/16$), Specificity = $94.00\%$ ($47/50$)
- **Snapshot Agreement:** Phase 2 ONNX vs Phase 3 FP32: **66/66 (100.00%)**; Phase 3 FP32 vs Phase 3 FP16: **66/66 (100.00%)**.

### 7.2 Protocol 2 (Uncapped, all patches):
- **Authoritative Confusion Matrix:** $\text{TN} = 46, \text{FP} = 4, \text{FN} = 4, \text{TP} = 12$
- **Total Cases:** 66 ($50\text{ MSS}, 16\text{ MSI}$)
- **Performance Metrics:** Accuracy = $87.88\%$ ($58/66$), Sensitivity = $75.00\%$ ($12/16$), Specificity = $92.00\%$ ($46/50$)
- **Snapshot Agreement:** Phase 2 ONNX vs Phase 3 FP32: **66/66 (100.00%)**; Phase 3 FP32 vs Phase 3 FP16: **66/66 (100.00%)**.

---

## 8. Revision History & Phase 3.9.1 Correction

| Revision | Date | Author / Auditor | Scope and Key Changes |
| :---: | :---: | :---: | :--- |
| **v1.0** | October 17, 2026 | Antigravity Numerical Auditor | Initial Phase 3.9 release resolving provenance of Phase 3.3 values to Phase 2 ONNX parity benchmark data. |
| **v1.1 (Phase 3.9.1)** | October 17, 2026 | Antigravity Numerical Auditor | **Phase 3.9.1 Numerical Wording Correction:**<br>1. Corrected ULP spacing definition for probability interval $[0.5, 1.0)$ to $2^{-24} \approx 5.960464 \times 10^{-8}$ (distinguishing it from machine epsilon $2^{-23}$ in $[1.0, 2.0)$).<br>2. Updated reported case deltas: `CASE_59` $\approx 1\text{ ULP}$, `CASE_35` $\approx 2\text{ ULPs}$, `CASE_20` (P2) $\approx 2\text{ ULPs}$, `CASE_36` $= 0\text{ ULPs}$ (bit-identical).<br>3. Clarified threshold margin comparisons: individual case self-margins are $\sim 90,000\times\text{--}256,000\times$ larger than snapshot deltas ($\sim 5$ orders of magnitude); smallest cohort margin (`CASE_36`, $\approx 1.065 \times 10^{-4}$) is $\approx 893.36\times$ larger than maximum snapshot delta ($\approx 3$ orders of magnitude).<br>4. Preserved distinction between observed evidence (output float records) and inferred cause (CPU SIMD non-associativity framed as a plausible technical explanation).<br>5. Recalculated SHA-256 hashes for all 5 protected artifacts and reconfirmed confusion matrices. |

---

## 9. Final Gate Recommendation: GO

* **Provenance Identified:** Every discrepant value in `phase3_3_boundary_case_audit.md` has been conclusively traced to the Phase 2 ONNX benchmark snapshot ([`phase2_parity_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_parity_benchmark_data.json)).
* **ULP Bound Verified:** The delta between the Phase 2 snapshot and Phase 3 snapshot is strictly bounded by $\le 2\text{ ULPs}$ ($\le 1.192 \times 10^{-7}$), well within theoretical IEEE-754 single-precision float bounds.
* **Zero Decision Flips:** All 66 cases retain bit-identical classifications and confusion matrices under both snapshots.
* **Artifact Security:** All models, holdout keys, and manifests are verified 100% bit-intact via fresh SHA-256 recalculation.

**GATE RECOMMENDATION: GO.**  
Phase 3.9 / 3.9.1 is formally concluded. All numerical evidence is fully reconciled and accounted for. The repository is cleared for Phase 4 Android Edge Integration.

