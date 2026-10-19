# Phase 3.9.1 — Numerical Wording Correction & Phase 4 Readiness Gate

> **Document ID:** `AUDIT-PHASE3.9.1-NUMERICAL-WORDING-CORRECTION`  
> **Repository Root:** [`d:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2)  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
> **FP32 Reference Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **FP16 Target Model:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Audit Date:** October 17, 2026  
> **Audit Status:** **COMPLETE — MATHEMATICALLY RECONCILED — PHASE 4 GATE: GO**

---

## 1. Executive Summary

This audit note formally records the numerical corrections and precision wording refinements applied to [`reports/mobile_deployment/phase3_9_fp32_reference_provenance_audit.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_9_fp32_reference_provenance_audit.md).

No model weights, ONNX graphs, input tensors, evaluation manifests, holdout labels, or benchmark JSON datasets were modified during this audit. All analyses were conducted strictly read-only against the existing frozen benchmark files:
- Phase 2 ONNX Parity Benchmark: [`reports/mobile_deployment/phase2_parity_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_parity_benchmark_data.json)
- Phase 3 Comparative FP16 Benchmark: [`reports/mobile_deployment/phase3_fp16_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_fp16_benchmark_data.json)

---

## 2. IEEE-754 FP32 Spacing in the Binade $[0.5, 1.0)$

In IEEE-754 single-precision (FP32), a normalized floating-point number is expressed as:
$$x = (-1)^s \times 2^{e - 127} \times \left(1 + \sum_{i=1}^{23} b_{23-i} 2^{-i}\right)$$

For probabilities falling in the interval $[0.5, 1.0) = [2^{-1}, 2^0)$, the exponent field is $e - 127 = -1$ (biased exponent $e = 126$). Within this binade, the unit in the last place (ULP) represents:
$$\text{ULP}_{[0.5, 1.0)} = 2^{-1} \times 2^{-23} = 2^{-24} = 5.9604644775390625 \times 10^{-8}$$

*Distinction Note:* The standard machine epsilon $\epsilon_{\text{mach}} = 2^{-23} \approx 1.192093 \times 10^{-7}$ represents the spacing between numbers in the binade $[1.0, 2.0)$. In the probability binade $[0.5, 1.0)$, the spacing is exactly half of $\epsilon_{\text{mach}}$, namely $2^{-24}$.

---

## 3. Exact Recomputed Case Deltas & ULP Discrepancies

Comparing the Phase 2 ONNX benchmark run (`onnx_prob`) against the Phase 3 FP32 benchmark run (`fp32_prob`):

| Case Identifier | Protocol Context | Phase 2 Snapshot (`onnx_prob`) | Phase 3 Snapshot (`fp32_prob`) | Absolute Delta $|\Delta P|$ | Exact ULP Count ($\Delta / 2^{-24}$) | Rounded ULP Statement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CASE_36`** | Protocol 1 & 2 ($N=80$) | `0.83010649681091309` | `0.83010649681091309` | $0.000000 \times 10^0$ | $0.000000\text{ ULPs}$ | **Bit-identical (0 ULPs)** |
| **`CASE_59`** | Protocol 1 & 2 ($N=166$) | `0.81799751520156860` | `0.81799745559692383` | $5.960464 \times 10^{-8}$ | $1.000000\text{ ULP}$ | **$\approx 1\text{ ULP}$** |
| **`CASE_35`** | Protocol 1 & 2 ($N=361$) | `0.86058723926544189` | `0.86058735847473145` | $1.192093 \times 10^{-7}$ | $2.000000\text{ ULPs}$ | **$\approx 2\text{ ULPs}$** |
| **`CASE_20`** | Protocol 2 ($N=3,914$) | `0.84074997901916504` | `0.84075009822845459` | $1.192093 \times 10^{-7}$ | $2.000000\text{ ULPs}$ | **$\approx 2\text{ ULPs}$** |
| **`CASE_20`** | Protocol 1 ($N=500$) | `0.72812438011169430` | `0.72812438011169430` | $0.000000 \times 10^0$ | $0.000000\text{ ULPs}$ | **Bit-identical (0 ULPs)** |

---

## 4. Distance to Decision Threshold ($\tau = 0.83$) & Margin Ratio Breakdown

Each case's distance to the clinical decision threshold $\tau = 0.83$ is computed for both snapshots:
$$M = |P - 0.83|$$

| Case Identifier | Ground Truth | Distance $M$ (Snapshot A: Phase 2) | Distance $M$ (Snapshot B: Phase 3) | Ratio: Self-Margin to Snapshot Delta ($M / \Delta$) | Order of Magnitude (Self-Margin) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`CASE_36`** | MSI (1) | $0.0001064968$ | $0.0001064968$ | N/A ($\Delta = 0$) | Boundary Reference ($1.065 \times 10^{-4}$) |
| **`CASE_59`** | MSS (0) | $0.0120024848$ | $0.0120025444$ | $201,369.28\times$ | $\approx 10^{5.30}$ ($\sim 5$ orders of magnitude) |
| **`CASE_35`** | MSI (1) | $0.0305872393$ | $0.0305873585$ | $256,585.36\times$ | $\approx 10^{5.41}$ ($\sim 5$ orders of magnitude) |
| **`CASE_20` (P2)**| MSS (0) | $0.0107499790$ | $0.0107500982$ | $90,178.36\times$ | $\approx 10^{4.95}$ ($\sim 5$ orders of magnitude) |

### Global Margin vs. Maximum Snapshot Difference:
- **Maximum Snapshot Difference:** $\Delta_{\max} = 1.1920928955078125 \times 10^{-7} \approx 2\text{ ULPs}$
- **Smallest Relevant Cohort Margin:** `CASE_36` margin $M_{\min} = 1.0649681091312591 \times 10^{-4} \approx 1.065 \times 10^{-4}$
- **Actual Ratio:**
  $$\frac{M_{\min}}{\Delta_{\max}} = \frac{1.064968 \times 10^{-4}}{1.192093 \times 10^{-7}} = \mathbf{893.36\times} \quad (\approx 10^{2.951})$$

### Precision Wording Clarification:
The previous statement that numerical differences are "five orders of magnitude smaller than the threshold margin" applies to each discrepant case evaluated against its **own** threshold margin ($\sim 90,000\times$ to $256,000\times$, $\sim 5$ orders of magnitude). When comparing the maximum snapshot delta ($\approx 1.192 \times 10^{-7}$) against the **tightest threshold margin in the cohort** (`CASE_36`, $\approx 1.065 \times 10^{-4}$), the margin is approximately **893 times larger ($\sim 3$ orders of magnitude)**.

Because `CASE_36` itself is bit-identical between snapshots ($\Delta = 0$), and because all other cases are separated from $\tau$ by $> 10^{-2}$, zero classification flips occur across the cohort under either snapshot.

---

## 5. Distinction: Observed Evidence vs. Inferred Plausible Cause

To adhere to rigorous validation standards, this audit formally separates directly observed empirical evidence from inferred causal mechanisms:
1. **Directly Observed Evidence:**
   The benchmark JSON records `phase2_parity_benchmark_data.json` and `phase3_fp16_benchmark_data.json` demonstrate that identical model weights and input vectors evaluated across distinct execution runs produced output probabilities differing by 0 to 2 ULPs ($0$ to $1.192 \times 10^{-7}$).
2. **Inferred Plausible Root Cause:**
   ONNX Runtime CPU execution provider executes dense matrix multiplications by tiling across threads and SIMD vector registers (AVX2). In floating-point arithmetic, addition is non-associative ($(a + b) + c \ne a + (b + c)$). Minor variations in thread scheduling or accumulation order between independent benchmark sessions represent a standard, mathematically plausible engineering explanation. However, because benchmark logs record final floating-point values and latencies rather than register-level SIMD instruction traces, this mechanism is documented as a **plausible explanation**, not a directly proven fact.

---

## 6. Cryptographic Integrity Re-Verification

All five protected repository artifacts were re-verified via direct SHA-256 recalculation during this audit:

| Target Artifact Path | Authoritative SHA-256 Digest | Audit Session Recalculated SHA-256 | Verification Result |
| :--- | :--- | :--- | :---: |
| [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth) | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **PASS (BIT-IDENTICAL)** |
| [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx) | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **PASS (BIT-IDENTICAL)** |
| [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx) | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | **PASS (BIT-IDENTICAL)** |
| [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json) | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | **PASS (BIT-IDENTICAL)** |
| [`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json) | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | **PASS (BIT-IDENTICAL)** |

---

## 7. Metrics & Classification Reconfirmation

Both evaluation protocols maintain 100% agreement across all benchmark runs:

1. **Protocol 1 (Capped, $N \le 500$):**
   - Confusion Matrix: $\text{TN}=47, \text{FP}=3, \text{FN}=4, \text{TP}=12$
   - Performance: Accuracy = $89.39\%$ ($59/66$), Sensitivity = $75.00\%$ ($12/16$), Specificity = $94.00\%$ ($47/50$)
   - Agreement: Phase 2 ONNX vs Phase 3 FP32: **66/66 (100.00%)**; Phase 3 FP32 vs Phase 3 FP16: **66/66 (100.00%)**.

2. **Protocol 2 (Uncapped, all patches):**
   - Confusion Matrix: $\text{TN}=46, \text{FP}=4, \text{FN}=4, \text{TP}=12$
   - Performance: Accuracy = $87.88\%$ ($58/66$), Sensitivity = $75.00\%$ ($12/16$), Specificity = $92.00\%$ ($46/50$)
   - Agreement: Phase 2 ONNX vs Phase 3 FP32: **66/66 (100.00%)**; Phase 3 FP32 vs Phase 3 FP16: **66/66 (100.00%)**.

---

## 8. Final Gate Recommendation

- **Numerical Wording Corrected:** The ULP definitions and threshold margin ratios have been updated to exact IEEE-754 FP32 spacing ($2^{-24} \approx 5.960 \times 10^{-8}$).
- **Evidence vs. Cause Disambiguated:** Inferred SIMD/CPU accumulation order is explicitly framed as a plausible explanation.
- **Zero Flips Maintained:** Diagnostic decisions are 100% invariant across all snapshots and between FP32 and FP16.
- **Artifact Security:** All models, holdout keys, and manifests are verified bit-intact.

**FINAL GATE RECOMMENDATION: GO FOR PHASE 4.**  
All numerical audits are complete and reconciled. The repository is fully ready for Phase 4 Android Edge Integration.
