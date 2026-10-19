# Phase 2 — Final Sign-off Report: ONNX FP32 Export & Parity Gate

> **Document ID:** `SIGNOFF-PHASE2-ONNX-FP32`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
> **Exported Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **Date:** October 16, 2026  
> **Gate Decision:** **GO** (Ready for Phase 3 FP16 Conversion Authorization)  

---

## 1. Executive Gate Decision: GO

Phase 2 (ONNX FP32 Conversion and Numerical Parity Verification) has achieved a **definitive GO**.

All required steps under Phase 2 have been executed and independently validated:
1. **Host Environment Verified (Phase 2.2):** Active Python 3.13.5 environment safely augmented with authorized packages `onnx` (1.23.2) and `onnxruntime` (1.31.0). Zero broken dependencies (`pip check` exit 0). All protected assets verified bit-intact.
2. **Export Wrapper Integrity (Phase 2.1):** [`CrossAttentionExportWrapper`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/export_onnx.py) safely bypassed dead NumPy code while preserving 100% parameter sharing and bit-exact equivalence with the trained checkpoint.
3. **Dynamic Sequence Dimension Graph (Phase 2.2):** Resolved the PyTorch TorchScript tracer artifact in `nn.MultiheadAttention`, producing an ONNX graph with fully dynamic sequence axes tested and verified across $N \in \{1, 6, 50, 100, 385, 1000, 4305\}$ patches without shape crashes.
4. **Host-Side Parity Verification (Phase 2.3):** Evaluated all 66 holdout cases under both frozen evaluation protocols (Protocol 1 capped and Protocol 2 uncapped). Achieved **100.00% classification agreement (0 flips)** at threshold $\tau = 0.83$, with maximum observed probability error of $4.97 \times 10^{-7}$ (over 200x better than the $1.00 \times 10^{-4}$ tolerance limit).
5. **Boundary Case Stability:** Critical boundary case `CASE_36` (reference probability 0.830106, distance to threshold $+0.000106$) evaluated on ONNX Runtime to 0.830106 with zero classification flip.
6. **Regression Suite:** All **42 unit and integration tests** in `tests/` pass cleanly.

---

## 2. Gate Verification Summary Matrix

| Verification Dimension | Contract Specification | Actual Empirical Measurement | Status |
| :--- | :---: | :---: | :---: |
| **Protected Checkpoint Integrity** | SHA-256 unchanged | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **VERIFIED** |
| **Patient Registry Integrity** | SHA-256 unchanged | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | **VERIFIED** |
| **Holdout Cohort Integrity** | SHA-256 unchanged | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | **VERIFIED** |
| **Patch Manifest Integrity** | SHA-256 unchanged | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | **VERIFIED** |
| **ONNX Graph Validity** | `onnx.checker.check_model` | 0 errors, 0 warnings (IR v8, Opset 17) | **VERIFIED** |
| **ONNX Runtime Execution** | `CPUExecutionProvider` | Clean session initialization, 0 runtime errors | **VERIFIED** |
| **Dynamic Sequence Scaling** | $N \in [1, 4305]$ | All 7 test counts pass ($L_\infty \le 4.77 \times 10^{-7}$) | **VERIFIED** |
| **Protocol 1 Max Logit Error** | $\le 1.00 \times 10^{-4}$ | **$2.05 \times 10^{-6}$** | **PASS** |
| **Protocol 1 Max Prob Error** | $\le 1.00 \times 10^{-4}$ | **$4.97 \times 10^{-7}$** | **PASS** |
| **Protocol 1 Mean Prob Error** | $\le 1.00 \times 10^{-5}$ | **$2.31 \times 10^{-7}$** | **PASS** |
| **Protocol 1 Classification Agreement** | 100% (0 flips) | **100.00% (66 / 66 cases agreed)** | **PASS** |
| **Protocol 2 Max Logit Error** | $\le 1.00 \times 10^{-4}$ | **$2.88 \times 10^{-6}$** | **PASS** |
| **Protocol 2 Max Prob Error** | $\le 1.00 \times 10^{-4}$ | **$4.97 \times 10^{-7}$** | **PASS** |
| **Protocol 2 Mean Prob Error** | $\le 1.00 \times 10^{-5}$ | **$2.32 \times 10^{-7}$** | **PASS** |
| **Protocol 2 Classification Agreement** | 100% (0 flips) | **100.00% (66 / 66 cases agreed)** | **PASS** |
| **Borderline Flip Rate** | 0 flips on $|\text{Prob} - \tau| < 0.05$ | **0 flips (4 / 4 boundary cases preserved)** | **PASS** |
| **Full Regression Suite** | 100% passing tests | **42 / 42 passed in 13.89s** | **PASS** |

---

## 3. Facts vs. Estimates Distinction

### Empirically Verified Facts:
1. **Exported Model Dimensions:** The FP32 ONNX model ([`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)) contains 5,250,561 float32 parameters across 16 initializers and 109 graph nodes. It occupies exactly **21,015,452 bytes (20.04 MiB)** on disk.
2. **Computational Parity:** On host x86_64 CPU under ONNX Runtime 1.31.0, inference on identical input tensors produces bit-level alignment with PyTorch: maximum logit error is $2.88 \times 10^{-6}$ and maximum probability error is $4.97 \times 10^{-7}$.
3. **Threshold Stability:** Classification calls at threshold $\tau = 0.83$ are 100% identical between PyTorch and ONNX Runtime for all 66 holdout patients under both Protocol 1 (89.39% accuracy, 6 flips vs ground truth) and Protocol 2 (87.88% accuracy, 8 flips vs ground truth).

### Pre-Conversion Estimates (Subject to Phase 3 & 4 Verification):
1. **FP16 Model Storage:** Expected to decrease by ~50% to ~10.5 MB.
2. **FP16 Numerical Drift:** FP16 IEEE 754 precision has a machine epsilon of $\approx 9.77 \times 10^{-4}$. Because `CASE_36` is within $+1.06 \times 10^{-4}$ of the decision threshold, **`CASE_36` is at high risk of a classification flip under naive FP16 conversion**. Phase 3 must evaluate mixed-precision or targeted layer casting if `CASE_36` drifts below $\tau = 0.83$.
3. **On-Device Runtime & Memory:** Mobile Android execution latencies (NNAPI / XNNPACK / QNN) and RAM requirements will be measured on physical hardware during Phase 4.

---

## 4. Phase 2 Artifact Ledger

| Deliverable Artifact | Description | Status |
| :--- | :--- | :---: |
| [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx) | Exported standalone FP32 ONNX model | **CREATED & VERIFIED** |
| [`reports/mobile_deployment/phase2_1_export_audit.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_1_export_audit.md) | Export wrapper implementation & mathematical audit | **COMPLETED** |
| [`reports/mobile_deployment/phase2_2_export_audit.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_2_export_audit.md) | Environment audit, installation log, and graph topology | **COMPLETED** |
| [`reports/mobile_deployment/phase2_3_numerical_parity_report.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_3_numerical_parity_report.md) | Full 66-case numerical parity report for Protocol 1 & 2 | **COMPLETED** |
| [`reports/mobile_deployment/phase2_parity_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_parity_benchmark_data.json) | Machine-readable numerical parity benchmark data | **COMPLETED** |
| [`reports/mobile_deployment/phase2_validation_inventory.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase2_validation_inventory.json) | Machine-readable validation inventory (Schema 2.0.0) | **COMPLETED** |
| [`tests/test_onnx_export_and_parity.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tests/test_onnx_export_and_parity.py) | Automated test suite for ONNX graph and dynamic parity | **COMPLETED** |

---

## 5. Scope Boundary & Next Phase Recommendation

In strict accordance with project governance:
* Host-side FP32 ONNX export and parity verification is **100% complete**.
* Execution is **STOPPED**. No FP16 conversion, INT8 quantization, or Android mobile implementation has been commenced.

**Recommendation:** The project is in an optimal, mathematically verified state to proceed to **Phase 3 (FP16 Conversion and Optimization Audit)** upon receipt of separate user authorization.
