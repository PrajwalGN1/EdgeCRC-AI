# Phase 1 — FP32 Reference Benchmark Report (Revised)
## Multimodal Fusion for CRC MSI/MSS Detection: Baseline Reference & Protocol Provenance

> **Document ID:** `PHASE1-BENCHMARK-20261016-REV2`  
> **Repository Root:** [`d:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2)  
> **Active Git Branch:** `edge-deployment` (base commit: `189b6638c31d80344ecc8c45133c7058c3e47fa1`)  
> **Execution Environment:** Windows 11 (AMD64) • Python 3.13.5 • PyTorch 2.9.1+cpu • 4 CPU Threads  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth) (21,007,347 bytes, SHA-256 `e9a7bbcd...`)  
> **Target Architecture:** [`SummaryToPatchCrossAttention`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/cross_attention_fusion.py#L4-L58) (5,250,561 float32 parameters)  
> **Holdout Cohort:** [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json) (66 isolated validation patients: 50 MSS, 16 MSI)  
> **Decision Threshold:** Hardcoded operating boundary $\tau = 0.83$ (Class 1: MSI if $P \ge 0.83$, Class 0: MSS if $P < 0.83$)  
> **Machine-Readable Database:** [`reports/mobile_deployment/phase1_fp32_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase1_fp32_benchmark_data.json)  
> **Status:** **PHASE 1 COMPLETE — FP32 BASELINE ESTABLISHED & FROZEN UNDER DUAL-PROTOCOL SPECIFICATION**

---

## 1. Executive Summary & Resolution of Pre-Phase 2 Issues

Following senior engineering review, three critical methodological issues were investigated, documented, and formally resolved in this revised baseline report:

1. **Resolution of Protocol A vs. Training Pipeline Provenance (Issue A):**  
   - Forensic inspection of [`src/cross_attentiion_train.py:64`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attentiion_train.py#L64) confirmed that the checkpoint `fusion_model_v1_1.pth` was **strictly trained with sequences capped at 1,000 patches** (`max_patches=1000`).
   - The original holdout evaluation script [`src/cross_attention_eval.py:73`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py#L73) likewise instantiated `MultimodalTCGADataset` with `max_patches=1000`.
   - **Decision:** The original capped protocol is formally frozen as **Protocol 1 (Training-Aligned Baseline, Capped at 1,000 Patches with Seed 42)** to serve as the true historical reference. The uncapped protocol is frozen as **Protocol 2 (Candidate Deployment Mode, Full Sequences Preserved)**. Neither protocol replaces the other; both are fully benchmarked, with per-patient logits, probabilities, and exact input tensors frozen in [`reports/mobile_deployment/phase1_fp32_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase1_fp32_benchmark_data.json).
2. **Refining Floating-Point Determinism to Numerical Tolerances (Issue B):**  
   - The preliminary claim of "bit-exact determinism across mobile runtimes" has been corrected. Removing stochastic sampling eliminates algorithm-level non-determinism, but **floating-point arithmetic is non-associative**. Operator implementations, SIMD vectorization (AVX2 vs. ARM NEON), matrix tiling, and compiler optimizations cause minor numerical divergence between PyTorch CPU, ONNX Runtime, and mobile accelerators.
   - **Remedy:** We establish explicit numerical error bounds ($|\Delta \text{logit}| \le 10^{-4}$, $|\Delta P| \le 10^{-4}$) and establish a dedicated **Boundary Sensitivity Zone Audit** for cases within $[0.83 \pm 0.05]$ to track classification flip risks.
3. **Contextualizing Host Development vs. Mobile Edge Resource Profiles (Issue C):**  
   - The host latency ($139.80\text{ ms}$) and peak memory ($665.33\text{ MB}$) were measured on a development x86_64 host using **batch preloading of all 66 holdout tensor pairs**.
   - **Remedy:** We explicitly dissociate host development metrics from on-device Android projections. Edge deployment executes **strictly single-patient sequential inference** (typical activation footprint $<25\text{ MB}$, input tensor size $2 - 4\text{ MB}$). On-device latency and thermal behavior will be measured directly on physical ARM64 hardware in subsequent phases.
4. **Clinical Scope Caveat:**  
   - The 66-case holdout cohort from TCGA-COAD is an isolated research partition sufficient for technical conversion validation and regression detection. It is **not sufficiently powered by itself to assert generalized clinical readiness**.

---

## 2. Working-Tree & Asset Integrity Confirmation

* **Current Branch:** `edge-deployment`
* **Head Commit:** `189b6638c31d80344ecc8c45133c7058c3e47fa1`
* **Original Assets Verified Unchanged:**
  - Checkpoint [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth) (SHA-256: `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453`) is intact and unmodified.
  - All 652 precomputed tensors in [`data/clean_tensors/`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/clean_tensors) are intact and unmodified.
  - Master registry [`data/patient_registry.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/patient_registry.json) (326 records) is intact and unmodified.
  - Holdout manifest [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json) (66 keys) is intact and unmodified.
  - Decision threshold $\tau = 0.83$ is preserved.
  - Pre-existing user modifications in [`src/ui/app.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/ui/app.py) and historical image plots in `results/` remain untouched.

---

## 3. Protocol Provenance & Discrepancy Resolution

### 3.1 Training and Original Evaluation Verification

Forensic inspection of codebase configurations confirms the exact history of sequence length handling:

| Pipeline Stage | Source Code Location | Configured Patch Cap | Sampling Mechanism | Behavior on Slides with $N > 1000$ |
| :--- | :--- | :---: | :--- | :--- |
| **Model Training** | [`src/cross_attentiion_train.py:64`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attentiion_train.py#L64) | `max_patches=1000` | `torch.randperm(N)[:1000]` | **Subsampled to 1,000 patches** on every epoch |
| **Original Holdout Evaluation** | [`src/cross_attention_eval.py:73`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py#L73) | `max_patches=1000` | `torch.randperm(N)[:1000]` | **Subsampled to 1,000 patches** (unseeded PRNG) |
| **Raw Preprocessing Engine** | [`src/inference/image_preprocessing.py:176`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/inference/image_preprocessing.py#L176) | `max_patches=1000` | Deterministic `np.random.choice(..., seed=42)` | Subsampled to 1,000 patches |

### 3.2 Root Cause of Historical Metric Variations

1. In the 66-patient holdout cohort, exactly **8 patients** have $N > 1000$ patches (up to 3,914 patches).
2. The legacy evaluation script [`src/cross_attention_eval.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py) ran `torch.randperm` without setting `torch.manual_seed()` inside `evaluate_model()`.
3. In a 10-trial Monte Carlo simulation, exactly **1 patient** crossed the operating decision threshold $\tau = 0.83$:
   - **Borderline Patient:** `TCGA-AY-A71X` (Ground Truth: MSS / Class 0, $N = 3,914$ patches).
   - Under unseeded random subsampling of 1,000 patches, its probability fluctuates between **$0.6556$ and $0.9167$**:
     - Subsampled runs where $P \ge 0.83$ produce a False Positive MSI call $\implies \mathbf{58 / 66\text{ correct} \implies 87.88\%\text{ Accuracy}}$ (matching the committed `results/Cross_Attention_Model_Evaluation_Report.txt` and ROC-AUC of 0.9250).
     - Subsampled runs where $P < 0.83$ produce a True Negative MSS call $\implies \mathbf{59 / 66\text{ correct} \implies 89.39\%\text{ Accuracy}}$ (ROC-AUC 0.9263).

### 3.3 Protocol Definitions & Roles

To ensure methodological rigor, the project defines and freezes two explicit benchmark protocols:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PROTOCOL 1: Training-Aligned Baseline (Historical Reference)           │
│   • max_patches = 1000 (enforced via torch.manual_seed(42))            │
│   • Exact operational match to cross_attentiion_train.py               │
│   • Metrics: Accuracy 89.39%, ROC-AUC 0.9262, MSI F1 0.7742            │
│   • Role: True reference for checking model equivalence                │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ PROTOCOL 2: Full Uncapped Candidate (Exploratory Deployment Mode)      │
│   • Full sequence evaluated without capping (N in [10, 3914])          │
│   • Evaluates complete WSI slide evidence without tile dropping        │
│   • Metrics: Accuracy 87.88%, ROC-AUC 0.9338, MSI F1 0.7500            │
│   • Role: Candidate edge deployment profile under technical evaluation │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Performance Metrics Comparison

Both protocols were executed using [`src/eval/benchmark_fp32_reference.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_fp32_reference.py) on the host CPU:

| Evaluation Metric | Protocol 1 (Training-Aligned Capped `seed=42`) | Protocol 2 (Full Uncapped Candidate) | Published Academic Report (`results/...Report.txt`) |
| :--- | :---: | :---: | :---: |
| **Patch Sequence Rule** | Capped at 1,000 patches | Full Sequence ($10 \le N \le 3914$) | Capped at 1,000 patches (Unseeded) |
| **Overall Accuracy** | **89.39%** (59 / 66) | **87.88%** (58 / 66) | **87.88%** (58 / 66) |
| **Balanced Accuracy** | **84.50%** | **83.50%** | **83.50%** |
| **ROC-AUC Score** | **0.9262** | **0.9338** | **0.9250** |
| **PR-AUC Score** | **0.8415** | **0.8368** | — |
| **Confusion Matrix `[[TN, FP], [FN, TP]]`** | `[[47, 3], [4, 12]]` | `[[46, 4], [4, 12]]` | `[[46, 4], [4, 12]]` |
| **MSS (Class 0) Precision** | **0.9216** (47 / 51) | **0.9200** (46 / 50) | **0.92** |
| **MSS (Class 0) Recall / Specificity** | **0.9400** (47 / 50) | **0.9200** (46 / 50) | **0.92** |
| **MSS (Class 0) F1-Score** | **0.9307** | **0.9200** | **0.92** |
| **MSI (Class 1) Precision** | **0.8000** (12 / 15) | **0.7500** (12 / 16) | **0.75** |
| **MSI (Class 1) Recall / Sensitivity** | **0.7500** (12 / 16) | **0.7500** (12 / 16) | **0.75** |
| **MSI (Class 1) F1-Score** | **0.7742** | **0.7500** | **0.75** |
| **Macro Average F1** | **0.8524** | **0.8350** | **0.83** |
| **Weighted Average F1** | **0.8928** | **0.8788** | **0.88** |

> [!IMPORTANT]
> The single discrepancy between Protocol 1 and Protocol 2 is case `TCGA-AY-A71X` ($N=3,914$, True MSS):
> - Under Protocol 1 (seed=42): $P = 0.7632 < 0.83 \implies$ **True Negative MSS** (Accuracy: 59/66 = 89.39%).
> - Under Protocol 2 (all 3,914 patches): $P = 0.8407 \ge 0.83 \implies$ **False Positive MSI** (Accuracy: 58/66 = 87.88%).

---

## 5. Numerical Tolerances & Boundary Sensitivity Zone

### 5.1 Floating-Point Non-Associativity & Runtime Tolerances

Different execution engines (PyTorch ATen, ONNX Runtime CPU, ONNX Runtime Mobile NNAPI/XNNPACK) accumulate floating-point matrix multiplications in different block tiles and SIMD orders. Because floating-point addition is non-associative, minor numerical variations ($\Delta \approx 10^{-6} \text{ to } 10^{-4}$) are expected.

For Phase 2 ONNX conversion and subsequent mobile validation, we define the following numerical tolerance contract:

| Metric | Target Tolerance Threshold | Verification Criterion |
| :--- | :---: | :--- |
| **Max Absolute Logit Error ($L_\infty$)** | $|\text{logit}_{\text{ref}} - \text{logit}_{\text{runtime}}| \le \mathbf{10^{-4}}$ | All 66 holdout patients |
| **Max Absolute Probability Error ($L_\infty$)** | $|P_{\text{ref}} - P_{\text{runtime}}| \le \mathbf{10^{-4}}$ | All 66 holdout patients |
| **Mean Absolute Probability Error (MAE)** | $\frac{1}{66} \sum |P_{\text{ref}} - P_{\text{runtime}}| \le \mathbf{10^{-5}}$ | Cohort aggregate |
| **Non-Borderline Classification Parity** | **100.0% Agreement** | All cases outside the boundary sensitivity zone |

### 5.2 Boundary Sensitivity Zone Audit

Predictions falling within $[0.83 \pm 0.05]$ (i.e. $P \in [0.78, 0.88]$) are sensitive to small numerical perturbations that could flip the diagnostic call. In the 66-patient holdout set, exactly **4 patients** fall within this window under Protocol 2:

| Patient Key | Patches ($N$) | Ground Truth | Uncapped Prob ($P$) | Distance to $\tau=0.83$ | Reference Call | Risk Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`TCGA-D5-6928`** | 80 | **MSI (1)** | **0.830106** | **+0.000106** | **MSI (TP)** | **CRITICAL:** Prediction is only $+0.0001$ above threshold. A negative drift of $0.0002$ will cause a False Negative flip. |
| **`TCGA-AY-A71X`** | 3,914 | **MSS (0)** | **0.840750** | **+0.010750** | **MSI (FP)** | **HIGH:** Sensitivity to sequence length and accumulation order; flips to TN in Protocol 1. |
| **`TCGA-D5-6534`** | 166 | **MSS (0)** | **0.817998** | **-0.012002** | **MSS (TN)** | **MODERATE:** Prediction is $-0.0120$ below threshold. A positive drift of $0.013$ would cause a False Positive flip. |
| **`TCGA-AD-5900`** | 361 | **MSI (1)** | **0.860587** | **+0.030587** | **MSI (TP)** | **LOW:** Well-separated ($+0.0306$), but monitored during FP16 conversion. |

*Phase 2 Action:* During ONNX graph validation, these 4 patients will undergo dedicated floating-point drift auditing to verify that conversion does not trigger unintended diagnostic classification flips.

---

## 6. Resource Profiling: Host Development vs. Mobile Edge Expectations

### 6.1 Host Development Benchmark (Measured)

* **Platform:** Intel Core (4 threads) • Windows 11 AMD64 • PyTorch 2.9.1+cpu
* **Latency Profile:**
  - Mean Slide Latency: **$139.80\text{ ms}$** (across all 66 slides, mean $N = 491.77$)
  - Median Slide Latency: **$89.81\text{ ms}$** (50th percentile)
  - Latency Range: **$30.01\text{ ms}$** ($N=35$) to **$820.50\text{ ms}$** ($N=3,914$)
* **Memory Profile:**
  - Model Weight Footprint: **$20.03\text{ MB}$**
  - Baseline Process Memory: **$355.40\text{ MB}$** RSS
  - Peak Process Memory: **$665.33\text{ MB}$** RSS

> [!CAUTION]
> **Host Memory Context:** The measured peak memory of $665.33\text{ MB}$ reflects holding **all 66 patient tensor pairs in host RAM simultaneously** to accelerate multi-trial Monte Carlo evaluations. It must **NOT** be interpreted as the memory required to execute one patient on Android.

### 6.2 Mobile Edge Target Expectations (Architectural Specification)

On Android devices, the application operates strictly in **single-patient sequential inference mode**:

| Component | Target Budget (FP32) | Target Budget (FP16 Target) | Notes |
| :--- | :--- | :--- | :--- |
| **Model Weight Storage** | **20.03 MB** | **~10.02 MB** | Embedded in Android APK `assets/` |
| **Precomputed Input: Text `[1, 768]`** | $\approx 3\text{ KB}$ | $\approx 1.5\text{ KB}$ | Single pathology summary vector |
| **Precomputed Input: Vision `[1, N, 1024]`** | • $N=500$: $\approx 2.05\text{ MB}$<br>• $N=1000$: $\approx 4.10\text{ MB}$ | • $N=500$: $\approx 1.02\text{ MB}$<br>• $N=1000$: $\approx 2.05\text{ MB}$ | Transmitted or loaded per patient case |
| **Peak Activation Memory** | $<25\text{ MB}$ ($N=1000$) | $<15\text{ MB}$ ($N=1000$) | Multihead attention intermediate matrices |
| **Target Runtime** | ONNX Runtime Mobile (CPU) | ONNX Runtime Mobile (NNAPI / GPU) | Android C++ / Java bindings |
| **Projected Mobile Latency** | $\approx 30 - 60\text{ ms}$ | **$<15\text{ ms}$** | Real physical device profiling in Phase 4 |

---

## 7. Complete Reference Data Inventory

Both protocols have been frozen into [`reports/mobile_deployment/phase1_fp32_benchmark_data.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase1_fp32_benchmark_data.json) (3,526 lines, 95.6 KB).

### Summary of JSON Structure:
```json
{
  "metadata": { ... },
  "protocol_provenance_and_relationship": {
    "training_configuration": { "max_patches": 1000, ... },
    "original_evaluation_configuration": { "max_patches": 1000, ... },
    "protocol_roles": { ... }
  },
  "protocols_summary": {
    "protocol_1_training_aligned_capped_seed42": { "accuracy": 0.8939, "roc_auc": 0.9262, ... },
    "protocol_2_full_uncapped_candidate": { "accuracy": 0.8788, "roc_auc": 0.9338, ... },
    "protocol_3_legacy_unseeded_monte_carlo": { "accuracy_range": [0.8788, 0.8939], ... }
  },
  "numerical_tolerances_and_determinism_guidelines": {
    "target_tolerances_for_phase2": { "max_absolute_logit_diff": 0.0001, "max_absolute_prob_diff": 0.0001 },
    "boundary_sensitivity_zone": { "threshold": 0.83, "borderline_patients": [ ... ] }
  },
  "resource_profiling_framing": {
    "host_development_profile": { ... },
    "mobile_runtime_target_expectation": { ... }
  },
  "clinical_scope_caveat": { ... },
  "per_patient_reference_data_protocol_1_capped_seed42": [ ... 66 patient objects ... ],
  "per_patient_reference_data_protocol_2_uncapped": [ ... 66 patient objects ... ]
}
```

---

## 8. Unit & Integration Test Verification

The full pytest test suite was executed:
```bash
python -m pytest tests/ -v
```
* **Result:** **27 passed, 0 failed** in 71.89 seconds.
* **Coverage:** Input validation (12 tests), Image preprocessing & tiling (4 tests), Encoder wrappers & fail-closed policies (4 tests), Fusion model loading, strict weights, and predictor (7 tests).

---

## 9. Readiness & Next Steps for Phase 2 (ONNX Model Conversion)

With the resolution of all three issues, the project has established a rigorous foundation for Phase 2:

1. **Dual-Protocol Parity:** ONNX export will be verified against **Protocol 1** (to prove mathematical equivalence to the trained model) and against **Protocol 2** (to prove dynamic shape capability on full sequences).
2. **Tolerance Contract:** Output probabilities will be checked against the reference database with an error threshold $\le 10^{-4}$, with dedicated focus on the 4 borderline cases (`TCGA-D5-6928`, `TCGA-AY-A71X`, `TCGA-D5-6534`, `TCGA-AD-5900`).
3. **Execution Guardrails:** Line 48 of `cross_attention_fusion.py` will be handled cleanly in the export wrapper to avoid PyTorch tracer errors.
4. **Tooling Authorization:** Phase 2 will require installing `onnx` and `onnxruntime` to build and test the exported computation graphs.

---

## 10. Sign-Off

Phase 1 baseline preparation is complete. I will stop here and await your review of this revised report and explicit authorization before proceeding to Phase 2.
