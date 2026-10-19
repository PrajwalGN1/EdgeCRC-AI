# Phase 1.5.1 — Patch-Selection Reproducibility & Input Policy Audit

> **Document ID:** `AUDIT-PHASE1.5.1-SAMPLING-REPRODUCIBILITY`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Audit Status:** **VERIFIED & SECURED (MANIFEST-CONTROLLED)**  
> **Target Manifest:** [`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json)  

---

## 1. Executive Summary

This audit establishes a mathematically verifiable and execution-order-independent patch-selection policy for the holdout evaluation cohort ($N=66$). 

Prior evaluations relied on setting `torch.manual_seed(42)` once prior to looping across holdout cases. We proved that this sequential global PRNG approach is **fundamentally order-dependent**: evaluating a patient in isolation (as occurs on mobile edge devices) yields a completely different patch subset than evaluating the same patient as the $i$-th entry in a batch loop (e.g., only 24.4% overlap for `CASE_20`).

To eliminate this vulnerability, the project has established an immutable **Patch-Selection Manifest** ([`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json)), freezing the exact patch indices and SHA-256 tensor fingerprints for all 66 cases across both evaluation protocols.

---

## 2. Codebase Investigation of Stochastic Patch Operations

A forensic review of the source code identified every random operation affecting sequence length and patch subsampling:

| Component | Source File & Lines | Configured Behavior | Random Operation | Reproducibility Defect Identified |
| :--- | :--- | :---: | :--- | :--- |
| **Dataset Loader** | [`src/multimodal_dataset.py:L76-L80`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/multimodal_dataset.py#L76-L80) | `max_patches=1000` | `torch.randperm(num_patches)[:self.max_patches]` | Invokes PyTorch default global generator; unanchored per item. |
| **Training Pipeline** | [`src/cross_attentiion_train.py:L61-L70`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attentiion_train.py#L61-L70) | `max_patches=1000` | `random_split`, `DataLoader(shuffle=True)` | `set_seed(42)` set once at startup; global PRNG state advances across epochs and batches. |
| **Legacy Evaluation** | [`src/cross_attention_eval.py:L73-L98`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py#L73-L98) | `max_patches=1000` | `torch.randperm` inside dataset `__getitem__` | **Zero seeds set** in script. Relies entirely on OS entropy, explaining historical score jitter (87.88% vs 89.39%). |
| **Phase 1 Benchmark** | [`src/eval/benchmark_fp32_reference.py:L269-L277`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_fp32_reference.py#L269-L277) | `max_patches=1000` | Sequential `torch.manual_seed(42)` before loop | Order-dependent: PRNG state advances sequentially across cases. |

---

## 3. Order-Dependence Proof: Sequential vs. Isolated Execution

To demonstrate why `torch.manual_seed(42)` before a loop fails in edge deployment, we simulated evaluating `CASE_20` ($N=3,914$ patches, Ground Truth: MSS / Class 0):

* **Condition A (Sequential Batch):** `torch.manual_seed(42)` is set once. Case 19 ($N=1,690$) is evaluated first (consuming 1,690 pseudo-random values). `CASE_20` is evaluated second.
* **Condition B (Isolated Mobile Inference):** `torch.manual_seed(42)` is set, and `CASE_20` is evaluated immediately in isolation.

```
┌────────────────────────────────────────────────────────────────────────┐
│ EMPIRICAL COMPARISON: CASE_20 PATCH SUBSETS                            │
│   • Total Available Patches: 3,914                                     │
│   • Subsample Size: 1,000                                              │
│   • Overlapping Selected Indices: 244 / 1,000 (24.4% agreement)        │
│   • Are Selected Subsets Equal?: FALSE (torch.equal() == False)        │
└────────────────────────────────────────────────────────────────────────┘
```

**Conclusion:** In an edge deployment where mobile inference processes single cases on-demand, relying on a loop-level seed produces input drift and decision instability.

---

## 4. Established Reproducible Evaluation Input Policy

To achieve bit-exact reproducibility regardless of processing order, runtime architecture, or execution framework:

1. **Explicit Patch-Selection Manifest:**  
   The project has generated [`protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json) containing all 66 holdout records.
2. **Deterministic Tensor Fingerprinting:**  
   For every record, the exact float32 binary content of the input tensors is fingerprinted using SHA-256:
   - `protocol1_vision_tensor_fingerprint`: Hash of selected vision tensor (`[1, min(N, 1000), 1024]`).
   - `protocol2_vision_tensor_fingerprint`: Hash of complete uncapped vision tensor (`[1, N, 1024]`).
   - `text_tensor_fingerprint`: Hash of clinical text summary vector (`[1, 768]`).
3. **Audit of Capped Holdout Cases:**  
   Exactly **8 cases** in the holdout cohort have $N > 1000$:

| De-identified ID | Original Patches ($N$) | Ground Truth | Selected Capped Patches | Fingerprint Status |
| :---: | :---: | :---: | :---: | :---: |
| `CASE_19` | 1,690 | MSI (1) | Exactly 1,000 unique indices | Verified |
| `CASE_20` | 3,914 | MSS (0) | Exactly 1,000 unique indices | Verified |
| `CASE_26` | 1,403 | MSS (0) | Exactly 1,000 unique indices | Verified |
| `CASE_43` | 1,433 | MSS (0) | Exactly 1,000 unique indices | Verified |
| `CASE_49` | 1,921 | MSI (1) | Exactly 1,000 unique indices | Verified |
| `CASE_51` | 2,504 | MSS (0) | Exactly 1,000 unique indices | Verified |
| `CASE_65` | 1,413 | MSS (0) | Exactly 1,000 unique indices | Verified |
| `CASE_66` | 2,090 | MSI (1) | Exactly 1,000 unique indices | Verified |

*(The remaining 58 holdout cases have $N \le 1000$ and evaluate 100% of available patches without capping).*

---

## 5. Empirical Verification of Policy Integrity

We executed automated verification tests in [`tests/test_sampling_manifest.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tests/test_sampling_manifest.py):

* **Reverse-Order Test:** Loading the 66 holdout tensors in reverse order ($66 \to 1$) and evaluating through the manifest produced **0 fingerprint mismatches** across all 66 cases.
* **Isolated-Load Test:** Loading individual cases independently produced **0 fingerprint mismatches**.
* **Index Boundary Checks:** All 8,000 selected indices across the 8 capped cases are strictly unique ($1000$ distinct integers per case) and bounded by $[0, N_i - 1]$.

---

## 6. Dual-Protocol Governance

The project maintains two distinct protocols under this policy:

* **Protocol 1 (Training-Aligned Baseline):** Capped at 1,000 patches using the manifest indices. Represents the configuration under which model weights were trained.
* **Protocol 2 (Candidate Deployment Mode):** Evaluates all $N$ patches dynamically without dropping tissue tiles.

**Important Provenance Caveat:**  
The frozen manifest indices for Protocol 1 anchor the benchmark to the Phase 1 reference database. We do **not** claim that this deterministic subset reproduces the exact pseudo-random subset selected during historical training runs, as historical per-batch PRNG states were not persisted during training. The manifest serves as the immutable golden contract for all forward conversions (PyTorch $\to$ ONNX $\to$ Mobile).
