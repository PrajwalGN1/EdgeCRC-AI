# Phase 1.5.4 — Resource Claims & Hardware Profile Audit

> **Document ID:** `AUDIT-PHASE1.5.4-RESOURCE-CLAIMS`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Audit Status:** **AUDITED, RIGOROUSLY SEGREGATED & GROUNDED**  
> **Target Scope:** Clarification of host benchmark measurements vs. edge projections  

---

## 1. Executive Summary

This audit establishes a strict segregation between **empirically measured development-host metrics** and **analytical/theoretical edge projections**. 

Prior reports cited a peak RSS of $665.33\text{ MB}$ and mean latency of $139.80\text{ ms}$. This audit clarifies that the host peak RSS was driven by simultaneous in-memory caching of all 66 patient tensor pairs on a Windows x86_64 CPU workstation. That figure must not be misconstrued as the on-device memory required to evaluate a single patient on Android.

Storage units are explicitly disambiguated between decimal megabytes ($\text{MB} = 10^6\text{ bytes}$) and binary mebibytes ($\text{MiB} = 2^{20} = 1,048,576\text{ bytes}$).

---

## 2. Model Parameters & Storage Footprint (Empirically Verified)

All parameter counts and weight file sizes have been verified via direct inspection of [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth):

| Attribute | Measured Exact Value | Unit (Decimal MB) | Unit (Binary MiB) | Verification Method |
| :--- | :---: | :---: | :---: | :--- |
| **Total Model Parameters** | **5,250,561** float32 | — | — | Sum of `p.numel()` across `SummaryToPatchCrossAttention` |
| **Raw Parameter Buffer Size** | **21,002,244 bytes** | **21.00 MB** | **20.03 MiB** | $5,250,561 \times 4\text{ bytes}$ (exact float32 payload) |
| **PyTorch Checkpoint on Disk** | **21,007,347 bytes** | **21.01 MB** | **20.03 MiB** | `Path.stat().st_size` (includes ZIP/tarball metadata headers) |
| **Projected FP16 Parameters** | $5,250,561 \times 2\text{ bytes}$ | **~10.50 MB** | **~10.01 MiB** | Theoretical calculation ($2\text{ bytes / float16}$) |

### Layer-by-Layer Parameter Breakdown:
1. `text_projection` (Linear $768 \to 1024$): $768 \times 1024 + 1024 = \mathbf{787,456}$ parameters
2. `cross_attn` (MultiheadAttention $1024 \times 1024$, 8 heads):
   - `in_proj_weight` ($3072 \times 1024$) + `in_proj_bias` ($3072$): $3,148,800$ parameters
   - `out_proj.weight` ($1024 \times 1024$) + `out_proj.bias` ($1024$): $1,050,624$ parameters
   - Subtotal Cross-Attention: $\mathbf{4,199,424}$ parameters
3. `layer_norm` (LayerNorm $1024$): $\mathbf{2,048}$ parameters
4. `classifier`:
   - `classifier[0]` (Linear $1024 \to 256$): $262,400$ parameters
   - `classifier[3]` (Linear $256 \to 1$): $257$ parameters
   - Subtotal Classifier: $\mathbf{262,657}$ parameters
5. **Total Architecture Parameters:** $787,456 + 4,199,424 + 2,048 + 262,657 = \mathbf{5,250,561}$ parameters.

---

## 3. Input Tensor Size Profiles (Calculated Estimates)

Under Mode 1 (Feature Cache Edge Deployment), the mobile device receives precomputed embeddings. Tensor buffer sizes are purely calculated estimates based on tensor dimensions:

| Input Modality | Shape | Data Type | Calculated Exact Bytes | Decimal KB / MB | Binary KiB / MiB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clinical Text Summary** | `[1, 768]` | float32 | 3,072 bytes | 3.07 KB | 3.00 KiB |
| **Vision: Minimum Holdout ($N=80$)** | `[1, 80, 1024]` | float32 | 327,680 bytes | 327.68 KB | 320.00 KiB |
| **Vision: Median Holdout ($N=384$)** | `[1, 384, 1024]` | float32 | 1,572,864 bytes | 1.57 MB | 1.50 MiB |
| **Vision: Protocol 1 Cap ($N=1000$)** | `[1, 1000, 1024]` | float32 | 4,096,000 bytes | 4.10 MB | 3.91 MiB |
| **Vision: Maximum Holdout ($N=3914$)** | `[1, 3914, 1024]` | float32 | 16,031,744 bytes | 16.03 MB | 15.29 MiB |

*(For FP16 transmission, all buffer sizes are exactly halved).*

---

## 4. Separation of Measured Host vs. Edge Projections

To maintain audit integrity, we document the operational boundaries between development measurements and edge runtime expectations:

| Resource Dimension | Measured Development Baseline (x86_64 CPU) | Target Edge Mode (Android Mobile) | Status & Source of Figure |
| :--- | :--- | :--- | :--- |
| **Execution Paradigm** | Batch holdout evaluation (all 66 cases in memory) | Single-patient sequential inference on demand | Documented design constraint |
| **Process RSS Memory** | Baseline: **355.40 MB**<br>Peak: **665.33 MB** (holding 66 tensor pairs) | Model: ~20 MB (FP32) / ~10 MB (FP16)<br>Input: 2–4 MB<br>Activations: <25 MB | **Measured on host**; mobile memory is an architectural estimate |
| **Warm Inference Latency** | Mean: **139.80 ms**<br>Median: **89.81 ms**<br>Range: 30.01 ms ($N=35$) to 820.50 ms ($N=3914$) | Projected: 30–60 ms (CPU)<br>Projected: <15 ms (NNAPI/GPU FP16) | **Measured on host**; mobile latency is an unverified projection |
| **Cold End-to-End Latency** | Model load: ~220 ms (disk read + state_dict parsing) | APK asset extraction + ONNX session creation | Host measured; mobile cold-start must be profiled in Phase 4 |
| **Thermal / Battery Profile** | Not applicable (AC-powered host workstation) | Sustained thermal dissipation, battery draw | **Unmeasured**; strictly deferred to physical Android hardware tests |

> [!IMPORTANT]
> **Prohibition Against Extrapolation:**  
> Development machine latency and memory numbers were gathered on an Intel Core x86_64 CPU with Windows multi-threading. They serve as a development reference only. Under no circumstances should host CPU measurements be extrapolated directly to ARM64 Android devices. True mobile latency, memory pressure, and thermal headroom will be measured exclusively on physical target hardware during Phase 4.
