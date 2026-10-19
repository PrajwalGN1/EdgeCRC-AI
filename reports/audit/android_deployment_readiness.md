# Android Edge Deployment Readiness Assessment

**Audit Date:** 2026-10-12  
**Target Platform:** Android Mobile Device / Edge NPU  
**Status:** **MODE 1 FEASIBLE IMMEDIATELY • MODE 2 REQUIRES HIGH-END NPU HARDWARE**  

---

## 1. Two Distinct Deployment Paradigms

A critical finding of this audit is that "edge deployment" means two vastly different architectures:

```
MODE 1: Feature Cache / Embedding Pre-Screening
┌─────────────────────────────────┐      ┌───────────────────────────────┐
│ Clean Tensors (*.pt)            │ ───► │ SummaryToPatchCrossAttention  │ ───► [MSI / MSS]
│ [N, 1024] Vision + [1, 768] Text│      │ (21 MB FP32 / 10 MB FP16)     │      (<10 ms)
└─────────────────────────────────┘      └───────────────────────────────┘
  • Feasibility: 100% Ready.
  • Compatible with virtually ANY modern Android smartphone (Snapdragon 600/700/800 series).

MODE 2: Full End-to-End Raw Ingestion Pipeline
┌─────────────────────────┐      ┌─────────────────────────┐
│ Raw WSI Slide           │ ───► │ Phikon-v2 ViT-L/16      │ (123 TFLOPs for 1000 patches)
│ (500–1000 H&E Patches)  │      │ (1.13 GB FP32 / 578 MB) │
├─────────────────────────┤      ├─────────────────────────┤
│ Clinical Report Text    │ ───► │ ClinicalBERT            │ (0.09 TFLOPs)
│ (Unstructured Note)     │      │ (413 MB FP32 / 206 MB)  │
└─────────────────────────┘      └─────────────────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │ Cross-Attention Fusion  │ ───► [MSI / MSS]
                                 └─────────────────────────┘
  • Feasibility: Requires Flagship Edge NPU (Snapdragon 8 Gen 2/3, MediaTek Dimensity 9300).
```

---

## 2. Hardware & Runtime Feasibility Matrix

| Hardware / Runtime Parameter | Mode 1 (Fusion Only) | Mode 2 (Full Raw Pipeline) |
| :--- | :--- | :--- |
| **Model Weight Storage** | **~10.5 MB** (FP16) / **~5.5 MB** (INT8) | **~795 MB** (FP16) / **~450 MB** (INT8) |
| **Peak Activation Memory** | $<50\text{ MB}$ RAM | **$1.5 - 2.5\text{ GB}$ RAM** |
| **Compute Workload** | **4.2 GFLOPs** | **$61.5 - 123.1\text{ TFLOPs}$** |
| **Target Runtime** | ONNX Runtime Mobile / ExecuTorch | Qualcomm QNN / MediaTek NeuroPilot / ONNX Runtime |
| **Expected Mobile Latency**| **$<10\text{ ms}$** | **$10 - 25\text{ seconds}$ (Flagship NPU)** |
| **Thermal / Battery Risk** | Negligible | Moderate (sustained heavy matrix multiplication) |
| **Minimum Required Device**| Standard Android phone (4GB RAM) | Flagship Android phone ($\ge 12\text{ GB}$ RAM, $\ge 45\text{ TOPS}$ NPU) |

---

## 3. Recommended 8-Stage Implementation Roadmap

1. **Stage 1 — Cached Tensor Mobile Benchmark:** Export the fusion model to ONNX, run on mobile CPU/GPU via ONNX Runtime Mobile.
2. **Stage 2 — Fusion Precision Optimization:** Benchmark FP16 and INT8 quantized fusion models on mobile hardware.
3. **Stage 3 — Foundation Encoder Acquisition:** Obtain complete weights for Phikon-v2 and ClinicalBERT.
4. **Stage 4 — Raw Tile & Text Preprocessing Module:** Port Reinhard stain normalization, Otsu filtering, and WordPiece tokenization to mobile-compatible C++ / Java / Python.
5. **Stage 5 — End-to-End FP32 Reference:** Benchmark full end-to-end inference on workstation GPU as golden truth.
6. **Stage 6 — Encoder Quantization:** Export Phikon-v2 and ClinicalBERT to FP16 and INT8 using ONNX Runtime / Qualcomm AI Hub.
7. **Stage 7 — Mobile Operator Verification:** Verify NPU operator coverage (especially multihead attention softmax and dynamic sequence lengths).
8. **Stage 8 — On-Device Validation:** Measure physical latency, thermals, battery draw, and numerical accuracy on real Android hardware.
