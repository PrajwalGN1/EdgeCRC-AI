# Precision Optimization Readiness: FP16 and INT8

**Audit Date:** 2026-10-12  
**Target:** Multimodal Fusion Network & Upstream Encoders  
**Status:** **READY FOR FUSION FP16/INT8 EXPERIMENTS • BLOCKED FOR ENCODERS UNTIL WEIGHTS OBTAINED**  

---

## 1. FP16 (Half-Precision) Readiness Assessment

### A. Fusion Model (`SummaryToPatchCrossAttention`)
* **Technical Feasibility:** **100% Ready Immediately**.
* **Model Footprint:** Drops from **21.0 MB** down to **~10.5 MB**.
* **Architecture Compatibility:**
  * Linear projection (`768 → 1024`), Multihead Attention (`nn.MultiheadAttention`), LayerNorm, and Classifier MLP all support IEEE FP16.
  * Inputs can be cast seamlessly via `vision.half()` and `text.half()`.
* **Execution Considerations:**
  * On modern GPUs and mobile NPUs, FP16 runs natively on Tensor Cores with ~2× higher arithmetic throughput.
  * On the current CPU-only development environment (`torch 2.9.1+cpu`), PyTorch may emulate FP16 in software. However, numerical divergence testing can be executed immediately.
* **Validation Protocol:**
  * Run inference across the 66 validation patients in both FP32 and FP16.
  * Compute maximum probability divergence:
    $$\Delta_{\max} = \max_{i} \left| p_{\text{FP32}}^{(i)} - p_{\text{FP16}}^{(i)} \right|$$
  * Confirm that ROC-AUC and classification decisions at $\tau = 0.83$ remain unchanged ($\Delta \text{AUC} < 0.001$).

### B. Foundation Encoders (Phikon-v2 & ClinicalBERT)
* **Status:** **BLOCKED** until full weights are downloaded or provided locally.
* **Expected Impact Once Available:**
  * Combined weights decrease from **1.57 GB** to **~785 MB**, making them feasible for memory-constrained edge hardware.

---

## 2. INT8 (Integer Quantization) Readiness Assessment

### A. Quantization Approaches for the Fusion Model
1. **Dynamic Quantization (`torch.ao.quantization.quantize_dynamic`):**
   * Can be applied to `nn.Linear` layers immediately with zero calibration requirements.
   * Model size drops from **21.0 MB** to **~5.5 MB**.
2. **Post-Training Static Quantization (PTQ) via ONNX Runtime / TensorRT:**
   * Quantizes activations and weights into INT8 using KL-divergence calibration.
   * Calibration dataset: The 66 validation patient tensors serve as an exact, representative calibration set.
3. **Operator Constraints:**
   * Attention softmax (`softmax(QK^T / √d)`) and LayerNorm should remain in FP16/FP32 to avoid numerical overflow and precision loss.

---

## 3. Recommended Phased Implementation Sequence

```
Phase 1 (Immediate - No Downloads Required):
  ├── Evaluate fusion_model_v1_1.pth in FP16 on 66 validation patients.
  ├── Verify accuracy/AUC parity with FP32 baseline.
  └── Apply dynamic INT8 quantization to fusion model and measure latency/memory.

Phase 2 (Requires Foundation Weights):
  ├── Acquire full Phikon-v2 and ClinicalBERT weights.
  ├── Benchmark FP16 encoder inference.
  └── Export complete pipeline to ONNX with mixed precision (INT8 Linear + FP16 Attention).
```
