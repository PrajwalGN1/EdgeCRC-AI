# FP32 Computational Complexity Benchmark Audit

**Audit Date:** 2026-10-12  
**Target Script:** [`src/eval/benchmark_computational_power.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_computational_power.py)  
**Output Reports:** [`results/computational_power_fp32.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/computational_power_fp32.json), [`results/computational_power_fp32.txt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/computational_power_fp32.txt)  

---

## 1. Profiling Methodology & Implementation Review

The benchmark implementation was audited for measurement rigor and correctness:

* **Profiling Engine:** Utilizes PyTorch's native profiler (`torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True)`).
* **Warmup Protocol:** 2 complete warmup iterations executed prior to latency recording (`time.perf_counter()`), preventing cold-start JIT/memory allocation overhead.
* **Repetition Count:** 3 to 5 forward passes averaged per model.
* **Precision Format:** FP32 single-precision (`torch.float32`).
* **Hardware Context:** Executed on the local host CPU (`torch.device('cpu')`).
* **MACs vs. FLOPs Definition:** Adheres to standard hardware arithmetic convention:
  $$1\ \text{MAC (Multiply-Accumulate)} = 2\ \text{FLOPs}$$

---

## 2. Reconciled Metrics Summary

| Model Component | Exact Parameters | FP32 Weight Memory | Compute per Forward Pass | Measured Host CPU Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Phikon-v2** (`owkin/phikon-v2`) | **303,351,808** | **1,157.20 MB** (1.13 GB) | **123.13 GFLOPs** (61.56 GMACs) / patch | 1,644.0 ms / patch |
| **ClinicalBERT** (`Bio_ClinicalBERT`) | **108,310,272** | **413.17 MB** (0.40 GB) | • Seq 128: **22.35 GFLOPs**<br>• Seq 512: **96.69 GFLOPs** | • Seq 128: 318.0 ms<br>• Seq 512: 1,197.9 ms |
| **Cross-Attention Fusion Head** | **5,250,561** | **20.03 MB** | **4.20 GFLOPs** (1,000 patches) | 44.75 ms (1,000 patches) |

---

## 3. Discrepancy & Attribution Analysis

1. **Parameter Rounding in Documentation:**
   * Documentation references approximate figures (e.g., "~307M" for Phikon-v2, "~110M" for ClinicalBERT).
   * Exact counted parameters are **303,351,808** and **108,310,272**, respectively.
2. **Physical Measurement vs. Hardware Projection:**
   * The benchmark's timing metrics (1.64s per patch for Phikon-v2, 44.8ms for the fusion head) represent **single-threaded unaccelerated CPU execution** on the local developer laptop.
   * Projected slide-level throughput figures (e.g., `<0.6s` on an NVIDIA A100 GPU or `8-12s` on an NVIDIA Jetson Orin) are **analytical extrapolations** derived from theoretical TFLOP requirements ($\approx 123.13\text{ TFLOPs}$ per 1,000-patch slide). They were not directly measured on physical edge hardware.
