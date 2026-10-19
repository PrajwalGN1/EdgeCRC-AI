# Phase 3.4 — Resource, Storage & Performance Comparison Audit

> **Document ID:** `AUDIT-PHASE3.4-RESOURCE-COMPARISON`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **FP32 Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **FP16 Model:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Benchmark Artifact:** [`reports/mobile_deployment/phase3_latency_benchmark.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/phase3_latency_benchmark.json)  
> **Audit Status:** **COMPLETED — EMPIRICAL BENCHMARKS RECORDED**  

---

## 1. Executive Summary

This report establishes the empirical resource comparison between the validated FP32 ONNX model and the uniform FP16 ONNX model.

### Key Measured Outcomes:
1. **On-Disk Model Footprint:** Model storage is reduced from **21,015,452 bytes (20.04 MiB)** to **10,519,654 bytes (10.03 MiB)**, representing a measured **49.94% reduction**.
2. **Host-Side CPU Inference Latency:** On the host x86_64 development machine, FP16 execution under ONNX Runtime CPU is slightly slower than FP32 ($1.17\times$ to $1.75\times$ latency), due to lack of native FP16 arithmetic instructions on standard consumer x86 CPUs (which emulate FP16 via software or internal upcasting).
3. **Fact vs. Projection Governance:** In strict adherence to Phase 3 guidelines, **no claims of Android mobile edge latency, battery, or thermal improvements are asserted as facts**. All mobile hardware performance gains remain explicitly categorized as projected engineering expectations to be empirically validated on physical target hardware during Phase 4.

---

## 2. Model Storage & Memory Footprint Comparison

Both models were inspected on disk using exact byte measurements:

| Metric | FP32 Reference (`fusion_model_fp32.onnx`) | FP16 Candidate (`fusion_model_fp16.onnx`) | Absolute Reduction | Percentage Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Exact File Size (Bytes)** | **21,015,452 bytes** | **10,519,654 bytes** | **10,495,798 bytes** | **49.94%** |
| **Decimal Megabytes (MB)** | 21.02 MB | 10.52 MB | 10.50 MB | 49.94% |
| **Binary Mebibytes (MiB)** | 20.04 MiB | 10.03 MiB | 10.01 MiB | 49.94% |
| **Parameter Precision** | IEEE-754 32-bit Float | IEEE-754 16-bit Float | 16 bits per weight | 50.00% |
| **Learned Weights Count** | 5,250,561 | 5,250,561 | 0 (Identical) | 0.00% |
| **Graph Node Count** | 109 nodes | 109 nodes | 0 (Identical) | 0.00% |

**Clinical Edge Implication:** A ~10 MB reduction in model size represents a substantial advantage for mobile edge deployment, reducing Android application package (APK) download footprint, flash storage overhead, and on-device cold-start loading time.

---

## 3. Host-Side Latency Benchmark Methodology & Results

### 3.1 Benchmark Configuration
* **Host Platform:** Windows 11 Build 10.0.26100 AMD64
* **Runtime:** ONNX Runtime 1.31.0 (`CPUExecutionProvider`)
* **Warm-up Setting:** 10 consecutive warm-up iterations prior to timing
* **Timing Protocol:** 50 timed repetitions using high-resolution wall-clock timer (`time.perf_counter()`)
* **Input Scenarios Evaluated:**
  * $N = 100$: Standard small biopsy patch count
  * $N = 385$: Cohort median patch count
  * $N = 1,000$: Protocol 1 maximum sequence limit

### 3.2 Measured Latency Summary Table

| Sequence Length ($N$) | Model Format | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Std Dev (ms) | Relative Host Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 100$** | **FP32** | **2.70 ms** | 2.53 ms | 3.28 ms | 0.49 ms | Baseline ($1.00\times$) |
| | **FP16** | **4.73 ms** | 4.60 ms | 5.93 ms | 0.44 ms | $0.57\times$ (Host CPU penalty) |
| **$N = 385$** | **FP32** | **8.84 ms** | 8.68 ms | 9.88 ms | 0.63 ms | Baseline ($1.00\times$) |
| | **FP16** | **10.90 ms** | 10.63 ms | 12.81 ms | 0.96 ms | $0.81\times$ (Host CPU penalty) |
| **$N = 1,000$** | **FP32** | **21.56 ms** | 21.05 ms | 24.53 ms | 1.83 ms | Baseline ($1.00\times$) |
| | **FP16** | **25.33 ms** | 24.89 ms | 28.70 ms | 1.95 ms | $0.85\times$ (Host CPU penalty) |

---

## 4. Architectural Analysis of Host-Side vs. Mobile Edge Execution

### 4.1 Why Host x86 CPU Latency is Slower for FP16
The observed host-side latency penalty for FP16 ($25.33\text{ ms}$ vs. $21.56\text{ ms}$ at $N=1,000$) is a well-understood phenomenon in machine learning systems engineering:
1. **Instruction Set Architecture (ISA) Disparity:** Standard consumer x86_64 CPUs (prior to server-grade Intel Sapphire Rapids with AVX-512_FP16) **lack native vector execution units for 16-bit floating point arithmetic**.
2. **Software Emulation / Upcasting Overhead:** When executing an FP16 graph on standard x86 CPU cores, ONNX Runtime's `CPUExecutionProvider` either converts 16-bit operands to 32-bit registers to execute standard AVX2 FMA instructions or relies on software emulation libraries, introducing an additional instruction conversion cycle penalty per operation.

### 4.2 Contrast with Target Mobile Hardware (ARMv8.2-A / NPU)
On mobile edge devices (the target deployment platform for Phase 4):
1. **Native Hardware FP16 Vectors (ARM NEON):** Modern mobile ARM application processors (Cortex-A78, Cortex-X1, Kryo, Cortex-A715) feature native **ARMv8.2-A FP16 vector instructions**, executing twice as many 16-bit operations per clock cycle as 32-bit operations.
2. **Memory Bandwidth Dominance:** Mobile neural network inference is typically memory-bandwidth bound. Reducing weight tensors from 21 MB to 10.5 MB cuts DDR memory bus traffic in half, reducing memory fetch latency and DRAM power dissipation.
3. **NPU / GPU Acceleration (NNAPI / QNN):** Mobile NPUs (Qualcomm Hexagon, MediaTek APU) and mobile GPUs (Adreno, Mali) are natively optimized for FP16 matrix multiplication and cannot run FP32 at full throughput.

---

## 5. Strict Governance: Facts vs. Projections Ledger

| Claim / Metric | Measured Fact or Projected Expectation | Verification Basis |
| :--- | :---: | :--- |
| **FP16 File Size: 10.52 MB** | **VERIFIED FACT** | Measured directly on filesystem (`10,519,654 bytes`). |
| **Storage Reduction: 49.94%** | **VERIFIED FACT** | Computed from actual binary byte counts. |
| **Host CPU Latency: 25.33 ms ($N=1000$)** | **VERIFIED FACT** | 50 timed iterations on host Windows 11 x86 CPU. |
| **Host FP16 Latency is Slower** | **VERIFIED FACT** | Measured empirically ($0.85\times$ relative to FP32). |
| **Mobile Android Speedup** | **PROJECTED EXPECTATION** | Requires empirical benchmarking on physical Android phone (Phase 4). |
| **Mobile RAM / Battery Savings** | **PROJECTED EXPECTATION** | Requires physical device measurement with Android Profiler (Phase 4). |

---

## 6. Resource Audit Conclusion

The uniform FP16 ONNX model ([`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)) achieves the primary storage optimization objective by saving **10.5 MB (49.94%)** with zero parameter loss. The host CPU execution penalty is thoroughly explained and does not impede progression to mobile edge deployment.
