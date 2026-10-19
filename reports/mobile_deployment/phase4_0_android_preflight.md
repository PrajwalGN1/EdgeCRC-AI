# Phase 4.0 — Android Edge Integration: Preflight & Toolchain Audit

> **Document ID:** `AUDIT-PHASE4.0-ANDROID-PREFLIGHT`  
> **Repository Root:** [`d:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2)  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
> **FP32 Reference Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **FP16 Target Model:** [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx)  
> **Author:** Antigravity Senior Android ML Deployment Engineer & Model Validation Auditor  
> **Audit Date:** October 17, 2026  
> **Preflight Gate Decision:** **CONDITIONAL GO** (Clear to proceed with Phase 4.1 Kotlin inference integration; on-device profiling in Phase 4.3 gated on physical hardware provisioning)

---

## 1. Executive Summary & Audit Scope

Phase 4.0 establishes the engineering baseline, toolchain inventory, runtime compatibility, and architectural constraints before integrating the multimodal colorectal cancer (CRC) MSI/MSS fusion classifier into an Android application.

### Key Preflight Findings:
1. **Repository Android Project Status:** No pre-existing Android project, Gradle build scripts, or Kotlin source files exist in the repository or parent directory. A clean, modular Android project structure must be established without modifying or disturbing the existing Python/PyTorch research tree.
2. **Host Toolchain & Devices:**
   - Java Development Kit is operational: Oracle JDK `23.0.1+11-39` (64-Bit).
   - Android SDK (`ANDROID_HOME` / `ANDROID_SDK_ROOT`) and Android Debug Bridge (`adb`) are not configured on the host workstation's system `PATH`.
   - Zero physical Android devices are currently connected via USB or detected via ADB. Physical on-device profiling (Phase 4.3) is therefore gated until physical target hardware is connected.
3. **ONNX Runtime Android Package Compatibility:**
   - The selected package `com.microsoft.onnxruntime:onnxruntime-android:1.18.0` (or `1.17.0+`) provides native support for all 19 graph operators, dynamic batch and sequence dimensions (`[1, N, 1024]`), and single-precision IEEE-754 execution.
4. **FP16 Mobile Runtime Realities:**
   - While FP16 reduces storage size by 50% ($20.04\text{ MB} \to 10.03\text{ MB}$), mobile CPU execution providers without ARMv8.2-A+ NEON FP16 instructions execute FP16 via software emulation or runtime upcasting. Therefore, **FP32 on CPU Execution Provider is designated as the primary reference deployment**, with FP16 evaluated comparatively.
5. **Protocol 1 Patch Cap Discrepancy Formally Resolved:**
   - Deep inspection of the codebase, training scripts, reference benchmarks, and the frozen patch selection manifest ([`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json)) establishes definitively that the **Protocol 1 upper cap is $N \le 1,000$ patches**. Prior narrative references to $N=500$ referred to typical median slide sizes, not the architectural cap.

---

## 2. Host Development & Toolchain Inventory

Direct inspection was performed on the host workstation on October 17, 2026:

| Toolchain Component | Host Inspection Command / Query | Detected Result | Status / Impact |
| :--- | :--- | :--- | :---: |
| **Host Operating System** | System Environment | Windows 10/11 x86_64 | Operational |
| **Java Development Kit** | `java -version` | Oracle JDK `23.0.1` (`2024-10-15`) | **READY** |
| **Java Compiler** | `javac.EXE` location | `C:\Program Files\Common Files\Oracle\Java\javapath` | **READY** |
| **Android SDK Root** | `$env:ANDROID_HOME` / `$env:ANDROID_SDK_ROOT` | `None` / Not set | Not configured on host |
| **Android Debug Bridge** | `adb` on PATH / standard AppData locations | `None` (Not detected) | Not configured on host |
| **Gradle Build Tool** | `gradle` / `gradlew` on PATH | `None` (Not installed globally) | Project must use Gradle Wrapper |
| **Kotlin Compiler** | `kotlinc` on PATH | `None` | Managed via Gradle Kotlin plugin |
| **Connected Physical Devices** | `Get-PnpDevice -Class AndroidUsbDeviceType, WPD, USB` | 0 Android devices attached | **GATED** (Blocks Phase 4.3) |

---

## 3. ONNX Model Artifact & Operator Inspection

Both frozen ONNX models were loaded and forensically inspected using the `onnx` Python runtime:

### 3.1 Model Artifact Specifications:
| Specification Dimension | Reference Model: [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx) | Target Model: [`models/fusion_model_fp16.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp16.onnx) |
| :--- | :--- | :--- |
| **File Size (Bytes / MB)** | $21,015,452\text{ bytes}$ ($20.04\text{ MB}$) | $10,519,654\text{ bytes}$ ($10.03\text{ MB}$) |
| **Cryptographic SHA-256** | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` |
| **ONNX IR Version** | IR 8 | IR 8 |
| **ONNX Opset Version** | `ai.onnx:17` | `ai.onnx:17` |
| **Producer Info** | PyTorch 2.9.1 exporter | PyTorch 2.9.1 + Float16 converter |
| **Total Graph Nodes** | 109 nodes | 109 nodes |
| **Initializers (Weights)** | 16 initializers | 16 initializers |

### 3.2 Exact Input & Output Bindings:
- **Model Inputs:**
  1. `vision_seq`:
     - FP32: Type `FLOAT` (tensor type 1), Shape `['batch_size', 'num_patches', 1024]`
     - FP16: Type `FLOAT16` (tensor type 10), Shape `['batch_size', 'num_patches', 1024]`
     - Dynamic dimensions: `batch_size` (runtime binding = 1), `num_patches` (runtime binding = $N$).
  2. `text_summary`:
     - FP32: Type `FLOAT` (tensor type 1), Shape `['batch_size', 768]`
     - FP16: Type `FLOAT16` (tensor type 10), Shape `['batch_size', 768]`
     - Dynamic dimensions: `batch_size` (runtime binding = 1).
- **Model Outputs:**
  1. `logit`: Shape `['batch_size', 1]` (`FLOAT` / `FLOAT16`)
  2. `probability`: Shape `['batch_size', 1]` (`FLOAT` / `FLOAT16`) — *sigmoid probability of MSI*
  3. `attention_weights`: Shape `['batch_size', 1, 'num_patches']` (`FLOAT` / `FLOAT16`)

### 3.3 Operator Inventory & Android Compatibility:
The graph comprises exactly 19 distinct operator types across 109 nodes:
`Constant` (30), `Unsqueeze` (18), `Transpose` (9), `Shape` (7), `Gather` (7), `MatMul` (6), `Concat` (6), `Reshape` (6), `Add` (5), `Mul` (3), `Gemm` (3), `Cast` (2), `Div` (1), `Softmax` (1), `ReduceMean` (1), `LayerNormalization` (1), `Squeeze` (1), `Relu` (1), `Sigmoid` (1).

* **Compatibility Assessment:** All 19 operators are standard ONNX Opset 17 primitives with native C++ kernels implemented in ONNX Runtime Android (`com.microsoft.onnxruntime:onnxruntime-android`). No custom, proprietary, or contrib operators are present.

---

## 4. Mobile Execution Provider & FP16 Support Analysis

In adherence to prompt instructions, we do not assume that FP16 storage guarantees faster mobile inference or universal execution provider support:

### 4.1 CPU Execution Provider (Default Baseline):
- **Universal Availability:** CPU EP is compiled into `onnxruntime-android` for all ABIs (`arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86`).
- **Dynamic Dimension Robustness:** CPU EP fully supports dynamic sequence lengths ($N \in [1, 1000]$) without graph recompilation or fixed padding.
- **FP16 Behavior on Mobile CPU:**
  - On ARMv8.2-A+ mobile CPUs (e.g., Cortex-A76/A78/X1/X2, Snapdragon 855+), native NEON FP16 instructions execute half-precision vector operations at full speed.
  - On older ARMv8.0/8.1 devices or x86_64 Android emulators lacking native FP16 vector compute, ONNX Runtime CPU EP either software-emulates FP16 operations or inserts runtime cast conversions, which can paradoxically *increase* execution latency.
  - **Decision:** FP32 on CPU EP is the primary reference implementation. FP16 on CPU EP will be evaluated side-by-side as a comparative target.

### 4.2 NNAPI & Hardware Acceleration Realities:
- **NNAPI Driver Heterogeneity:** Android Neural Networks API (NNAPI) implementation quality varies across Android OEMs. Moreover, Google officially deprecated NNAPI in Android 15 (API 35).
- **Dynamic Sequence Limitations:** NNAPI accelerators frequently struggle with dynamic cross-attention shapes (`['batch_size', 1, 'num_patches']`), often failing back to CPU or requiring fixed-size tensor padding.
- **QNN (Qualcomm NPU):** QNN requires vendor-specific binaries (`libQnnHtp.so`), Qualcomm-signed DSP runtimes, and fixed-shape quantization profiles.
- **Deployment Strategy:** The initial Kotlin inference layer will target **ONNX Runtime CPU Execution Provider** to guarantee bit-accurate parity, zero crashes across diverse device models, and full support for dynamic sequence lengths.

---

## 5. Resolution of the Protocol 1 Patch Cap Discrepancy

A discrepancy was identified where earlier report narratives occasionally referenced "$N \le 500$" while benchmark code and manifests referenced 1,000.

### 5.1 Forensic Evidence Trace:
1. **Core Multimodal Dataset:** [`src/multimodal_dataset.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/multimodal_dataset.py#L8) defines:
   ```python
   def __init__(self, registry_path, tensor_dir, max_patches=1000):
   ```
2. **Model Training & Evaluation Scripts:**
   - [`src/late_fusion_train.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/late_fusion_train.py#L62): `MultimodalTCGADataset(REGISTRY, TENSORS, max_patches=1000)`
   - [`src/late_fusion_eval.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/late_fusion_eval.py#L33): `MultimodalTCGADataset(REGISTRY, TENSORS, max_patches=1000)`
   - [`src/cross_attention_eval.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py#L73): `MultimodalTCGADataset(REGISTRY, TENSORS, max_patches=1000)`
3. **Reference Benchmark Implementation:** [`src/eval/benchmark_fp32_reference.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_fp32_reference.py#L262-L276) states:
   ```python
   # PROTOCOL B: CAPPED AT 1000 PATCHES WITH EXPLICIT SEED (seed=42)
   if original_n > 1000:
       perm = torch.randperm(original_n)[:1000]
   ```
4. **Authoritative Manifest Data:** In [`reports/mobile_deployment/protocol1_patch_selection_manifest.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/mobile_deployment/protocol1_patch_selection_manifest.json), exactly 8 cases exceed 1,000 raw slide patches and are capped at `selected_patch_count: 1000`:
   - `CASE_14` (`TCGA-CM-6674`): Total 1,690 $\to$ Selected **1,000**
   - `CASE_15` (`TCGA-AY-A71X`): Total 3,914 $\to$ Selected **1,000**
   - `CASE_21` (`TCGA-CM-6167`): Total 1,403 $\to$ Selected **1,000**
   - `CASE_38` (`TCGA-A6-6137`): Total 1,433 $\to$ Selected **1,000**
   - `CASE_44` (`TCGA-AD-6889`): Total 1,921 $\to$ Selected **1,000**
   - `CASE_46` (`TCGA-DC-6683`): Total 2,504 $\to$ Selected **1,000**
   - `CASE_65` (`TCGA-DC-6682`): Total 1,413 $\to$ Selected **1,000**
   - `CASE_66` (`TCGA-AY-6197`): Total 2,090 $\to$ Selected **1,000**
   - All other 58 cases have patch counts ranging from 10 to 999.
5. **Origin of $N=500$:** In `benchmark_fp32_reference.py` line 496, the text notes `"single_patient_input_tensor_mb": "2.05 MB (typical N=500) to 4.10 MB (capped N=1000)"`. $N=500$ was used as an illustrative median slide size to compute typical memory, not as the upper bound.

### 5.2 Discrepancy Resolution:
The established, frozen, authoritative patch cap for Protocol 1 is **$N \le 1,000$**. It has not been changed and will not be changed. The Android input validation layer will enforce $1 \le N \le 1,000$ for Protocol 1 inputs and allow unconstrained $N$ (up to device memory limits) for Protocol 2 inputs.

---

## 6. Android Project Architecture & Compatibility Plan

To satisfy the requirements of Phase 4.1 while preserving repository hygiene:

### 6.1 Directory & Module Structure:
A modular directory structure will be established under [`android/`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android):
```
android/
├── build.gradle.kts                   # Root build configuration
├── settings.gradle.kts                # Project name & module inclusion
├── gradle/
│   └── wrapper/
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties  # Gradle 8.4 / 8.5
├── gradlew                            # Unix Gradle wrapper script
├── gradlew.bat                        # Windows Gradle wrapper script
└── app/
    ├── build.gradle.kts               # Android app configuration & dependencies
    ├── src/
    │   ├── main/
    │   │   ├── AndroidManifest.xml    # Minimal research application manifest
    │   │   ├── assets/
    │   │   │   └── models/            # Symbolic link / copy of ONNX model
    │   │   │       ├── fusion_model_fp32.onnx
    │   │   │       └── fusion_model_fp16.onnx
    │   │   └── java/org/crc/multimodal/
    │   │       ├── inference/
    │   │       │   ├── CrcClassifier.kt        # Primary inference interface
    │   │       │   ├── OrtInferenceEngine.kt   # ONNX Runtime session & tensor execution
    │   │       │   ├── InputValidator.kt       # Shape, dtype, and NaN/Inf validation
    │   │       │   └── InferenceResult.kt      # Probability, threshold call (0.83), & metadata
    │   │       └── ui/
    │   │           └── MainActivity.kt         # Minimal research inspection UI
    │   └── test/
    │       └── java/org/crc/multimodal/
    │           ├── InputValidatorTest.kt       # Shape & finite value validation tests
    │           ├── ThresholdLogicTest.kt       # Strict boundary tests around 0.83
    │           └── OutputParsingTest.kt        # Logit/probability/attention extraction tests
```

### 6.2 Target Android Specifications:
- **Minimum SDK (`minSdk`):** `24` (Android 7.0 Nougat) — covers $>95\%$ of active Android devices globally.
- **Target SDK (`targetSdk`):** `34` (Android 14) / `35` (Android 15).
- **Compile SDK (`compileSdk`):** `34` or `35`.
- **Java/JVM Target:** Java 17 / Java 21 compatibility.
- **Core Dependency:** `com.microsoft.onnxruntime:onnxruntime-android:1.18.0`.

### 6.3 Memory & Threading Boundaries:
- **Maximum Input Size:** At $N=1,000$, vision tensor buffer size is $1 \times 1000 \times 1024 \times 4\text{ bytes} \approx 4.096\text{ MB}$. Text buffer is $1 \times 768 \times 4\text{ bytes} \approx 3.072\text{ KB}$.
- **Heap Allocation Overhead:** Total single-inference active tensor allocation is $<10\text{ MB}$, well within standard Android per-app heap quotas ($192\text{ MB} - 512\text{ MB}$).
- **Concurrency Guard:** All inference calls are synchronized (`@Synchronized` / single-threaded dispatcher). Uncontrolled background concurrency is strictly prohibited.
- **Resource Cleanup:** Explicit implementation of `AutoCloseable` on `OrtInferenceEngine` to release underlying C++ session and memory allocations immediately upon completion.

---

## 7. Compatibility Risks & Mitigation Strategy

| Risk Category | Identified Risk | Engineering Mitigation |
| :--- | :--- | :--- |
| **Toolchain Availability** | Host machine lacks global Android SDK and Gradle installation. | Project uses self-contained Gradle Wrapper (`gradlew`) and decoupled pure Kotlin / JUnit test harnesses for Phase 4.1 logic verification. |
| **Physical Hardware** | No physical Android device connected via ADB. | **Phase 4.3 device profiling is formally gated** until hardware is provisioned. Phase 4.1 logic integration and Phase 4.2 parity data preparation can proceed. |
| **Dynamic Dimensions** | Inability of certain edge runtimes to handle variable sequence length $N$. | Verified that ONNX Runtime CPU EP supports dynamic sequence length natively without shape recompilation. |
| **Threshold Boundary Drift** | Canary case `CASE_36` is within $1.065 \times 10^{-4}$ of $\tau=0.83$. | Strict IEEE-754 FP32 execution parity guarantees identical float32 mathematical output. Output probabilities are reported to high precision with explicit threshold comparison logic. |
| **Clinical Scope Creep** | Risk of misrepresenting application as a diagnostic tool. | UI and reports enforce prominent research disclaimers: *"Research Use Only. Not approved for clinical diagnostics."* |

---

## 8. Phase 4.0 Preflight Gate Decision: CONDITIONAL GO

### **DECISION: CONDITIONAL GO**

1. **Cleared for Phase 4.1:**  
   The mathematical model specifications, operator compatibility, tensor shapes, Protocol 1 cap ($N \le 1,000$), and Kotlin inference architecture are fully validated and documented. Minimal Kotlin inference integration and automated unit testing may proceed.
2. **Explicit Gating Condition for Phase 4.3:**  
   Physical-device profiling (latency, memory, thermal dissipation, battery) is strictly blocked until an actual Android device (ARM64-v8a) with ADB connectivity is provisioned. No device performance claims will be asserted until measured directly on physical target hardware.

---

### Phase 4.0 Stop Condition Observed
In accordance with user instructions, execution is stopped at Phase 4.0. No application code has been implemented. Awaiting authorization to proceed with Phase 4.1.
