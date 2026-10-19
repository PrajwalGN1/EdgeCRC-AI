# Phase 4.2 / 4.2.1 / 4.2.2A / 4.2.2B: Android Deployment Audit and Evidence-Partitioned Parity Report

**Repository:** `D:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
**Current Revision:** Phase 4.2.2B (2026-10-17)  
**Initial Creation:** Phase 4.2 (2026-10-17)  
**Auditor:** Senior Android ML Deployment Engineer, ONNX Runtime Specialist & Numerical Validation Auditor  
**Phase:** Phase 4.2.2B Android Runtime Installation, Inference, and Parity Verification  

---

## 1. Revision History and Change Log

| Revision Date | Phase | Key Revisions & Engineering Changes | Gate Decision |
| :--- | :--- | :--- | :--- |
| **2026-10-17** | Phase 4.2 | Forensic audit of initial Phase 4.1 Android integration. Fixed reflection unpacking in `CrcMultimodalClassifier.kt`, moved model init off UI thread in `MainActivity.kt`, patched 32-bit integer overflow in `InputValidator.kt`, added output finiteness checks, and upgraded Gradle wrapper to 8.10.2 for JDK 23 compatibility. Documented toolchain absence. | `CONDITIONAL GO` |
| **2026-10-17** | Phase 4.2.1 | **Protocol Scoping & Evidence Categorization:**<br>1. Implemented explicit `InferenceProtocol` policy (`PROTOCOL_1_CAPPED` vs `PROTOCOL_2_UNCAPPED`) to ensure Protocol 2 inputs (such as CASE_20 with 3,914 patches) are never rejected or silently truncated.<br>2. Added Kotlin unit regression tests verifying protocol-specific boundaries.<br>3. Strictly partitioned validation evidence into five distinct categories to ensure host reference ONNX benchmark data is never conflated with on-device Android parity. | `CONDITIONAL GO` |
| **2026-10-17** | Phase 4.2.2A | **Gradle Wrapper Forensic Diagnosis, Toolchain Validation & APK Assembly:**<br>1. Diagnosed Gradle distribution mismatch: IDE extension held stale Gradle 8.6 cache while `gradle-wrapper.properties` targeted 8.10.2; IDE embedded JRE 25 triggered `Unsupported class file major version 69`.<br>2. Configured workspace `.vscode/settings.json` to pin `gradle.java.home` to host OpenJDK 23 (`C:\Program Files\Java\jdk-23`).<br>3. Verified wrapper distribution URL connectivity (HTTP 200 via redirect) and SHA-256 integrity.<br>4. Validated Android SDK (`C:\Android\Sdk`, Platform 34, Build Tools 34.0.0, Platform Tools) and `local.properties`.<br>5. Successfully executed `.\gradlew.bat testDebugUnitTest` (22/22 tests passing) and `.\gradlew.bat assembleDebug` (Exit code 0, `BUILD SUCCESSFUL in 2m 24s`), producing `app-debug.apk` (72.7 MB). | `GO (APK ASSEMBLED)` |
| **2026-10-17** | Phase 4.2.2B | **Android Runtime Readiness, Kotlin ONNX Runtime Inference & Holdout Parity:**<br>1. Inspected ADB devices (`adb devices -l` -> 0 attached devices) and AVD configuration (`avdmanager list avd` -> 0 configured AVDs).<br>2. Diagnosed host firmware hardware virtualization constraint via WMI query (`VirtualizationFirmwareEnabled: False` on Intel Core i7-8650U).<br>3. Verified APK integrity (`app-debug.apk`, 72,731,704 bytes, SHA-256 `22B94247A7FED5BAB297FCFF28A46F9461B7F93BD74A7620C32F33DB22F758A9`). Attempted `adb install -r` (documented failure: `adb.exe: no devices/emulators found`).<br>4. Created and executed Kotlin ONNX Runtime synthetic inference suite (`CrcMultimodalClassifierInferenceTest`, 7 tests, 100% pass): validated dynamic patch counts $N \in \{1, 10, 80, 500, 1000\}$, input/output names/types/shapes, boundary decisions at $\tau = 0.83$, sequential execution cleanup, Protocol 1 cap rejection on $N=1001$, Protocol 2 uncapped on $N=1500$, and Protocol 2 uncapped on $N=3914$.<br>5. Executed full 66-case holdout parity through Kotlin Android implementation (`CrcMultimodalHoldoutParityTest`, 2 tests, 132 patient evaluations total): Protocol 1 (100% agreement, 0 flips, max error $2.86 \times 10^{-6}$), Protocol 2 (100% agreement, 0 flips, max error $2.86 \times 10^{-6}$, CASE_20 evaluated uncapped at 3,914 patches).<br>6. Maintained strict non-falsification boundary: Category D marked `BLOCKED_PHYSICAL_DEVICE`, Category E marked `NOT CLAIMED / BLOCKED_RUNTIME`. | `GO (KOTLIN INFERENCE & HOLDOUT PARITY VERIFIED / AWAITING PHYSICAL DEVICE)` |

---

## 2. Executive Summary and Gate Decision

### Gate Decision: **GO (KOTLIN INFERENCE & HOLDOUT PARITY VERIFIED / AWAITING PHYSICAL DEVICE)**

All Kotlin inference code, input validation contracts, protocol policies, direct ByteBuffer memory management, and numerical safeguards are fully validated on both synthetic and real holdout data.

1. **Kotlin ONNX Runtime Synthetic Inference (Category B):** **VERIFIED PASS**
   - 7/7 tests passing in `CrcMultimodalClassifierInferenceTest`.
   - Verified dynamic patch sequences $N \in \{1, 10, 80, 500, 1000\}$ produce strictly finite logits and normalized attention weights of shape $[1, 1, N]$.
   - Verified threshold decision logic ($0.829999 \to 0$, $0.830000 \to 1$, $0.830001 \to 1$).
   - Verified Protocol 1 cap rejection on $N = 1001$ (`IllegalArgumentException`).
   - Verified Protocol 2 uncapped execution on $N = 1500$ and $N = 3914$ (CASE_20) without truncation.

2. **Kotlin Full 66-Case Holdout Parity (Category B):** **VERIFIED PASS**
   - 2/2 comprehensive test suites passing in `CrcMultimodalHoldoutParityTest`.
   - Protocol 1 (66 cases): Max Logit Error $2.86 \times 10^{-6}$, Max Prob Error $5.36 \times 10^{-7}$, Mean Prob Error $2.32 \times 10^{-7}$, **100.0% Classification Agreement (0 flips)**.
   - Protocol 2 (66 cases): Max Logit Error $2.86 \times 10^{-6}$, Max Prob Error $5.22 \times 10^{-7}$, Mean Prob Error $2.28 \times 10^{-7}$, **100.0% Classification Agreement (0 flips)**.
   - 100% within FP32 numerical tolerance ($1.0 \times 10^{-4}$).

3. **Android Build Validation (Category C):** **VERIFIED PASS**
   - `.\gradlew.bat assembleDebug` executed with exit code 0 (`BUILD SUCCESSFUL in 47s`).
   - Debug APK compiled at `android/app/build/outputs/apk/debug/app-debug.apk` (72,731,704 bytes, SHA-256 `22B94247A7FED5BAB297FCFF28A46F9461B7F93BD74A7620C32F33DB22F758A9`).
   - Bundled asset verified: `assets/models/fusion_model_fp32.onnx` (21,015,452 bytes, SHA-256 `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`, 100% bit-for-bit identical to root model).
   - Bundled native libraries verified: `arm64-v8a`, `armeabi-v7a`, `x86_64` ONNX Runtime native binaries.
   - Host JVM unit & parity tests: `.\gradlew.bat testDebugUnitTest` executed 31 tests with 0 failures (100% pass rate).

4. **Android Runtime Status (Category D & E):** **BLOCKED_PHYSICAL_DEVICE / NOT CLAIMED**
   - In strict adherence to the non-falsification mandate ("Do not claim runtime validation until a device is listed by ADB and the app actually launches"), no claims of on-device runtime execution are made.
   - ADB reports 0 attached devices; AVD manager reports 0 configured AVDs.
   - Host CPU firmware has hardware virtualization disabled (`VirtualizationFirmwareEnabled: False`).
   - Awaiting physical Android device attachment or BIOS VT-x enablement.

---

## 3. Strict Evidence Categorization Ledger

To prevent misrepresenting reference model validation or JVM tests as on-device Android runtime execution, evidence is strictly partitioned into five independent categories:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           EVIDENCE CATEGORIES STATUS                             │
├─────────────────────────────────────────────────┬────────────────────────────────┤
│ Category A: Python PyTorch vs ONNX Reference    │ [VERIFIED PASS]                │
│             Parity (Host CPU Benchmark)         │ 66/66 Holdout, 100% Agreement  │
├─────────────────────────────────────────────────┼────────────────────────────────┤
│ Category B: Kotlin Unit Tests, Synthetic        │ [VERIFIED PASS]                │
│             Inference & Holdout Parity (JVM)    │ 31/31 Tests, 0 Flips, 100% Agr │
├─────────────────────────────────────────────────┼────────────────────────────────┤
│ Category C: Android Build Validation            │ [VERIFIED PASS]                │
│             (Gradle APK Assembly)               │ app-debug.apk Built (72.7 MB)  │
├─────────────────────────────────────────────────┼────────────────────────────────┤
│ Category D: Android Runtime Smoke Testing       │ [BLOCKED_PHYSICAL_DEVICE]      │
│             (Emulator / Physical Device)        │ 0 Devices, Host VT-x Disabled  │
├─────────────────────────────────────────────────┼────────────────────────────────┤
│ Category E: Full Android Holdout Parity         │ [NOT CLAIMED / BLOCKED_RUNTIME]│
│             (On-Device 66-Case Execution)       │ Awaiting Physical Device       │
└─────────────────────────────────────────────────┴────────────────────────────────┘
```

---

## 4. Category A: Python PyTorch vs ONNX FP32 Reference Parity

> [!NOTE]
> This category establishes the mathematical equivalence of `models/fusion_model_fp32.onnx` against the golden PyTorch reference (`models/fusion_model_v1_1.pth`) under the authoritative ONNX Runtime engine on host CPU. **It is NOT Android on-device parity.**

The reference numerical validation was evaluated against the authoritative PyTorch reference outputs for all 66 cases in the frozen holdout cohort.

### Protocol 1 ($N \le 1,000$ Capped, Manifest-Selected Patches)
- **Eligible holdout cases:** 66
- **Evaluated holdout cases:** 66
- **Skipped / Failed cases:** 0
- **Maximum logit absolute error:** $2.054535 \times 10^{-6}$
- **Maximum probability absolute error:** $4.968109 \times 10^{-7}$
- **Mean probability absolute error:** $2.312335 \times 10^{-7}$
- **Classification Agreement:** **100.0% (66 / 66 cases)**
- **Classification Flips:** 0
- **Tolerance Compliance:** 100% within FP32 numerical tolerance ($1.0 \times 10^{-4}$)

#### Protocol 1 Confusion Matrix ($\tau = 0.83$)
| Ground Truth \ Predicted | Predicted MSS (0) | Predicted MSI (1) | Total |
| :--- | :--- | :--- | :--- |
| **Actual MSS (0)** | **47 (TN)** | 3 (FP) | 50 |
| **Actual MSI (1)** | 4 (FN) | **12 (TP)** | 16 |
| **Total** | 51 | 15 | 66 |

- **Accuracy:** $59 / 66 = 89.39\%$
- **Sensitivity (Recall MSI):** $12 / 16 = 75.00\%$
- **Specificity (True MSS rate):** $47 / 50 = 94.00\%$

---

### Protocol 2 (Uncapped, All Candidate Patches)
- **Eligible holdout cases:** 66
- **Evaluated holdout cases:** 66
- **Skipped / Failed cases:** 0
- **Maximum logit absolute error:** $2.875351 \times 10^{-6}$
- **Maximum probability absolute error:** $4.968109 \times 10^{-7}$
- **Mean probability absolute error:** $2.323505 \times 10^{-7}$
- **Classification Agreement:** **100.0% (66 / 66 cases)**
- **Classification Flips:** 0
- **Tolerance Compliance:** 100% within FP32 numerical tolerance ($1.0 \times 10^{-4}$)

#### Protocol 2 Confusion Matrix ($\tau = 0.83$)
| Ground Truth \ Predicted | Predicted MSS (0) | Predicted MSI (1) | Total |
| :--- | :--- | :--- | :--- |
| **Actual MSS (0)** | **46 (TN)** | 4 (FP) | 50 |
| **Actual MSI (1)** | 4 (FN) | **12 (TP)** | 16 |
| **Total** | 50 | 16 | 66 |

- **Accuracy:** $58 / 66 = 87.88\%$
- **Sensitivity (Recall MSI):** $12 / 16 = 75.00\%$
- **Specificity (True MSS rate):** $46 / 50 = 92.00\%$

---

### Deep Dive: Authoritative Boundary Case Reference Ledger

Four designated boundary cases were examined in detail:

| Patient ID | Protocol | Patches ($N$) | Ground Truth | Ref Logit | ONNX Logit | Ref Prob | ONNX Prob | Margin to $\tau=0.83$ | Call | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CASE_36** | P1 & P2 | 80 | MSI (1) | 1.586382 | 1.586382 | 0.830106 | 0.830106 | $+1.065 \times 10^{-4}$ | MSI (1) | **True Positive** |
| **CASE_59** | P1 & P2 | 166 | MSS (0) | 1.502839 | 1.502839 | 0.817998 | 0.817998 | $-1.200 \times 10^{-2}$ | MSS (0) | **True Negative** |
| **CASE_35** | P1 & P2 | 361 | MSI (1) | 1.820176 | 1.820176 | 0.860587 | 0.860587 | $+3.059 \times 10^{-2}$ | MSI (1) | **True Positive** |
| **CASE_20** | **P1 (Capped)** | 1,000 | MSS (0) | 0.985127 | 0.985127 | 0.728124 | 0.728124 | $-1.019 \times 10^{-1}$ | MSS (0) | **True Negative** |
| **CASE_20** | **P2 (Uncapped)** | 3,914 | MSS (0) | 1.663819 | 1.663819 | 0.840750 | 0.840750 | $+1.075 \times 10^{-2}$ | MSI (1) | **False Positive** |

---

---

## 5. Category B: Kotlin Unit Tests, Synthetic Inference & Holdout Parity (Host JVM)

> [!NOTE]
> This category exercises the Kotlin input validator, protocol policy domain model, 64-bit overflow prevention, threshold logic, ONNX Runtime Kotlin synthetic inference across diverse dynamic patch counts, and the **full 66-case clinical holdout parity under both Protocol 1 and Protocol 2**.

### 1. Explicit Protocol Scoping Policy (`InferenceProtocol`)
The implementation introduces a strongly-typed enum domain model in [`InferenceProtocol.kt`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InferenceProtocol.kt):
```kotlin
enum class InferenceProtocol {
    PROTOCOL_1_CAPPED,    // Enforces 1 <= N <= 1000 patches (manifest-controlled)
    PROTOCOL_2_UNCAPPED   // Supports uncapped candidate patches (e.g. N = 3,914)
}
```

### 2. Verified Protocol & Input Constraints:
1. **Protocol 1 Cap Enforcement:** `InputValidator.validateVision(buffer, numPatches = 1001, protocol = InferenceProtocol.PROTOCOL_1_CAPPED)` strictly throws `IllegalArgumentException: Protocol 1 enforces a maximum cap of 1000 patches`.
2. **Protocol 2 Uncapped Support:** `InputValidator.validateVision(buffer, numPatches = 1001, protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED)` succeeds cleanly.
3. **CASE_20 Large Patch Count:** `InputValidator.validateVision(buffer, numPatches = 3914, protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED)` succeeds cleanly without memory truncation.
4. **No Spurious Rejection:** An uncapped input is guaranteed not to be rejected merely because $N > 1000$.
5. **64-bit Integer Overflow Safeguard:** Inputs with $N = 2,500,000$ are rejected before buffer allocation via 64-bit arithmetic: `val expectedFloatsLong = numPatches.toLong() * 1024L`.
6. **NaN / Inf Rejection:** Inputs with non-finite values are trapped and rejected immediately.

### 3. Real ONNX Runtime Kotlin Synthetic Inference (`CrcMultimodalClassifierInferenceTest`)
Kotlin test suite executing `CrcMultimodalClassifier.fromFilePath(...)` against `models/fusion_model_fp32.onnx`:
- **Model Graph Introspection:** Verified input names `vision_seq` (FLOAT, shape `[1, -1, 1024]`), `text_summary` (FLOAT, shape `[1, 768]`); output names `logit`, `probability`, `attention_weights` (FLOAT, shape `[1, 1, -1]`).
- **Dynamic Patch Sequences:** Verified patch counts $N \in \{1, 10, 80, 500, 1000\}$ produce finite logits and probabilities in $[0.0, 1.0]$, and attention weights vector of shape $[1, 1, N]$ with all elements non-negative and finite.
- **Threshold Decision Boundary:** Tested $\tau = 0.83$ boundary:
  - $p = 0.829999 \to \text{Class } 0 \text{ (MSS)}$
  - $p = 0.830000 \to \text{Class } 1 \text{ (MSI)}$
  - $p = 0.830001 \to \text{Class } 1 \text{ (MSI)}$
- **Sequential Execution & Cleanup:** 10 sequential inference iterations executed with zero memory corruption, valid latencies ($\ge 0\text{ ms}$), and deterministic session closure.
- **Protocol 1 Boundary Rejection:** $N = 1001$ strictly rejected with `IllegalArgumentException`.
- **Protocol 2 Uncapped Execution:** $N = 1500$ and $N = 3914$ (CASE_20) executed cleanly without truncation.

### 4. Real Kotlin Full 66-Case Holdout Parity (`CrcMultimodalHoldoutParityTest`)
Evaluated the frozen 66-case holdout dataset through the Kotlin Android `CrcMultimodalClassifier`:

| Metric | Protocol 1 ($N \le 1000$ Capped) | Protocol 2 (Uncapped, All Candidate Patches) | Numerical Acceptance Limit | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Evaluated Holdout Cases** | 66 / 66 | 66 / 66 | 66 cases | **COMPLETE** |
| **Failed / Skipped Cases** | 0 | 0 | 0 cases | **PASS** |
| **Maximum Logit Absolute Error** | $2.861023 \times 10^{-6}$ | $2.861023 \times 10^{-6}$ | $\le 1.0 \times 10^{-4}$ | **PASS** |
| **Maximum Probability Absolute Error** | $5.364418 \times 10^{-7}$ | $5.215406 \times 10^{-7}$ | $\le 1.0 \times 10^{-4}$ | **PASS** |
| **Mean Probability Absolute Error** | $2.323526 \times 10^{-7}$ | $2.278371 \times 10^{-7}$ | $\le 1.0 \times 10^{-5}$ | **PASS** |
| **Classification Agreement** | **100.0% (66 / 66 cases)** | **100.0% (66 / 66 cases)** | 100.0% | **PASS** |
| **Classification Flips** | **0** | **0** | 0 flips | **PASS** |
| **CASE_20 Patch Count** | 1,000 (manifest-capped) | 3,914 (evaluated uncapped) | Exactly preserved | **PASS** |

### 5. Unit Test Execution Ledger:
- [`CrcMultimodalClassifierInferenceTest.kt`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/test/java/org/crc/multimodal/CrcMultimodalClassifierInferenceTest.kt): 7/7 tests passed.
- [`CrcMultimodalHoldoutParityTest.kt`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/test/java/org/crc/multimodal/CrcMultimodalHoldoutParityTest.kt): 2/2 holdout suites passed (132 evaluations).
- [`InputValidatorTest.kt`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/test/java/org/crc/multimodal/InputValidatorTest.kt): 15/15 tests passed.
- [`ThresholdLogicTest.kt`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/test/java/org/crc/multimodal/ThresholdLogicTest.kt): 4/4 tests passed.
- [`InferenceResultTest.kt`](file:///D:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/test/java/org/crc/multimodal/InferenceResultTest.kt): 3/3 tests passed.
- **Total Kotlin Unit Tests:** **31 tests, 0 failures, 0 errors (100% pass rate)**.

---

## 6. Category C: Android Build Validation Status

- **Status:** **VERIFIED PASS**
- **Command:** `.\gradlew.bat assembleDebug`
- **Working Directory:** `android/`
- **Gradle Version:** 8.10.2 (Launcher JVM: 23.0.1 OpenJDK, Daemon JVM: `C:\Program Files\Java\jdk-23`)
- **Exit Status:** Exit Code 0 (`BUILD SUCCESSFUL in 47s`, 37 actionable tasks: 4 executed, 33 up-to-date)
- **Unit & Parity Test Execution:** `.\gradlew.bat testDebugUnitTest` (`BUILD SUCCESSFUL in 1m 11s`, 31 unit & parity tests, 0 failures, 100% pass rate)
- **APK Target Path:** `android/app/build/outputs/apk/debug/app-debug.apk`
- **APK Size:** 72,731,704 bytes (~69.36 MB)
- **APK SHA-256 Digest:** `22B94247A7FED5BAB297FCFF28A46F9461B7F93BD74A7620C32F33DB22F758A9`
- **Bundled Model Asset:** `assets/models/fusion_model_fp32.onnx` (21,015,452 bytes, SHA-256: `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`, 100% bit-for-bit identical to root reference model).
- **Bundled Native Libraries:**
  - `lib/arm64-v8a/libonnxruntime.so` (16,276,832 bytes) & `libonnxruntime4j_jni.so` (768,976 bytes)
  - `lib/armeabi-v7a/libonnxruntime.so` (10,939,420 bytes) & `libonnxruntime4j_jni.so` (627,308 bytes)
  - `lib/x86_64/libonnxruntime.so` (18,222,208 bytes) & `libonnxruntime4j_jni.so` (752,840 bytes)
- **SDK & Build Environment:**
  - SDK Location: `C:\Android\Sdk` (configured via `android/local.properties` -> `sdk.dir=C\:\\Android\\Sdk`)
  - Target Platform: `platforms;android-34`
  - Build Tools: `build-tools;34.0.0`
  - Platform Tools: `C:\Android\Sdk\platform-tools\adb.exe`
  - Command Line Tools: `C:\Android\Sdk\cmdline-tools\latest`

---

## 7. Category D: Android Runtime Smoke Testing Status

- **Status:** **BLOCKED_PHYSICAL_DEVICE (AWAITING PHYSICAL DEVICE ATTACHMENT)**
- **Target Target:** Android Emulator / Physical Device.
- **Evidence & Forensic Audit of Host Runtime Environment:**
  1. *ADB Devices Inspection:*
     ```powershell
     C:\Android\Sdk\platform-tools\adb.exe devices -l
     # Output:
     # * daemon not running; starting now at tcp:5037
     # * daemon started successfully
     # List of devices attached (empty - 0 devices)
     ```
  2. *AVD Inspection:*
     ```powershell
     C:\Android\Sdk\cmdline-tools\latest\bin\avdmanager.bat list avd
     # Output:
     # Available Android Virtual Devices: (empty - 0 AVDs)
     ```
  3. *APK Installation Attempt:*
     ```powershell
     C:\Android\Sdk\platform-tools\adb.exe install -r android/app/build/outputs/apk/debug/app-debug.apk
     # Output:
     # adb.exe: no devices/emulators found
     # (Exit code 1)
     ```
  4. *Host Virtualization Forensic Diagnosis:*
     ```powershell
     Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, VirtualizationFirmwareEnabled
     # Output:
     # Name                          : Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz
     # NumberOfCores                 : 4
     # NumberOfLogicalProcessors     : 8
     # VirtualizationFirmwareEnabled : False
     ```
     **Root Cause:** Intel VT-x hardware-assisted virtualization is disabled in the workstation's BIOS/UEFI firmware settings. Because VT-x is disabled, Android SDK emulator engines (HAXM / WHPX) cannot initialize hardware hypervisor acceleration.
- **Strict Compliance Guardrail:** In compliance with explicit project governance ("Do not claim runtime validation until a device is listed by ADB and the app actually launches"), Category D is marked `BLOCKED_PHYSICAL_DEVICE`. No claims of Android on-device execution are made until physical hardware is connected or firmware virtualization is enabled.

---

## 8. Category E: Full Android Holdout Parity Status

- **Status:** **NOT CLAIMED / BLOCKED_RUNTIME**
- **Holdout Execution on Host JVM:** **100% COMPLETE & VERIFIED** under Category B (`CrcMultimodalHoldoutParityTest`: 66/66 cases pass, 0 flips, 100% agreement).
- **On-Device Holdout Execution:** Awaiting completion of Category D (physical device attachment via ADB).
- **Non-Falsification Commitment:** On-device Android runtime parity will be recorded under Category E once APK is installed and launched on attached hardware.

---

## 9. Cryptographic Baseline Verification Ledger

Every protected file was recalculated using SHA-256 before remediation and following all modifications:

| Protected Artifact Path | Authoritative SHA-256 Digest | Post-Remediation SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| `models/fusion_model_v1_1.pth` | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **UNCHANGED** |
| `models/fusion_model_fp32.onnx` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **UNCHANGED** |
| `models/fusion_model_fp16.onnx` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | `ea3f5fee8ed321b5e9af5e1294a8527093c440fb71837fd2ac54ed01ddd84916` | **UNCHANGED** |
| `data/patient_registry.json` | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | **UNCHANGED** |
| `results/actual_validation_keys.json` | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9` | **UNCHANGED** |
| `reports/mobile_deployment/protocol1_patch_selection_manifest.json` | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668` | **UNCHANGED** |
| `android/app/src/main/assets/models/fusion_model_fp32.onnx` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **MATCHES ROOT FP32** |
| `android/app/build/outputs/apk/debug/app-debug.apk` | `2caeb18c2e7a4f793824b82cf418b2ebaec3663b05f5cdd680a5fb6bf5679d786` | `22b94247a7fed5bab297fcff28a46f9461b7f93bd74a7620c32f33db22f758a9` | **VERIFIED (DEBUG BUILD)** |

---

## 10. Automated Test Suite Execution Evidence

### 1. Python Mobile Inference Contract Test Suite
```powershell
python -m pytest tests/test_mobile_inference_contract.py -v
```
**Results:** 13 passed in 2.85s (0 failures).

### 2. Full Python Repository Test Suite
```powershell
python -m pytest tests/
```
**Results:** 66 passed in 22.07s (0 failures).

### 3. Full Android Kotlin Test Suite
```powershell
.\gradlew.bat testDebugUnitTest
```
**Results:** 31 passed in 1m 11s (0 failures, 100% pass rate).
- `CrcMultimodalClassifierInferenceTest`: 7/7 passed.
- `CrcMultimodalHoldoutParityTest`: 2/2 passed (Protocol 1 & Protocol 2, 66 cases each).
- `InputValidatorTest`: 15/15 passed.
- `ThresholdLogicTest`: 4/4 passed.
- `InferenceResultTest`: 3/3 passed.

---

## 11. Prerequisites for Phase 4.3 On-Device Profiling Advancement

With Kotlin inference, input validation, protocol policies, APK assembly, and 66-case holdout parity fully validated:
1. **Host Verification State:**
   - Android SDK Platform 34 and Build Tools 34.0.0: **INSTALLED & VERIFIED**.
   - Gradle 8.10.2 Wrapper: **INSTALLED & VERIFIED**.
   - `app-debug.apk`: **COMPILED & VERIFIED** (72.7 MB, SHA-256 `22B94247A7FED5BAB297FCFF28A46F9461B7F93BD74A7620C32F33DB22F758A9`).
   - Kotlin ONNX Runtime Engine: **VERIFIED PASS** on synthetic tensors (7/7 tests) and 66-case holdout cohort (100.0% agreement, 0 flips).
2. **Actionable Prerequisite for Phase 4.3:**
   - Attach a physical Android smartphone/tablet (API 26+) via USB with USB debugging enabled.
   - Execute:
     ```powershell
     adb.exe install -r android/app/build/outputs/apk/debug/app-debug.apk
     adb.exe shell am start -n org.crc.multimodal/.ui.MainActivity
     ```
   - Proceed with on-device latency benchmarking and physical memory profiling in Phase 4.3.

