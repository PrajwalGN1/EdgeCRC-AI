# READ-ONLY CODE AUDIT REPORT: MANUAL SLIDE PATCH COUNT SELECTOR VS PROTOCOL 1 AND PROTOCOL 2

**Project:** Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2  
**Target:** Android Edge AI On-Device Inference Pipeline  
**Auditor:** Senior Android/Kotlin Engineer & ML Inference Pipeline Auditor  
**Date:** 2026-10-27  
**Audit Scope:** Read-only inspection of manual patch selection UI, data flow, validation rules, protocol definitions, tensor allocation, and ONNX execution.

---

## 1. Executive Conclusion

The manual **Slide Patch Count selector** (`10`, `50`, `100`, `500`, `1,000`) present in `fragment_new_specimen.xml` and managed in `NewSpecimenFragment.kt` is **completely decoupled from model inference and is functionally REDUNDANT**. It does **NOT** determine, slice, truncate, downsample, or filter vision patch embeddings entering the ONNX inference session.

Specifically:
1. **Zero Downstream Effect on Vision Tensors:** The manual selector never determines how many patch embeddings enter inference. All vision tensor slicing or patch filtering logic is non-existent in the Android runtime.
2. **Authentic Bundle Authority:** In authentic specimen ingestion, the number of patches ($N$) is immutably defined by the imported `.crcbundle` binary header (`patchCount`) and verified by raw payload SHA-256 digests.
3. **Selector is Overridden and Ignored:** When a `.crcbundle` is imported, `NewSpecimenFragment.kt` overwrites `selectedPatchCount` with `bundle.patchCount`. When confirmed, the entity stored in SQLite records `bundle.patchCount`. During inference in `InferenceExecutionFragment.kt`, the execution pipeline bypasses the database value entirely and sets `executionPatches = bundle.patchCount`.
4. **Strict Buffer Length Enforcement:** `InputValidator.validateVision` strictly mandates that `visionData.size == numPatches * 1024`. Because `visionData` is never sliced at runtime, any divergence between `numPatches` and the authentic bundle patch count causes an immediate, fatal `IllegalArgumentException: Vision buffer size mismatch`.
5. **Protocol 1 Behavior:** Protocol 1 enforces an authoritative ceiling of $N \le 1,000$. It **does NOT dynamically subsample or truncate** patches at edge runtime. If an imported bundle contains $N \le 1,000$, all $N$ patches are evaluated. If an imported bundle contains $N > 1,000$, Protocol 1 strictly rejects the specimen with an `IllegalArgumentException`. (The deterministic `seed=42` patch selection originally occurred during upstream dataset preparation in Python, never dynamically on-device).
6. **Protocol 2 Behavior:** Protocol 2 evaluates all available patches ($N \ge 1$) without an upper cap, preserving 100% of the tissue evidence extracted from the whole-slide image.
7. **Safe Removal Justification:** The manual selector is a legacy UI artifact from early prototyping before authentic `.crcbundle` ingestion was implemented. It can be safely removed and replaced with a read-only badge indicating the authentic patch count detected from the bundle. Removing it will have zero impact on tensor shapes, numerical outputs, classification decisions, or protocol compliance.

---

## 2. Actual Source Files and Methods Inspected

| Component / Layer | Source File | Key Methods / Symbols Inspected |
| :--- | :--- | :--- |
| **UI Layout** | [`fragment_new_specimen.xml`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/res/layout/fragment_new_specimen.xml#L460-L560) | `toggleGroupNewPatches`, `btnPatches10`, `btnPatches50`, `btnPatches100`, `btnPatches500`, `btnPatches1000`, `tvNewSpecimenProtocolBadge`, `tvNewSpecimenTensorDim` |
| **Specimen Ingestion UI** | [`NewSpecimenFragment.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/NewSpecimenFragment.kt#L36-L530) | `selectedPatchCount`, `updateTensorSummary()`, `btnSaveDraft.setOnClickListener`, `btnContinueToReview.setOnClickListener`, `displayImportSuccess()`, `confirmImportAndSave()` |
| **Specimen Review UI** | [`SpecimenReviewFragment.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/SpecimenReviewFragment.kt#L70-L125) | `onViewCreated()`, `tvCheckVisionDetails`, `tvCheckProtocolDetails` |
| **Inference Execution UI** | [`InferenceExecutionFragment.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt#L120-L450) | `toggleGroupProtocol`, `refreshCaseAndProvenanceUI()`, `executeAuthenticInference()`, error handling block |
| **Batch Analysis Engine** | [`BatchCoordinator.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/batch/BatchCoordinator.kt#L180-L450) | `ingestFiles()`, `executeBatch()`, protocol routing logic |
| **Bundle Deserialization** | [`AuthenticBundleLoader.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt#L124-L187) | `loadFromStream()`, `loadFromBytes()`, `loadFromFile()`, binary header parsing, SHA-256 verification |
| **Input Validation Contract** | [`InputValidator.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InputValidator.kt#L27-L66) | `validateVision()`, `PROTOCOL1_MAX_PATCHES`, buffer length validation |
| **Protocol Definitions** | [`InferenceProtocol.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InferenceProtocol.kt#L15-L65) | `InferenceProtocol.PROTOCOL_1_CAPPED`, `InferenceProtocol.PROTOCOL_2_UNCAPPED` |
| **Model Classifier & Tensor Prep** | [`CrcMultimodalClassifier.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/CrcMultimodalClassifier.kt#L148-L230) | `predictAuthentic()`, `predict()`, `OnnxTensor.createTensor`, direct `ByteBuffer` allocation |
| **Domain Entity** | [`ResearchCase.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/data/ResearchCase.kt#L8-L23) | `ResearchCase` data class definition, `patchCount`, `protocol` |
| **Reference Holdout Serializer** | [`serialize_holdout_for_kotlin.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/serialize_holdout_for_kotlin.py#L34-L75) | `serialize_protocol()`, offline pre-slicing using `protocol1_selected_indices` |
| **Upstream Benchmark Reference** | [`benchmark_fp32_reference.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_fp32_reference.py#L274-L279) | Upstream Protocol B sampling: `torch.randperm(original_n)[:1000]` with `torch.manual_seed(42)` |
| **Bundle Export Tool** | [`export_specimen_bundle.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tools/export_specimen_bundle.py#L94-L140) | `export_crcb_bundle()`, binary serialization without truncation |

---

## 3. Selector Data-Flow Trace with Line References

Tracing the user-selected value step by step from UI interaction to ONNX model execution:

```
[UI MaterialButtonToggleGroup]
          │
          ▼ (User clicks button: 10, 50, 100, 500, 1000)
NewSpecimenFragment.kt (Line 180): selectedPatchCount = when(checkedId) { ... }
          │
          ├────────────────────────────────────────┬────────────────────────────────────────┐
          ▼ (Branch A: Import Bundle)              ▼ (Branch B: btnSaveDraft)               ▼ (Branch C: btnContinueToReview)
NewSpecimenFragment.kt (Line 423):        NewSpecimenFragment.kt (Line 236):       NewSpecimenFragment.kt (Line 283):
selectedPatchCount = bundle.patchCount    ResearchCase(                            ResearchCase(
(OVERWRITTEN with bundle count)             patchCount = selectedPatchCount          patchCount = selectedPatchCount
          │                               ) (Saved to SQLite as DRAFT)             ) (Saved to SQLite as READY)
          ▼                                        │                                        │
NewSpecimenFragment.kt (Lines 506-507):            └───────────────────┬────────────────────┘
caseToSave = ResearchCase(                                             │
  patchCount = bundle.patchCount,                                      │
  protocol = if (bundle.patchCount <= 1000)                            │
               PROTOCOL_1_CAPPED else PROTOCOL_2_UNCAPPED              │
)                                                                      │
(OVERWRITTEN: selectedPatchCount discarded)                            │
          │                                                            │
          └─────────────────────────────┬──────────────────────────────┘
                                        │
                                        ▼
                       SpecimenReviewFragment.kt (Lines 84, 111):
                       - Uses currentCase.patchCount solely for UI label rendering
                       - Performs NO tensor slicing, loading, or subsampling
                                        │
                                        ▼
                      InferenceExecutionFragment.kt (Lines 120-135):
                      - User selects Protocol 1 or Protocol 2 via toggleGroupProtocol
                      - selectedProtocol initialized from rCase.protocol
                                        │
                                        ▼
                      InferenceExecutionFragment.kt (Lines 302-310):
                      val bundle = BundleStorageManager.loadBundle(...)
                      visionData = bundle.visionData           <── Full float array (N * 1024 floats)
                      executionPatches = bundle.patchCount     <── Sourced from bundle header
                      (cCase.patchCount and manual selector completely BYPASSED)
                                        │
                                        ▼
                      InputValidator.kt (Lines 27-66):
                      validateVision(visionData, executionPatches, protocol)
                      - Enforces: visionData.size == executionPatches * 1024
                      - If PROTOCOL_1_CAPPED: requires executionPatches <= 1000
                      - If PROTOCOL_2_UNCAPPED: allows arbitrary positive executionPatches
                                        │
                                        ▼
                      CrcMultimodalClassifier.kt (Lines 193-218):
                      - Allocates direct NIO ByteBuffer for visionData.size * 4 bytes
                      - Constructs visionShape = longArrayOf(1L, numPatches.toLong(), 1024L)
                      - Executes session.run(mapOf("vision_embeddings" to visionTensor, ...))
```

### Detailed Step-by-Step Code References

1. **UI Layout Definition:**
   - [`fragment_new_specimen.xml:L501-L550`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/res/layout/fragment_new_specimen.xml#L501-L550): Defines `toggleGroupNewPatches` (`MaterialButtonToggleGroup`) with child buttons `btnPatches10`, `btnPatches50`, `btnPatches100`, `btnPatches500`, and `btnPatches1000`.
2. **Local State Initialization & UI Listener:**
   - [`NewSpecimenFragment.kt:L36`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/NewSpecimenFragment.kt#L36): Initialized to `private var selectedPatchCount: Int = 10`.
   - [`NewSpecimenFragment.kt:L180-L191`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/NewSpecimenFragment.kt#L180-L191): Listener updates `selectedPatchCount` when checked. Calls `updateTensorSummary()`, which only updates two text views: `"Protocol 1 (Capped ≤ 1,000)"` or `"Protocol 2 (Uncapped)"` and `"Shape: [1, $selectedPatchCount, 1024]"`.
3. **Overwritten Upon Bundle Import:**
   - [`NewSpecimenFragment.kt:L423-L431`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/NewSpecimenFragment.kt#L423-L431): In `displayImportSuccess()`, the state is immediately overwritten: `selectedPatchCount = bundle.patchCount`. If the bundle patch count is not one of `{10, 50, 100, 500, 1000}`, it calls `toggleGroupPatchesRef?.clearChecked()`, leaving no button selected.
4. **Discarded Upon Confirmation:**
   - [`NewSpecimenFragment.kt:L506-L508`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/NewSpecimenFragment.kt#L506-L508): In `confirmImportAndSave()`, the `ResearchCase` is constructed with `patchCount = bundle.patchCount`. The `selectedPatchCount` field is completely bypassed.
5. **No Slicing in Review Screen:**
   - [`SpecimenReviewFragment.kt:L84, L111`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/SpecimenReviewFragment.kt#L84): `currentCase.patchCount` is used exclusively for string interpolation in UI display labels (`"Imported Authentic [1, ${currentCase.patchCount}, 1024] Float32 tensor"`). No tensor manipulation occurs.
6. **Bypassed at Inference Execution:**
   - [`InferenceExecutionFragment.kt:L302-L306`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt#L302-L306):
     ```kotlin
     val bundle = BundleStorageManager.loadBundle(requireContext(), cCase.caseIdentifier)
     visionData = bundle.visionData
     textData = bundle.textData
     executionPatches = bundle.patchCount
     ```
     Even if `cCase.patchCount` in the database had been manipulated, the execution pipeline derives `executionPatches` directly and exclusively from `bundle.patchCount`.
7. **Strict Validation Assertion:**
   - [`InputValidator.kt:L53-L55`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InputValidator.kt#L53-L55):
     ```kotlin
     val expectedSize = expectedFloatsLong.toInt()
     require(visionData.size == expectedSize) {
         "Vision buffer size mismatch: expected $expectedSize floats ($numPatches patches * $VISION_FEATURE_DIM dim), got ${visionData.size}"
     }
     ```
     This assertion guarantees that `executionPatches` must match the exact float count of `visionData / 1024`.
8. **Final ONNX Tensor Construction:**
   - [`CrcMultimodalClassifier.kt:L207-L218`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/CrcMultimodalClassifier.kt#L207-L218):
     `val visionShape = longArrayOf(1L, numPatches.toLong(), InputValidator.VISION_FEATURE_DIM.toLong())`
     The tensor shape is constructed using `numPatches` (= `bundle.patchCount`). The ONNX session evaluates all $N$ patches.

---

## 4. Protocol 1 versus Protocol 2 Comparison Table

The following table is grounded in the actual implementation across [`InputValidator.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InputValidator.kt), [`InferenceProtocol.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InferenceProtocol.kt), [`InferenceExecutionFragment.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt), [`BatchCoordinator.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/batch/BatchCoordinator.kt), and upstream scripts [`benchmark_fp32_reference.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_fp32_reference.py) and [`serialize_holdout_for_kotlin.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/serialize_holdout_for_kotlin.py).

| Attribute | Protocol 1 (`PROTOCOL_1_CAPPED`) | Protocol 2 (`PROTOCOL_2_UNCAPPED`) |
| :--- | :--- | :--- |
| **Operational Definition** | Training-aligned baseline protocol capping sequence length at 1,000 patches. | Full-sequence edge deployment candidate evaluating 100% of available tissue patches. |
| **Edge Runtime Patch Selection Algorithm** | **None at runtime.** Verifies $1 \le N \le 1,000$. (Subsampling using `torch.randperm(N)[:1000]` with `seed=42` occurred exclusively upstream during dataset export, never on-device). | **None.** Evaluates all $N$ patches provided in the bundle as extracted. |
| **Maximum Allowed Patch Count** | **Strictly capped at 1,000 patches** (`PROTOCOL1_MAX_PATCHES = 1000`). | **Uncapped** (bounded only by 32-bit direct buffer allocation capacity limit, $>500,000$ patches). |
| **Are All Available Patches Evaluated?** | **Conditionally yes:** If available $N \le 1,000$, all $N$ patches are evaluated. If $N > 1,000$, execution fails. | **Yes:** 100% of available patches are always evaluated. |
| **Is Patch Selection Deterministic?** | **Yes.** Pass-through of authentic input without runtime stochasticity. (Upstream subsampling used fixed seed=42). | **Yes.** Full sequence pass-through without sampling. |
| **Is Patch Order Preserved?** | **Yes.** Order in `.crcbundle` is preserved identically. | **Yes.** Order in `.crcbundle` is preserved identically. |
| **Does Manual Selector Change Selected Patches?** | **No.** Selector has zero impact on patch count or data. | **No.** Selector has zero impact on patch count or data. |
| **Vision Input Tensor Shape** | `[1, N, 1024]` where $N = \text{bundle.patchCount} \le 1000$. | `[1, N, 1024]` where $N = \text{bundle.patchCount}$ (supports $N > 1000$). |
| **Behavior when Bundle Contains $N > 1000$ Patches** | **Rejects execution.** In single-case analysis: throws `IllegalArgumentException: Protocol 1 enforces a maximum cap of 1000 patches, but received N` ([`InputValidator.kt:L38-L40`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InputValidator.kt#L38-L40)). In batch analysis: automatically routes to Protocol 2 if fallback allowed ([`BatchCoordinator.kt:L392-L395`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/batch/BatchCoordinator.kt#L392-L395)). | **Executes normally.** All $N$ patches enter the ONNX session without truncation or downsampling. |
| **Is Silent Truncation Possible?** | **No.** There is no truncation code in the entire Android repository. Inputs are either validated in full or rejected fail-closed. | **No.** All patches are evaluated in full. |

---

## 5. Tensor-Shape Analysis

The vision embedding input contract for the ONNX model (`fusion_model_fp32.onnx`) is:
$$\text{Input: } \texttt{vision\_embeddings} \in \mathbb{R}^{1 \times N \times 1024} \quad (\text{Float32})$$

### How $N$ is Determined at Each Stage

1. **Storage Stage (`.crcbundle` file):**
   - Serialized as:
     `f.write(struct.pack("<iiii", num_patches, VISION_DIM, TEXT_DIM, DATA_TYPE_FLOAT32))`
   - $N$ is fixed at bundle creation time ([`export_specimen_bundle.py:L132`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tools/export_specimen_bundle.py#L132)).
2. **Ingestion Stage (`AuthenticBundleLoader.kt`):**
   - Header parsed: `val patchCount = buffer.int` ([`AuthenticBundleLoader.kt:L142`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/AuthenticBundleLoader.kt#L142)).
   - Buffer allocated: `val visionData = FloatArray(patchCount * visionDim)`.
   - Length in floats: $N \times 1024$.
3. **Validation Stage (`InputValidator.kt`):**
   - Checked: `expectedFloatsLong = numPatches.toLong() * 1024L`
   - Checked: `require(visionData.size == expectedSize)` ([`InputValidator.kt:L53`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InputValidator.kt#L53)).
   - $N$ is strictly validated against `visionData.size / 1024`.
4. **Execution Stage (`CrcMultimodalClassifier.kt`):**
   - Direct byte buffer allocated: `visionBytesLong = visionData.size.toLong() * 4L`
   - Shape array constructed: `longArrayOf(1L, numPatches.toLong(), 1024L)` ([`CrcMultimodalClassifier.kt:L207`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/CrcMultimodalClassifier.kt#L207)).
   - ONNX tensor created: `OnnxTensor.createTensor(env, visionBuffer, visionShape)`.

**Key Architectural Invariant:**  
Because the ONNX runtime requires direct native buffers matching the declared tensor shape, and `InputValidator` strictly verifies that `visionData.size == numPatches * 1024`, **$N$ can never be altered without re-allocating and slicing `visionData`**. Since no slicing code exists anywhere in the Android codebase, $N$ is strictly equal to `bundle.patchCount`.

---

## 6. The Five Hypothetical Examples and Expected Behaviour

### Example 1: 541 available patches and manual selector value set to 100
- **Import:** The `.crcbundle` contains $N = 541$. `NewSpecimenFragment` reads `bundle.patchCount = 541`. It clears the toggle group (since 541 is not 10, 50, 100, 500, or 1000).
- **User Action:** The user taps the `100` button (`btnPatches100`). `selectedPatchCount` in memory becomes `100`.
- **Save/Proceed:** When the user confirms ingestion or proceeds to review, `confirmImportAndSave` executes: `caseToSave.patchCount = bundle.patchCount` (541). (Even if bypassed via draft save, `InferenceExecutionFragment` loads `executionPatches = bundle.patchCount = 541`).
- **Inference Execution:**
  - `executionPatches` = 541.
  - `visionData.size` = $541 \times 1024 = 554,024$ floats.
  - Protocol 1 check: $541 \le 1000$ (PASS).
  - Buffer size check: $554,024 == 541 \times 1024$ (PASS).
  - ONNX Vision Tensor Shape: **`[1, 541, 1024]`**.
- **Outcome:** **541 patches enter inference.** The manual selector value of 100 has **zero effect**.

### Example 2: 541 available patches and manual selector value set to 500
- **Import:** The `.crcbundle` contains $N = 541$.
- **User Action:** The user taps the `500` button (`btnPatches500`). `selectedPatchCount` in memory becomes `500`.
- **Inference Execution:**
  - Same execution path as Example 1. `executionPatches` is derived directly from `bundle.patchCount = 541`.
  - Protocol 1 check: $541 \le 1000$ (PASS).
  - ONNX Vision Tensor Shape: **`[1, 541, 1024]`**.
- **Outcome:** **541 patches enter inference.** The manual selector value of 500 has **zero effect**.

### Example 3: 2,296 available patches and manual selector value set to 500
- **Import:** The `.crcbundle` contains $N = 2,296$.
- **User Action:** The user taps the `500` button (`btnPatches500`).
- **Inference Execution:**
  - `executionPatches` is derived from `bundle.patchCount = 2296`. The selector value of 500 is completely ignored.
  - **If Protocol 1 is selected:** `InputValidator.validateVision` checks `numPatches <= 1000`. Because $2,296 > 1,000$, it throws `IllegalArgumentException: Protocol 1 enforces a maximum cap of 1000 patches, but received 2296`. Inference fails with an error card displayed on the screen. **0 patches enter the model.** It does *not* truncate to 500 or 1,000 patches.
  - **If Protocol 2 is selected:** `InputValidator.validateVision` imposes no upper cap. All 2,296 patches pass validation. ONNX Vision Tensor Shape: **`[1, 2296, 1024]`**.
- **Outcome:** The selector value of 500 has **zero effect**. Inference either **fails closed** (Protocol 1) or evaluates **all 2,296 patches** (Protocol 2).

### Example 4: 2,296 available patches under Protocol 1
- **Inference Configuration:** Protocol is set to `PROTOCOL_1_CAPPED`.
- **Execution Path:**
  - `executionPatches` = 2,296.
  - `visionData.size` = $2,296 \times 1024 = 2,351,104$ floats.
  - `InputValidator.validateVision(visionData, 2296, PROTOCOL_1_CAPPED)` is invoked ([`InferenceExecutionFragment.kt:L313`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt#L313)).
  - Line 38 executes: `require(numPatches <= PROTOCOL1_MAX_PATCHES)`.
  - Condition fails ($2,296 > 1,000$).
  - Throws `IllegalArgumentException("Protocol 1 enforces a maximum cap of 1000 patches, but received 2296")`.
  - Caught at [`InferenceExecutionFragment.kt:L433`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt#L433). Failure recorded in SQLite (`CaseStatus.FAILED`).
  - Error card displayed: `"Inference execution failure: Protocol 1 enforces a maximum cap of 1000 patches, but received 2296"`.
- **Outcome:** **Inference is rejected fail-closed.** No tensor is created, and no patches enter the model. Protocol 1 does not dynamically truncate on-device.

### Example 5: 2,296 available patches under Protocol 2
- **Inference Configuration:** Protocol is set to `PROTOCOL_2_UNCAPPED`.
- **Execution Path:**
  - `executionPatches` = 2,296.
  - `visionData.size` = $2,351,104$ floats (9.40 MB direct memory).
  - `InputValidator.validateVision(visionData, 2296, PROTOCOL_2_UNCAPPED)` is invoked.
  - Protocol 2 uncapped branch executes ([`InputValidator.kt:L42-L45`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/inference/InputValidator.kt#L42-L45)): No upper bound enforced.
  - Buffer length check passes: $2,351,104 == 2,296 \times 1024$.
  - Finiteness check passes (no NaNs or Infs).
  - `CrcMultimodalClassifier.predictAuthentic` allocates a 9.40 MB direct `ByteBuffer`.
  - Vision Tensor Shape: **`[1, 2296, 1024]`**.
  - ONNX runtime executes cross-attention across all 2,296 patches and the text summary.
  - Returns classification logit, MSI probability, and full 2,296-element attention vector.
- **Outcome:** **All 2,296 patches enter inference.** 100% of available tissue evidence is evaluated.

---

## 7. Classification of the Selector

Under rigorous static code analysis, the manual Slide Patch Count selector is classified as:

### **`DISPLAY_ONLY` / `IGNORED`**

**Formal Justification:**
1. **`DISPLAY_ONLY` when syncing:** When an authentic bundle is ingested, `NewSpecimenFragment.kt` (lines 423–431) attempts to reflect `bundle.patchCount` in the UI by checking the matching toggle button. If the count does not match one of the discrete buttons (e.g., 541, 447, 712), it calls `clearChecked()`.
2. **`IGNORED` and `OVERRIDDEN` when clicked:** If a user manually taps any button (`10`, `50`, `100`, `500`, `1,000`), the stored value in `selectedPatchCount` is overwritten during `confirmImportAndSave` (lines 506–507) with `bundle.patchCount`.
3. **Completely Disconnected from Inference:** In `InferenceExecutionFragment.kt` (line 306), the inference runner loads the `.crcbundle` from storage and explicitly sets `executionPatches = bundle.patchCount`. The database field `ResearchCase.patchCount` is completely bypassed during model execution.
4. **Not `ACTIVE`:** The selector never alters tensor dimensions, tensor values, or ONNX execution.
5. **Not `CONDITIONAL`:** There is no conditional logic in any protocol that activates this manual selector.

---

## 8. Redundancy and Safe Removal Analysis

### Audit Answers to Key Architectural Questions

1. **Is the selector required for the existing pipeline?**  
   **No.** The pipeline requires an authentic `.crcbundle` (or a frozen holdout benchmark asset). The patch count is an intrinsic property of the authentic tensor bundle.
2. **Does it perform a function different from Protocol 1 and Protocol 2?**  
   **No.** It performs no functional role. Protocol selection occurs on the `InferenceExecutionFragment` screen (`toggleGroupProtocol`) or via automated routing in `BatchCoordinator.kt`.
3. **Could removing it change the model input or prediction?**  
   **No.** Model inputs are derived solely from `bundle.visionData` and `bundle.patchCount`. Predictions will remain numerically bit-for-bit identical.
4. **Would removing it break any saved settings, tests, or other screens?**  
   **No.** No unit tests or UI tests reference `toggleGroupNewPatches` or `btnPatches*`. Other screens (`SpecimenReviewFragment`, `ReportFragment`) read `patchCount` from the database or `InferenceRun`, which is populated from the bundle.
5. **Is it safe to replace it with a read-only display of the actual available patch count?**  
   **Yes. In fact, it is strongly recommended.** The current toggle buttons are actively misleading to clinical researchers, as they imply the user can choose to evaluate 10, 50, 100, 500, or 1,000 patches, when in reality all patches in the bundle are evaluated.
6. **What is the smallest safe change if removal is justified?**  
   - In [`fragment_new_specimen.xml`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/res/layout/fragment_new_specimen.xml): Replace the `MaterialButtonToggleGroup` (`toggleGroupNewPatches`) with a read-only information card or badge (`tvDetectedPatchCount`, e.g., displaying `"541 Patches Detected from Bundle"`).
   - In [`NewSpecimenFragment.kt`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/NewSpecimenFragment.kt): Remove the toggle group listener and button references; bind `bundle.patchCount` directly to the read-only display.

---

## 9. Remaining Uncertainties and Runtime Evidence Needed

1. **Legacy Draft Cases without Bundles:**
   - If a user uses `btnSaveDraft` to manually create a case without importing a `.crcbundle`, `ResearchCase.patchCount` is saved as the selector value (default: 10).
   - However, attempting to run inference on such a case immediately fails closed at [`InferenceExecutionFragment.kt:L188-L201`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/android/app/src/main/java/org/crc/multimodal/ui/fragments/InferenceExecutionFragment.kt#L188-L201) with `"AUTHENTIC FEATURE EMBEDDINGS REQUIRED"`. The "Run Inference" button is disabled. If forced, line 303 throws an `IllegalStateException`.
   - **Runtime Confirmation:** Verified by static code trace that synthetic vectors are forbidden.
2. **Upstream Protocol 1 Subsampling Determinism:**
   - Protocol 1's 1,000-patch capping was performed during dataset export in Python (`torch.randperm(N)[:1000]` with `seed=42`).
   - The Android app receives either a pre-capped tensor (for holdout cases) or a raw authentic tensor (for exported `.crcbundle` files).
   - If a clinical user imports a raw `.crcbundle` with $N > 1000$ and desires Protocol 1 compliance, the Android app currently **fails closed** rather than performing on-device sampling. This is deliberate design: on-device unseeded sampling would introduce non-reproducible clinical predictions.

---

## Summary Verdict

| Question | Audit Finding |
| :--- | :--- |
| **Does manual selector determine patches entering inference?** | **NO** |
| **Does it select or truncate patches from imported `.crcbundle`?** | **NO** |
| **Does it change vision tensor shape `[1, N, 1024]`?** | **NO** |
| **Does Protocol 1 independently cap patches at 1,000?** | **YES** (enforces ceiling $\le 1000$, rejects $>1000$) |
| **Does Protocol 2 independently evaluate all patches?** | **YES** (evaluates all $N$ patches uncapped) |
| **Does manual selector override either protocol?** | **NO** (it is completely overridden and ignored) |
| **Is the manual selector redundant?** | **YES** (safe to replace with read-only badge) |
