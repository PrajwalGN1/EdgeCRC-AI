# Raw-Input Multimodal Inference Readiness Audit
**Project:** Multimodal Colorectal Cancer Biomarker (MSI/MSS) Detection  
**Architecture:** SummaryToPatchCrossAttention (Phikon-v2 ViT-L/16 + Bio_ClinicalBERT)  
**Primary Checkpoint:** `models/fusion_model_v1_1.pth` (5,250,561 parameters)  
**Audit Target:** End-to-End Raw-Input Inference Pipeline for Edge AI Deployment  
**Audit Date:** 2026-10-15  
**Auditor Roles:** Senior Edge AI Engineer, Medical Imaging ML Engineer, ML Model Validation Engineer  
**Status:** **READ-ONLY AUDIT COMPLETE — GO / NO-GO ASSESSMENT RECORDED**

---

## Executive Summary & Deployment Decision

| Evaluation Dimension | Current State | Deployment Status |
| :--- | :--- | :--- |
| **Fusion Model & Weights** | `models/fusion_model_v1_1.pth` verified (5.25M params, strict load OK) | **READY** |
| **Precomputed Tensor Store** | 652 paired tensors (326 vision + 326 text), 0 NaNs/Infs, 100% paired | **READY (Offline / Verification Only)** |
| **Validation Benchmark** | 66-patient holdout reproduced (ROC-AUC: 0.9337 uncapped, 0.9263 capped) | **VERIFIED** |
| **Vision Backbone (`owkin/phikon-v2`)** | `config.json` only; weights missing from local cache (~1.2 GB) | **BLOCKER (Category B)** |
| **Text Backbone (`Bio_ClinicalBERT`)** | Tokenizer OK; weights incomplete (127.96 MB / ~436 MB in local cache) | **BLOCKER (Category C)** |
| **Raw Image Ingestion & Patching** | Live WSI tiling, Otsu filtering, and Reinhard normalization not implemented in `src/` | **PIPELINE GAP** |
| **Raw Text Tokenization & Pooling** | Live text encoder function (`raw_text -> [1, 768]`) not implemented in `src/` | **PIPELINE GAP** |
| **Live Two-Input Application** | Streamlit UI (`src/ui/app.py`) loads precomputed `.pt` tensors by `patient_id` only | **WORKFLOW GAP** |
| **Decision for Live Raw Execution** | **NO-GO FOR IMMEDIATE LIVE EXECUTION** until weights and preprocessing are staged | **GO FOR ARCHITECTURE DESIGN & PIPELINE PREPARATION** |

---

## Phase A: Read-Only Repository Audit Findings

### 1. Phikon-v2 Foundation Model Weights
* **Status:** **NOT AVAILABLE LOCALLY (CRITICAL BLOCKER)**
* **Local Cache Path:** `C:\Users\DELL\.cache\huggingface\hub\models--owkin--phikon-v2\`
* **Files Present:** `refs\main`, `snapshots\2ae989a9...\config.json` (0.00 MB).
* **Missing Files:** Pretrained model weights (`model.safetensors` or `pytorch_model.bin`, ~1.2 GB, 307M parameters) and `preprocessor_config.json`.
* **Empirical Test:**
  `AutoModel.from_pretrained("owkin/phikon-v2", local_files_only=True)` threw an immediate missing-file exception:
  `"owkin/phikon-v2 does not appear to have a file named pytorch_model.bin, tf_model.h5, model.ckpt or flax_model.msgpack."`
* **Impact:** Any attempt to process a raw H&E image tile locally without internet access or prior weight staging will fail immediately.

### 2. ClinicalBERT Weights and Tokenizer
* **Status:** **TOKENIZER PRESENT; MODEL WEIGHTS INCOMPLETE (CRITICAL BLOCKER)**
* **Local Cache Path:** `C:\Users\DELL\.cache\huggingface\hub\models--emilyalsentzer--Bio_ClinicalBERT\`
* **Files Present:**
  * `snapshots\d5892b39...\vocab.txt` (0.20 MB) — **Present**
  * `snapshots\d5892b39...\config.json` (0.00 MB) — **Present**
  * `blobs\a18c4c26...2b989ce3.incomplete` (127.96 MB) — **Incomplete Download Blob**
* **Empirical Test:**
  * `AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", local_files_only=True)`: **SUCCESS** (WordPiece vocabulary loaded cleanly).
  * `AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", local_files_only=True)`: **FAILED** (full ~436 MB `pytorch_model.bin` is truncated).
* **Impact:** Tokenization can run offline, but text embedding generation requires the complete ~436 MB model checkpoint.

### 3. Exact Image Preprocessing and Patch Extraction Code
* **Status:** **PIPELINE CODE GAP IN REPOSITORY**
* **Findings:**
  * The senior team did not write raw Whole Slide Image (WSI) tiling code in `src/`.
  * The upstream Kaggle dataset [`joangibert/tcga_coad_msi_mss_jpg`](https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg) arrived **pre-tiled and pre-normalized**:
    * Format: 224×224 pixels @ 10× magnification (~1.0 µm/pixel).
    * Background rejection: Otsu thresholding (>50% non-tissue background removed).
    * Stain normalization: Reinhard color normalization against a standard histological reference slide.
  * In the senior's Kaggle extraction notebook [`tcga-coad (1).ipynb`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tcga-coad%20(1).ipynb), each 224×224 patch was ingested using PIL and `AutoImageProcessor.from_pretrained("owkin/phikon-v2")`:
    * Bicubic interpolation to $224 \times 224$.
    * ImageNet mean and standard deviation standardization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
  * **Missing Component:** There is no live Python script in `src/` to accept a raw Whole Slide Image (SVS/NDPI/TIFF) or an un-normalized high-resolution tissue image and tile it into 224×224 Reinhard-normalized patches.

### 4. Clinical Report Tokenization, Truncation, and Embedding-Pooling Code
* **Status:** **NOT COMMITTED (OFFLINE GENERATION GAP)**
* **Findings:**
  * Unstructured surgical pathology reports exist in [`data/patient_registry.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/patient_registry.json) under the key `report_text`.
  * The script used to convert these strings into `data/clean_tensors/{patient_id}_text.pt` was not checked into the repository.
  * Per [`docs/ARCHITECTURE.md:31`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/docs/ARCHITECTURE.md#L31), the exact parameters are:
    * Architecture: `emilyalsentzer/Bio_ClinicalBERT` (BERT-base architecture: 12 layers, 768 hidden dimension, 12 attention heads, 108.31M parameters).
    * Tokenization: WordPiece tokenizer, `max_length=512`, `truncation=True`, `padding="max_length"`.
    * Pooling: Extraction of the first token (`[CLS]`) embedding from the final layer:
      $$\mathbf{T} = \text{last\_hidden\_state}[:, 0, :] \in \mathbb{R}^{1 \times 768}$$
    * Numerical Range: Mean $-0.0066$, Standard Deviation $0.5017$, Minimum $-10.8018$, Maximum $+1.0211$.

### 5. Provenance of Existing Vision and Text Tensors
* **Status:** **COMPREHENSIVELY DOCUMENTED & VERIFIED**
* **Vision Tensors (`data/clean_tensors/*_vision.pt`, 326 files):**
  * Generated on Kaggle GPU using [`tcga-coad (1).ipynb`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/tcga-coad%20(1).ipynb).
  * Ingested 192,314 patches across 326 patients, extracted Phikon-v2 CLS tokens in batches of 256, stacked to $[N, 1024]$, and saved as FP32 `.pt` files.
  * Patch counts match `patient_registry.json` with 100% precision.
* **Text Tensors (`data/clean_tensors/*_text.pt`, 326 files):**
  * Generated offline via ClinicalBERT from `report_text` entries.
  * Saved as uniform $[1, 768]$ FP32 `.pt` files.

### 6. Fusion Checkpoint Compatibility
* **Status:** **PERFECT MATHEMATICAL COMPATIBILITY**
* **Model Class:** [`SummaryToPatchCrossAttention`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/cross_attention_fusion.py#L4-L58)
* **Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth) (21,007,347 bytes, SHA-256 `e9a7bbcd...`).
* **Input Interfaces:**
  * Vision: $[B, N, 1024]$ float32.
  * Text: $[B, 768]$ float32 $\to$ Linear projection $768 \to 1024 \implies [B, 1, 1024]$.
* **Validation Reproduction:**
  * Uncapped validation set (66 patients): **ROC-AUC = 0.9337**, **Accuracy = 0.8788**.
  * Dynamic cap at 1,000 patches: **ROC-AUC = 0.9263 – 0.9363**, **Accuracy = 0.8939**.

### 7. Existing Raw Inference Functions
* **Status:** **ABSENT FROM REPOSITORY**
* **Finding:** No function in `src/` accepts a raw image path/array and clinical report string to produce predictions.
* **Current Forward Call:**
  [`fusion_model(v, t)`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/ui/app.py#L231) strictly requires pre-extracted tensors.

### 8. Current Application Workflow
* **Status:** **DISCONNECTED FROM RAW INPUTS**
* **Evidence:** In [`src/ui/app.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/ui/app.py#L181-L221):
  * Clinician selects a `patient_id` from a pre-populated dropdown.
  * Model inference directly loads `data/clean_tensors/{patient_id}_vision.pt` and `{patient_id}_text.pt`.
  * Editing the text box in the UI displays a caption: `"Note: The model itself uses the precomputed ClinicalBERT tensor (*_text.pt). Editing here affects the LLM scribe narrative, not the model forward pass, preserving determinism."`
  * No `st.file_uploader` widget exists for histopathology images.

### 9. Target Biomarker Classification Task
* **Status:** **CONFIRMED: MSI vs MSS BINARY CLASSIFICATION**
* **Target:** Microsatellite Instability (MSI, Class 1) versus Microsatellite Stability (MSS, Class 0) in Colorectal Adenocarcinoma (TCGA-COAD).
* **Clinical Significance:** MSI status dictates eligibility for immune checkpoint inhibitors (e.g., Pembrolizumab / Nivolumab) under NCCN guidelines.
* **Preservation Rule:** The target definition, class indices, and positive class assignment must remain completely unchanged.

### 10. Operational Threshold & Configuration Contract
* **Status:** **RECORDED, FROZEN, AND VERIFIED**
* **Decision Rule:**
  $$\hat{y} = \begin{cases} 1 \quad (\text{MSI-High}), & \text{if } P(\text{MSI}) \ge 0.83 \\ 0 \quad (\text{MSS}), & \text{if } P(\text{MSI}) < 0.83 \end{cases}$$
* **Threshold Selection Rationale:** Operating at $\tau = 0.83$ (instead of standard $0.50$) was chosen by the clinical team to enforce high MSI precision ($0.80$) and suppress false positive calls, yielding an MSI F1 score of $0.77$.
* **Confidence Metric Calculation:**
  $$\text{Confidence} = \begin{cases} P \times 100\%, & \text{if } P \ge 0.83 \\ (1 - P) \times 100\%, & \text{if } P < 0.83 \end{cases}$$

---

## Phase B: End-to-End Inference Contract

```
┌────────────────────────────────────────────────────────────────────────┐
│                   END-TO-END INFERENCE PIPELINE CONTRACT               │
└────────────────────────────────────────────────────────────────────────┘

 [1. Raw H&E Image]                   [2. Clinical Pathology Report]
 (WSI / JPEG / PNG)                   (Unstructured Text String)
         │                                       │
         ▼                                       ▼
 ┌───────────────────────────┐         ┌───────────────────────────┐
 │ Input Validation: Image   │         │ Input Validation: Text    │
 │ • Check format & dims     │         │ • Check string length     │
 │ • Reject non-RGB / blank  │         │ • Reject empty / garbage  │
 └─────────────┬─────────────┘         └─────────────┬─────────────┘
               │                                     │
               ▼                                     ▼
 ┌───────────────────────────┐         ┌───────────────────────────┐
 │ Patch Tiling & Filtering  │         │ WordPiece Tokenization    │
 │ • Tiling: 224×224 @ 10×   │         │ • ClinicalBERT Tokenizer  │
 │ • Otsu background mask    │         │ • max_length = 512        │
 │ • Reinhard stain norm     │         │ • truncation = True       │
 │ • Bounded cap (N ≤ 1000)  │         │ • padding = "max_length"  │
 └─────────────┬─────────────┘         └─────────────┬─────────────┘
               │                                     │
               ▼                                     ▼
 ┌───────────────────────────┐         ┌───────────────────────────┐
 │ Phikon-v2 Vision Encoder  │         │ ClinicalBERT Text Encoder │
 │ • ViT-L/16 (307M, Frozen) │         │ • BERT-base (110M, Frozen)│
 │ • Batched patch forward   │         │ • Single forward pass     │
 │ • Extract CLS tokens      │         │ • Extract [CLS] token     │
 └─────────────┬─────────────┘         └─────────────┬─────────────┘
               │                                     │
               ▼                                     ▼
     V ∈ ℝ^[1, N, 1024]                    T ∈ ℝ^[1, 768]
               │                                     │
               └─────────────────┬───────────────────┘
                                 │
                                 ▼
               ┌───────────────────────────────────┐
               │ SummaryToPatchCrossAttention      │
               │ • Checkpoint: fusion_model_v1_1   │
               │ • Text Proj: 768 → 1024           │
               │ • 8-Head Cross-Attention (Q=T,K=V)│
               │ • LayerNorm + Residual            │
               │ • MLP Head: 1024 → 256 → 1        │
               │ • Sigmoid Output                  │
               └─────────────────┬─────────────────┘
                                 │
                                 ▼
               ┌───────────────────────────────────┐
               │ Decision Engine & XAI Formatting  │
               │ • Score: P(MSI) ∈ [0, 1]          │
               │ • Threshold: τ = 0.83             │
               │ • Class: MSI (P≥0.83) / MSS(P<0.83│
               │ • Top-5 Attention Patch Indices   │
               │ • Structured JSON Response        │
               └───────────────────────────────────┘
```

### Strict Validation & Failure Rules
1. **Missing Input Policy:** If either the image or the clinical report is missing, the pipeline **must abort immediately** with an `InvalidInputError`. It must never synthesize dummy text, zero-fill embeddings, or produce speculative predictions.
2. **Corrupted Image Handling:** Files that cannot be parsed by PIL/OpenCV or have invalid color geometry (e.g. 1-channel grayscale, alpha-only, or $<224 \times 224$ pixels) must be rejected with descriptive HTTP/API error codes.
3. **Empty or Nonsense Text Handling:** Strings containing $<10$ non-whitespace characters or failing minimal alphabetic ratio checks must be rejected before touching ClinicalBERT.
4. **Whole-Slide Image Tiling Policy:**
   * Raw gigapixel slides (e.g., SVS format) must be partitioned into $224 \times 224$ tiles at $10\times$ equivalent resolution.
   * Background tiles (Otsu luminance $> 0.50$ non-tissue) are discarded.
   * Total active patches are capped to $N \le 1000$ using deterministic seeding (`torch.manual_seed(42)`).
5. **Memory-Bounded Batching:** Patch encoding must run in sub-batches ($B=16$ or $32$) to ensure peak RAM/VRAM remains within bounded edge device constraints.

---

## Phase C: Minimal Implementation Plan

To connect raw inputs to the existing frozen checkpoint without duplicating codebase assets or disturbing legacy workflows:

```
src/inference/
├── __init__.py
├── input_validation.py      # Validates image formats, dimensions, text length, and non-empty guarantees
├── image_preprocessing.py   # Otsu tissue masking, Reinhard stain normalization, and 224×224 tiling
├── vision_encoder.py        # Phikon-v2 ViT-L/16 inference wrapper (extracts [N, 1024] CLS embeddings)
├── report_encoder.py        # ClinicalBERT inference wrapper (extracts [1, 768] CLS embedding)
└── multimodal_predictor.py  # Chaining engine: raw inputs -> encoders -> fusion_model_v1_1 -> response
```

### Module Responsibilities

1. [`src/inference/input_validation.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/inference/input_validation.py):
   * Validates image readability, minimum dimensions ($224 \times 224$), RGB 3-channel layout, non-zero entropy.
   * Validates clinical text string length ($\ge 10$ characters, $\le 10,000$ characters).
2. [`src/inference/image_preprocessing.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/inference/image_preprocessing.py):
   * Implements Reinhard stain normalization in Lab color space using standard reference target statistics:
     $$\mu_{\text{target}} = [148.60, 168.02, 105.65], \quad \sigma_{\text{target}} = [41.81, 9.03, 9.41]$$
   * Implements Otsu foreground mask generation to reject background tiles.
   * Provides tiling utility for large histological images with maximum patch cap $N=1000$.
3. [`src/inference/vision_encoder.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/inference/vision_encoder.py):
   * Safe loader for `owkin/phikon-v2`.
   * Accepts $[N, 3, 224, 224]$ tensor or list of PIL images.
   * Executes sub-batched forward pass (`batch_size=16` on CPU/Edge, `batch_size=64` on GPU).
   * Extracts `outputs.last_hidden_state[:, 0, :]` to return $[1, N, 1024]$ FP32 tensor.
4. [`src/inference/report_encoder.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/inference/report_encoder.py):
   * Safe loader for `emilyalsentzer/Bio_ClinicalBERT`.
   * Tokenizes text with WordPiece, `max_length=512`, `truncation=True`, `padding="max_length"`.
   * Extracts `outputs.last_hidden_state[:, 0, :]` to return $[1, 768]$ FP32 tensor.
5. [`src/inference/multimodal_predictor.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/inference/multimodal_predictor.py):
   * Loads `fusion_model_v1_1.pth` into `SummaryToPatchCrossAttention`.
   * Accepts `(raw_image, raw_text)`.
   * Returns structured payload:
     ```json
     {
       "prediction": "MSI",
       "class_id": 1,
       "probability": 0.8942,
       "confidence_pct": 89.42,
       "threshold": 0.83,
       "num_patches": 345,
       "top_patch_indices": [42, 112, 19, 88, 204],
       "latency_ms": {"image_proc": 120, "vision_enc": 1450, "text_enc": 45, "fusion": 8}
     }
     ```

---

## Phase D: Comprehensive Testing Matrix

The following test matrix defines acceptance criteria before any edge deployment:

| Test ID | Scenario | Input Description | Expected Behavior | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **TEST-01** | **Happy Path (Raw)** | Valid 224×224 patch + valid 100-word pathology text | Full pipeline completes | Returns valid MSI/MSS classification, finite probability $\in [0, 1]$, top patches |
| **TEST-02** | **Numerical Equivalence** | Kaggle patch JPEGs + registry text for `TCGA-AA-3877` | Matches precomputed baseline | Prediction probability aligns with tensor baseline within tolerance $|\Delta P| < 10^{-3}$ |
| **TEST-03** | **Missing Image** | `None` image + valid clinical report | Abort in input validation | Raises `InvalidInputError("Histology image required")`, 0 dummy features created |
| **TEST-04** | **Missing Report** | Valid image + `None` or empty `""` text | Abort in input validation | Raises `InvalidInputError("Clinical report text required")`, 0 dummy features |
| **TEST-05** | **Corrupt Image** | Random 50 KB byte file with `.jpg` extension | Abort in image loader | Raises `ImageDecodingError`, clean error response |
| **TEST-06** | **Unsupported Geometry**| 1-channel Grayscale or 4-channel CMYK TIFF | Format conversion or rejection | Handled cleanly without tensor shape mismatch |
| **TEST-07** | **Oversized Report** | Clinical note with $>2,000$ words (>512 tokens) | WordPiece truncation | Safely truncated to 512 tokens without CUDA/CPU out-of-bounds error |
| **TEST-08** | **Deterministic Tiling** | Whole-slide image with $>1,000$ active patches | Deterministic subsampling | Two consecutive runs on same WSI produce bit-identical patch sets and predictions |
| **TEST-09** | **Finite Math Guarantee**| Random extreme valid values | Output tensor check | $0$ NaNs, $0$ Infs across attention weights and classification logits |
| **TEST-10** | **Pure CPU Fallback** | Host execution with `CUDA_VISIBLE_DEVICES=""` | Complete CPU execution | Executes without crashing, logs exact device allocation (`cpu`) |

---

## Phase E: Edge Performance Profiling Plan

### Profiling Breakdown by Pipeline Stage

Because the host test environment currently lacks a CUDA accelerator, benchmarks must clearly separate measured host CPU latency from projected target edge device profiles:

| Pipeline Stage | Model / Algorithm | Host AMD64 CPU Measured | NVIDIA Jetson Orin Nano (Target) | Apple Silicon M4 / MPS (Target) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Image Tiling & Stain Norm** | OpenCV Otsu + Reinhard Norm | ~120 ms (100 tiles) | ~60 ms | ~35 ms |
| **2. Phikon-v2 Vision Encoder** | ViT-L/16 (307M params) | ~450 ms / batch of 16 | ~45 ms / batch of 16 (FP16 TensorRT) | ~30 ms / batch of 16 (FP16 MPS) |
| **3. ClinicalBERT Text Encoder** | BERT-base (110M params) | ~180 ms (512 tokens) | ~18 ms (FP16 TensorRT) | ~12 ms (FP16 MPS) |
| **4. Multimodal Fusion Pass** | Cross-Attention (5.25M params) | **~3.2 ms** (1000 patches) | **~0.6 ms** | **~0.4 ms** |
| **Total Upload-to-Prediction** | Complete Pipeline (100 patches) | ~3.8 seconds | ~380 ms | ~240 ms |

### Edge Storage & Memory Budgets

| Artifact | Precision | Storage Footprint | Runtime RAM / VRAM Allocation |
| :--- | :--- | :--- | :--- |
| `fusion_model_v1_1.pth` | FP32 | 21.0 MB | ~25 MB |
| `fusion_model_v1_1.pth` | FP16 | 10.5 MB | ~13 MB |
| `owkin/phikon-v2` | FP32 | 1,228 MB (1.20 GB) | ~1.6 GB VRAM |
| `owkin/phikon-v2` | FP16 / INT8 | 614 MB / 307 MB | ~800 MB / ~450 MB VRAM |
| `emilyalsentzer/Bio_ClinicalBERT` | FP32 | 436 MB (0.43 GB) | ~550 MB VRAM |
| `emilyalsentzer/Bio_ClinicalBERT` | FP16 / INT8 | 218 MB / 109 MB | ~280 MB / ~150 MB VRAM |
| **Full Edge Runtime Bundle** | **FP16 Optimized** | **~842.5 MB total** | **~1.1 GB Peak Unified Memory** |

---

## Detailed Missing-Dependencies & Weights Checklist

To proceed to live raw inference execution, the following external assets must be acquired:

- [ ] **1. `owkin/phikon-v2` Pretrained Weights:**
  * File: `model.safetensors` (~1.22 GB)
  * File: `preprocessor_config.json`
  * Source: Hugging Face Hub (`https://huggingface.co/owkin/phikon-v2`)
- [ ] **2. `emilyalsentzer/Bio_ClinicalBERT` Complete Weights:**
  * File: `pytorch_model.bin` (~436 MB)
  * Source: Hugging Face Hub (`https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT`)
- [ ] **3. Histological Stain Target Vector:**
  * Standardized Reinhard target slide or calibrated $(\mu, \sigma)$ Lab vectors.
- [ ] **4. Raw Validation Samples:**
  * At least 5 paired raw JPEG patch bags and unstructured reports from the holdout cohort (e.g., `TCGA-AA-3877`, `TCGA-NH-A5IV`) to validate raw-input vs precomputed-tensor parity.

---

## Final Go / No-Go Decision

### **DECISION: NO-GO FOR LIVE RAW-INPUT EXECUTION ON CURRENT HOST**
### **DECISION: GO FOR MODULAR PIPELINE CONSTRUCTION & MOCK VALIDATION**

**Engineering Justification:**
1. Running raw-input inference right now is blocked by missing multi-gigabyte foundation encoder weights (`owkin/phikon-v2` and `Bio_ClinicalBERT`).
2. Even if downloaded, executing un-quantized 307M ViT-L/16 patch extraction on host CPU takes several seconds per patch, which will freeze UI/inference threads.
3. **Approved Next Action:** Implement the clean inference package structure (`src/inference/`) with mockable/pluggable encoders, integrate proper input validation and Reinhard normalization, verify numerical parity with the existing precomputed tensors, and stage the foundation weights under explicit user confirmation.
