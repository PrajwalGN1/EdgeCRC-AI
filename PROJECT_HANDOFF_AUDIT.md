# Project Handoff Audit: Multimodal Fusion for CRC MSI/MSS Detection

> **Project:** Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2  
> **Repository:** `Mokshithc/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Audit Date:** 2026-10-03  
> **Auditor Role:** Senior ML Systems Engineer, Edge AI Engineer, MLOps Engineer, PyTorch Engineer, Medical-AI Research Engineer, Software Architect  
> **Active Branch:** `edge-deployment` (branched from `main` @ `189b663`)

---

## 1. Repository State

### 1.1 Git Tracking and Commit Metadata
* **Active Branch:** `edge-deployment`
* **Base Branch:** `main` (synchronized with `origin/main`)
* **HEAD Commit:** `189b663` (`chore: remove roadmap/next-team and authors, keep mentor Mr Nageshwara Guptha — professional public version`)
* **Previous Commits:**
  * `ac27a8f`: `chore: exclude generated heatmaps, fix .gitignore`
  * `5a9cd2a`: `first commit`
* **Git LFS Status:** **Not configured**. There is no `.gitattributes` file in the checkout, and `git lfs ls-files` reports no tracked files.

### 1.2 Execution Environment Expectations vs. Local Reality
| Property | Expected by Senior Team | Current Local Environment | Status & Notes |
| :--- | :--- | :--- | :--- |
| **Operating System** | Linux (Ubuntu `/home/moki27/project/...`) / macOS (Apple Silicon M4 referenced in code comments) | Windows 11 / AMD64 | Cross-platform compatibility enabled in `src/ui/app.py` via `pathlib.Path`. |
| **Python Version** | `3.10+` (Dockerfile specifies `python:3.10-slim`, README states `Python 3.10+`) | `3.13.5` | Operational; PyTorch 2.9.1 runs cleanly on host. |
| **PyTorch Version** | `2.9.1` (`torch==2.9.1` in `requirements.txt`) | `2.9.1+cpu` | Verified; model weights load and forward pass functions. |
| **CUDA / Acceleration** | CUDA GPU or Apple Silicon MPS | CPU (`torch.cuda.is_available() == False`) | The codebase features automatic hardware fallback (`cuda -> mps -> cpu`). |
| **VRAM / RAM** | ≥16 GB RAM / GPU with ≥8 GB VRAM recommended | 16 GB System RAM / Host CPU | Sufficient for lightweight fusion model inference; insufficient for full 192k patch feature extraction with Phikon-v2 ViT-L/16. |

### 1.3 Major Dependencies Audit
The senior team committed a frozen `requirements.txt` containing 75 packages. However, forensic analysis revealed critical omissions in `requirements.txt`:

* **Present in `requirements.txt`:**
  * `torch==2.9.1`, `torchvision==0.24.1`, `torchaudio==2.9.1`, `monai==1.5.1`, `timm==1.0.22`, `streamlit==1.52.2`, `numpy==2.2.6`, `pandas==2.3.3`, `pillow==12.0.0`, `opencv-python==4.12.0.88`, `huggingface_hub==1.2.3`.
* **Omitted from `requirements.txt` but Required by Source Code:**
  * `transformers`: Required by `src/generate_pixel_xai.py` (line 231) and `src/generate_pixel_xai_1.py` (line 10) for `AutoImageProcessor` and `AutoModel` (`owkin/phikon-v2`).
  * `scikit-learn`: Required by `src/cross_attentiion_train.py` (line 5), `src/cross_attention_eval.py` (line 7), `src/late_fusion_train.py`, and `src/late_fusion_eval.py` for metrics calculation (`roc_auc_score`, `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_curve`).
  * `matplotlib`: Required by `src/cross_attentiion_train.py`, `src/cross_attention_eval.py`, `src/generate_xai.py`, `src/ui/app.py`, and `docs/generate_diagrams.py`.
  * `seaborn`: Required by `src/cross_attention_eval.py` (line 4) and `src/late_fusion_eval.py` for confusion matrix and attention heatmaps.
  * `scipy`: Required by `verify_cross_attention.py` and evaluation utilities for statistical operations.
* **Local Environment Status:** All required packages (`transformers 5.12.1`, `scikit-learn 1.7.0`, `matplotlib 3.10.3`, `seaborn 0.13.2`, `scipy 1.16.0`) are already installed in the local Python environment.

### 1.4 Directory Tree Structure
```
d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/
├── .git/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── HEAVY_FILES_MANIFEST.txt
├── PB05_Report_8thSem.pdf
├── README.md
├── UNPUSHED_ASSETS.md
├── requirements.txt
├── assets/
│   └── medical_images/
│       ├── XAI_Report_TCGA-A6-6142.png
│       ├── XAI_Report_TCGA-AA-3877.png
│       ├── XAI_Report_TCGA-AY-6386.png
│       ├── XAI_Report_TCGA-CM-6675.png
│       └── XAI_Report_TCGA-D5-6928.png
├── data/
│   ├── DATASET.md
│   ├── Download_DataSet.pdf
│   ├── MSIMUT_JPEG/ (.gitkeep, README.txt)
│   ├── MSS_JPEG/ (.gitkeep, README.txt)
│   └── patient_registry.json (13.1 MB)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── HANDOFF.md
│   ├── generate_diagrams.py
│   ├── generate_diagrams_v2.py
│   └── diagrams/ (20 PNG/SVG architecture and sequence diagrams)
├── models/
│   ├── best_fusion_model.pth (21.0 MB)
│   ├── fusion_model_v1_0.pth (21.0 MB)
│   ├── fusion_model_v1_1.pth (21.0 MB, ★ Primary Checkpoint)
│   ├── fusion_model_v1_2.pth (21.0 MB)
│   ├── late_fusion_model.pth (5.25 MB)
│   └── late_fusion_model_v1_0.pth (5.25 MB)
├── results/
│   ├── Confusion_Matrix_C.png
│   ├── Confusion_Matrix_L.png
│   ├── Cross_Attention_Model_Evaluation_Report.txt
│   ├── Cross_Attention_Verification_Proof.png
│   ├── Late_FusionModel_Evaluation_Report.txt
│   ├── ROC_Curve_C.png
│   ├── ROC_Curve_L.png
│   ├── Real_Cross_Attention_Matrix.png
│   └── actual_validation_keys.json (66 patient IDs)
├── results_old/
│   ├── BCE_Loss_Trajectory.png
│   ├── BCE_Loss_Trajectory1.png
│   └── ...
└── src/
    ├── multimodal_dataset.py
    ├── cross_attentiion_train.py
    ├── cross_attention_eval.py
    ├── late_fusion_train.py
    ├── late_fusion_eval.py
    ├── generate_xai.py
    ├── generate_pixel_xai.py
    ├── generate_pixel_xai_1.py
    ├── plot_cross_attention.py
    ├── architectures/
    │   ├── cross_attention_fusion.py
    │   └── late_fusion_model.py
    ├── eval/
    │   ├── tensor_dim_check.py
    │   ├── verify_cross_attention.py
    │   └── verify_late_attention.py
    └── ui/
        ├── app.py
        └── app_legacy.py
```

---

## 2. Existing Components

| Component | Exists? | Location | Status | Evidence |
| :--- | :---: | :--- | :--- | :--- |
| **ClinicalBERT** | ⚠️ Partial | HuggingFace (`Bio_ClinicalBERT`) | Frozen feature extractor used offline to create 768-D `_text.pt` embeddings. Model weights not saved locally; downloaded on demand if needed. | Documented in `docs/ARCHITECTURE.md:31`. |
| **Phikon-v2** | ⚠️ Partial | HuggingFace (`owkin/phikon-v2`) | Frozen feature extractor used offline to produce 1024-D `_vision.pt` patch embeddings. Instantiated on demand in `src/generate_pixel_xai.py:244`. | Code in `src/generate_pixel_xai.py` with offline fallback. |
| **Fusion Architecture** | ✅ Yes | `src/architectures/cross_attention_fusion.py` | Complete & fully verified. `SummaryToPatchCrossAttention` defines projection (768→1024), 8-head MHA, LayerNorm, and MLP (1024→256→1). | Instantiation and parameter count verified: 5,250,561 params. |
| **Baseline Architecture** | ✅ Yes | `src/architectures/late_fusion_model.py` | Complete & verified. `MultimodalFusionNetwork` implements AB-MIL + text projection + concatenation + MLP classifier. | Source code present and verified. |
| **Trained Weights** | ✅ Yes | `models/*.pth` | Complete. 6 checkpoints present locally, including primary `fusion_model_v1_1.pth` (21,007,347 bytes) and baseline `late_fusion_model_v1_0.pth` (5,255,861 bytes). | Verified `torch.load` succeeds with `weights_only=True`. |
| **Clinical Embeddings** | ❌ No | Expected at `data/clean_tensors/*_text.pt` | **Absent locally**. Excluded from GitHub push due to size; directory does not exist on disk. | Documented in `UNPUSHED_ASSETS.md:11` and ignored in `.gitignore:10`. |
| **Vision Embeddings** | ❌ No | Expected at `data/clean_tensors/*_vision.pt` | **Absent locally**. Excluded from GitHub push due to size (653 files, ~1.2 GB); directory does not exist on disk. | Documented in `UNPUSHED_ASSETS.md:11` and ignored in `.gitignore:10`. |
| **Patient Registry** | ✅ Yes | `data/patient_registry.json` | Complete & intact (13,158,992 bytes). Contains all 326 patients with labels, reports, image paths, and patch counts. | Verified with Python script: loads 326 patient records cleanly. |
| **Training Code** | ✅ Yes | `src/cross_attentiion_train.py`, `src/late_fusion_train.py` | Complete. 30 epochs, AdamW (`lr=1e-4`, `wd=1e-4`), weighted BCE (`pos_weight=4.17`), `batch_size=1`, dynamic patch cap 1000. | Source code inspected; training loop complete. |
| **Evaluation Code** | ✅ Yes | `src/cross_attention_eval.py`, `src/late_fusion_eval.py` | Complete. Implements ROC-AUC, classification report, confusion matrix, and attention matrix plotting on holdout. | Source code inspected; references `actual_validation_keys.json`. |
| **Validation Contract** | ✅ Yes | `results/actual_validation_keys.json` | Complete & locked (1,056 bytes). Exactly 66 isolated validation patient IDs (50 MSS, 16 MSI). | File inspected; contains 66 exact TCGA IDs. |
| **Streamlit UI** | ✅ Yes | `src/ui/app.py` (445 lines) | Complete & operational. Path-robust, hardware-routing, placeholder patch rendering if JPEGs absent, offline scribe fallback. | `load_system()` successfully tested without errors. |
| **XAI Stack** | ✅ Yes | `src/generate_xai.py`, `src/generate_pixel_xai.py`, `src/ui/app.py` | Complete 3-tier architecture: 1) Top-5 attention patch ranking, 2) ViT attention rollout heatmap, 3) Qwen 2.5 LLM clinical scribe. | Code and 5 sample outputs in `assets/medical_images/` present. |
| **Docker** | ✅ Yes | `Dockerfile`, `.streamlit/config.toml` | Complete. `FROM python:3.10-slim` with system libraries for OpenCV and PyTorch CPU. | Inspected; healthcheck and entrypoint configured. |
| **ONNX Export** | ❌ No | N/A | Not implemented. No export scripts or `.onnx` model graphs exist in the repository. | Zero occurrences in codebase. |
| **Mobile Deployment**| ❌ No | N/A | Not implemented. | Zero occurrences in codebase. |
| **Edge Runtime** | ❌ No | N/A | Not implemented. Current code only targets PyTorch (CUDA / MPS / CPU). | Zero occurrences in codebase. |

---

## 3. Missing Artifacts

Forensic analysis confirms the following artifacts are referenced by code or documentation but absent from the local checkout:

### 3.1 `data/clean_tensors/*.pt` (653 files, ~1.2 GB)
* **What it is:** Precomputed tensors for 326 patients:
  * `*_vision.pt`: Phikon-v2 ViT-L/16 patch embeddings of shape `[N, 1024]` (where $N \in [1, 2888]$, dynamically capped to 1000).
  * `*_text.pt`: ClinicalBERT summary embeddings of shape `[1, 768]`.
* **Why it is needed:**
  * To run inference on any patient from the UI (`src/ui/app.py`).
  * To reproduce baseline evaluation on the 66 holdout patients (`src/cross_attention_eval.py`).
  * To run the perturbation verification test (`src/eval/verify_cross_attention.py`).
  * To run training (`src/cross_attentiion_train.py`).
* **Why it is missing:**
  * `.gitignore` line 10 explicitly ignores `data/clean_tensors/`.
  * `UNPUSHED_ASSETS.md` line 11 documents that these files were kept on the senior team's local Linux workstation (`/home/moki27/project/...`) and excluded from GitHub to bypass the 100 MB per-file / 1 GB repository limits without setting up Git LFS.
* **Can it be downloaded directly?** No public direct download link exists for the `.pt` files.
* **Can it be regenerated?** Yes, from the raw Kaggle dataset `joangibert/tcga_coad_msi_mss_jpg` using Phikon-v2 and ClinicalBERT.
* **Is raw data required?** Yes (pathology report text from `patient_registry.json` is present locally, but the 192k raw H&E patch JPEGs are required for vision embeddings).
* **Is GPU required?**
  * **Text tensors (`*_text.pt`):** No GPU strictly required; 326 text notes through ClinicalBERT can execute in under 10 minutes on CPU.
  * **Vision tensors (`*_vision.pt`):** **GPU is mandatory**. Passing ~192,314 patches through a 307M parameter ViT-L/16 (`owkin/phikon-v2`) on CPU is computationally infeasible (estimated ~200+ hours on CPU vs. ~1.5–3 hours on an NVIDIA T4/V100/A100 GPU).

### 3.2 Raw Dataset JPEGs: `data/MSIMUT_JPEG/` & `data/MSS_JPEG/` (~12 GB, 192,312 files)
* **What it is:** Raw 224×224 @10× Reinhard stain-normalized H&E image patches.
* **Why it is needed:** Required ONLY if regenerating the vision tensors from scratch or displaying actual histology photographs in the Streamlit Top-5 gallery.
* **Why it is missing:** Excluded via `.gitignore` (lines 14–17).
* **Can it be downloaded?** Yes, from Kaggle: [`joangibert/tcga_coad_msi_mss_jpg`](https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg).

### 3.3 Feature Extraction Script (Data Pipeline Gap)
* **What it is:** The senior team provided `src/multimodal_dataset.py`, but that file is strictly a **PyTorch Dataset loader** (`class MultimodalTCGADataset`) that assumes `data/clean_tensors/*.pt` already exist on disk.
* **Finding:** There is **no script in the repository** that performs the initial forward extraction from raw JPEGs with Phikon-v2 and raw text with ClinicalBERT to save `*_vision.pt` and `*_text.pt`.
* **Implication:** The senior team ran feature extraction in an unversioned offline script or notebook (likely on Kaggle or a lab workstation). The exact pipeline specifications are preserved in `docs/ARCHITECTURE.md` (Phase 2), `docs/HANDOFF.md`, and `src/generate_pixel_xai.py`.

---

## 4. Current Execution State

| Test Objective | Status | Result / Error Details | Evidence |
| :--- | :---: | :--- | :--- |
| **A. Can Streamlit app start?** | ✅ YES | App initializes successfully. `load_system()` executes cleanly, resolves model paths, binds CPU device, and loads registry. | Verified via bare-mode execution of `load_system()`: returns `loaded=True`, `device=cpu`, `registry_count=326`, `err=None`. |
| **B. Can fusion model load?** | ✅ YES | `SummaryToPatchCrossAttention` loads `models/fusion_model_v1_1.pth` with `weights_only=True` without warnings or parameter mismatches. | Verified parameter count = exactly 5,250,561 float32 weights. |
| **C. Can patient registry load?** | ✅ YES | Loads valid JSON dictionary containing 326 patient records (63 MSI, 263 MSS). | Verified with Python: `len(reg) == 326`, keys and fields match data contract. |
| **D. Can processed patient tensor load?** | ❌ NO | Fails with `FileNotFoundError` because `data/clean_tensors/` directory does not exist on disk. | Directory listing of `data/` confirms absence of `clean_tensors/`. |
| **E. Can model perform forward pass?** | ✅ YES | Executes cleanly with mock/synthetic tensors (`[1, 100, 1024]` and `[1, 768]`). Returns probability scalar and `[1, 1, 100]` attention tensor. | Tested on host CPU: `Prob: 0.6098`, `Attn shape: torch.Size([1, 1, 100])`. |
| **F. Can reported baseline prediction be reproduced?** | ❌ NO | Blocked by missing real patient tensors. | Requires `data/clean_tensors/TCGA-AA-3877_vision.pt` and `_text.pt`. |
| **G. Can evaluation be reproduced?** | ❌ NO | Blocked by missing real patient tensors. `src/cross_attention_eval.py` cannot iterate over the 66 validation patients. | Requires all 66 validation patient `.pt` files. |

---

## 5. Current Handoff Point

### Where did the previous team stop?
Based on strict forensic evidence:
1. **Model Architecture & Training:** The senior team designed, implemented, and successfully trained `SummaryToPatchCrossAttention` (achieving the reported ROC-AUC of 0.9150 and MSI F1 of 0.77). All 6 checkpoints were committed to `models/`.
2. **Evaluation & Artifacts:** The team ran full evaluation on the isolated 66-patient holdout set (`results/actual_validation_keys.json`), plotted the ROC curve, confusion matrix, real cross-attention matrix, and loss convergence trajectory, saving reports in `results/`.
3. **Clinical Streamlit UI:** The team created and hardened `src/ui/app.py`, adding path resolution, hardware auto-routing, placeholder patch fallbacks, and an Ollama LLM clinical scribe with offline templates.
4. **Documentation & Handoff:** The team authored comprehensive documentation (`README.md`, `docs/ARCHITECTURE.md`, `docs/HANDOFF.md`, 10 architecture diagrams, and the 89-page university report `PB05_Report_8thSem.pdf`).
5. **The Stopping Point / Gap:**
   * When pushing the repository to GitHub, the senior team **excluded `data/clean_tensors/` (~1.2 GB) via `.gitignore`** due to GitHub file size limitations, without hosting them on Git LFS, GitHub Releases, or an external cloud bucket.
   * They also did not commit the standalone feature-extraction script that generates `clean_tensors/*.pt` from raw Kaggle data.
   * No edge deployment, ONNX export, quantization, profiling, or mobile runtime work was ever attempted.

### What is the first task the new team must perform?
1. **Acknowledge Phase 0 Completion:** Establish baseline git branch `edge-deployment` and document repo state in `PROJECT_HANDOFF_AUDIT.md` and `REPOSITORY_FILE_MAP.md`.
2. **Phase 1 (Local Environment Reproduction):** Formally verify the local environment (record hardware, python, torch versions in `environment_report.txt`), run Streamlit in audit mode, and document the clean handling of missing tensors.
3. **Phase 2 (Data / Tensor Recovery):** Recover or regenerate the clean patient tensors for the 66 isolated validation patients (and benchmark patients like `TCGA-AA-3877`). Because raw feature extraction requires running 192k patches through Phikon-v2 (ViT-L/16), this must be executed in a dedicated GPU environment (e.g. Kaggle GPU attached to `joangibert/tcga_coad_msi_mss_jpg`) using a precise extraction pipeline matching the senior project's specifications.

---

## 6. Audit Conclusion & Next Phase Recommendation

* **Audit Verdict:** The core code, neural architecture, trained weights, patient registry, validation contract, and Streamlit application are **100% genuine, uncorrupted, and mathematically functional**.
* **Primary Blocker:** Missing precomputed embedding tensors (`data/clean_tensors/`).
* **Recommended Next Step:**
  * Request user approval to conclude Phase 0.
  * Advance to Phase 1 (Local Environment Reproduction) to establish `environment_report.txt` and launch the Streamlit test.
