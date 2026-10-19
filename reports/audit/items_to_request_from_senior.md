# Prioritized Inventory of Items to Request from Senior Team

**Audit Date:** 2026-10-12  
**Target:** Senior Research / Engineering Team  
**Status:** **READY FOR SUBMISSION**  

---

## 1. Executive Request Matrix

| Priority Classification | Item Description | Local Availability | Can Be Reconstructed? | Impact if Unavailable |
| :--- | :--- | :--- | :--- | :--- |
| **MUST HAVE NOW** | Checkpoint versioning clarification (`fusion_model_v1_1.pth` vs `best_fusion_model.pth` vs `v1_0` / `v1_2`) | Checkpoints exist locally with different SHA-256 hashes | No (requires author clarification on training run differences) | Ambiguity regarding which checkpoint represents the definitive published model. |
| **MUST HAVE NOW** | Offline tensor generation script (`extract_features.py` / Kaggle script) | Missing locally | Partially (can be reverse-engineered from docs and `generate_pixel_xai.py`) | Inability to verify exact stain normalization parameters and patch extraction thresholds used to create the 652 `.pt` files. |
| **REQUIRED FOR REPRODUCIBILITY** | Explicit list of Training Patient IDs (`actual_train_keys.json`) | Inferred by set complement (326 total − 66 val) | Yes (can be frozen immediately) | Risk of train-test leakage if registry entries are added or modified. |
| **REQUIRED FOR REPRODUCIBILITY** | Deterministic seed specification for tile subsampling (`torch.randperm` in `multimodal_dataset.py`) | Missing (currently unseeded) | Yes (can be fixed by setting `torch.manual_seed(42)`) | Stochastic run-to-run metric variance (~87.88% vs 89.39% accuracy) on slides with $>1000$ patches. |
| **REQUIRED FOR RAW-INPUT INFERENCE** | Full Pretrained Weights for `owkin/phikon-v2` (~1.2 GB) and `Bio_ClinicalBERT` (~436 MB) | Configs only in local cache; weights missing | Yes (can be downloaded from Hugging Face if internet/token permitted) | Blocked from processing raw H&E images or novel unstructured clinical reports. |
| **REQUIRED FOR RAW-INPUT INFERENCE** | Exact Reinhard stain normalization reference target tile | Missing | No (requires the specific reference image used during preprocessing) | Slight color domain shift when normalizing new histological slides. |
| **REQUIRED FOR EDGE DEPLOYMENT** | Target Android Device Specifications (SoC, NPU TOPS, RAM, OS version) | Undefined | No (requires client / product requirement) | Inability to select optimal runtime (ONNX Runtime vs Qualcomm QNN vs ExecuTorch) and memory budget. |
| **OPTIONAL** | Original Kaggle Notebook and training history logs | Documented in `PB05_Report_8thSem.pdf` | Not needed for execution | Helpful for audit provenance and research history, but not strictly required to run inference. |

---

## 2. Detailed Justifications by Item

### 1. Checkpoint Versioning Clarification
* **Priority:** **MUST HAVE NOW**
* **Why Required:** The `models/` directory contains four distinct Cross-Attention checkpoints (`best_fusion_model.pth`, `fusion_model_v1_0.pth`, `fusion_model_v1_1.pth`, and `fusion_model_v1_2.pth`). They all possess identical parameter counts (5,250,561) and file sizes (21,007,347 bytes) but have completely distinct SHA-256 hashes. `fusion_model_v1_1.pth` is hardcoded in the evaluation scripts, but clarification on what epoch or hyperparameter variation `best_fusion_model.pth` represents is essential.
* **Blocker if Missing:** We might benchmark an intermediate checkpoint instead of the definitive final epoch.

### 2. Offline Tensor Generation Script
* **Priority:** **MUST HAVE NOW**
* **Why Required:** The codebase currently operates on precomputed embeddings (`data/clean_tensors/*.pt`). The Python script that accepted raw whole slide images, applied Otsu foreground filtering, Reinhard stain normalization, sliced $224 \times 224$ tiles, and fed them through Phikon-v2 to save `*_vision.pt` is missing from `src/`.
* **Blocker if Missing:** We cannot ingest a new patient's raw pathology slide into the pipeline without reconstructing this preprocessing code.

### 3. Pretrained Foundation Weights (`owkin/phikon-v2` & `ClinicalBERT`)
* **Priority:** **REQUIRED FOR RAW-INPUT INFERENCE**
* **Why Required:** To run end-to-end inference from raw pixels and text on a new patient or test slide, the full weights (~1.2 GB and ~436 MB) are mandatory. The local Hugging Face cache currently holds only architecture configuration files.
* **Can be Reconstructed:** Yes, by running `AutoModel.from_pretrained("owkin/phikon-v2")` and `AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")` if network access is authorized.
* **Blocker if Missing:** Pipeline remains strictly restricted to precomputed tensor mode.

### 4. Target Android Device Hardware Profile
* **Priority:** **REQUIRED FOR EDGE DEPLOYMENT**
* **Why Required:** End-to-end inference requires ~123 TFLOPs and ~1 GB of model weights. Knowing whether the target Android device is a flagship with an NPU ($\ge 45\text{ TOPS}$, e.g. Snapdragon 8 Gen 3) or a mid-range phone dictates whether the entire vision backbone can run on-device or whether a client-server / tiled embedding architecture must be used.
