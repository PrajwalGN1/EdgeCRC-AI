# Repository File Map & Architectural Inventory

> **Project:** Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2  
> **Repository:** `Mokshithc/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Date:** 2026-10-03  
> **Branch:** `edge-deployment`

---

## 1. Model Components & Neural Architecture Map

| Symbol / Class | File Location | Line Range | Input Shape | Output Shape | Trainable Parameters | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SummaryToPatchCrossAttention` | [`cross_attention_fusion.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/cross_attention_fusion.py#L4-L57) | L4–L57 | `vision_seq`: `[B, N, 1024]`<br>`text_summary`: `[B, 768]` | `probability`: `[B, 1]`<br>`attn_weights`: `[B, 1, N]` | **5,250,561** | **Primary Architecture:**<br>1. Linear projection (768 → 1024)<br>2. 8-head MHA ($Q$=text, $K$=$V$=patches)<br>3. LayerNorm residual connection<br>4. MLP classifier (1024 → 256 → 1) + Sigmoid. |
| `MultimodalFusionNetwork` | [`late_fusion_model.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/late_fusion_model.py#L5-L77) | L5–L77 | `vision_tensor`: `[1, N, 1024]`<br>`text_tensor`: `[1, 768]` | `prob`: `[1, 1]`<br>`A`: `[N, 1]` | **1,313,281** | **Baseline Architecture (AB-MIL):**<br>1. Attention-based MIL on patches → `[1, 1024]`<br>2. Text projection (768 → 512)<br>3. Concatenation `[1, 1536]`<br>4. MLP classifier (1536 → 256 → 1). |
| `Phikon-v2` (Encoder) | HuggingFace (`owkin/phikon-v2`) | Referenced in `docs/` & `src/generate_pixel_xai.py#L244` | RGB Image patch `[3, 224, 224]` | Patch embedding `[1024]` | 307M (Frozen) | ViT-L/16 trained on 460M pan-cancer histology tiles. Used offline for patch feature extraction. |
| `ClinicalBERT` (Encoder) | HuggingFace (`Bio_ClinicalBERT`) | Referenced in `docs/` | Token IDs `[1, L]` ($L \le 512$) | CLS summary `[1, 768]` | 110M (Frozen) | BioBERT initialized, fine-tuned on 2M MIMIC-III clinical notes. Used offline for text feature extraction. |

---

## 2. Checkpoint Registry (`models/`)

| Checkpoint File | Size on Disk | Architecture Target | Validation Metric | Status in Git | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/fusion_model_v1_1.pth` | 21,007,347 bytes | `SummaryToPatchCrossAttention` | **ROC-AUC: 0.9150**, Acc: 89.39% | ✅ Committed to Git | **Primary Designated Checkpoint (Star Model).** Contains 5,250,561 float32 weights. |
| `models/best_fusion_model.pth` | 21,007,347 bytes | `SummaryToPatchCrossAttention` | ROC-AUC: ~0.9150 | ✅ Committed to Git | Primary fallback; binary identical or alternative epoch checkpoint. |
| `models/fusion_model_v1_0.pth` | 21,007,347 bytes | `SummaryToPatchCrossAttention` | ROC-AUC: ~0.90 | ✅ Committed to Git | Intermediate training checkpoint. |
| `models/fusion_model_v1_2.pth` | 21,007,347 bytes | `SummaryToPatchCrossAttention` | ROC-AUC: ~0.91 | ✅ Committed to Git | Experimental fine-tuning iteration. |
| `models/late_fusion_model_v1_0.pth` | 5,255,861 bytes | `MultimodalFusionNetwork` | ROC-AUC: 0.8950, Acc: 81.82% | ✅ Committed to Git | Baseline model trained weights. |
| `models/late_fusion_model.pth` | 5,255,717 bytes | `MultimodalFusionNetwork` | ROC-AUC: ~0.8950 | ✅ Committed to Git | Alternative baseline checkpoint. |

---

## 3. Data & Contract Dependencies (`data/`, `results/`)

| Path | Purpose | Format | Size | Present Locally? | Origin / How to Restore |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `data/patient_registry.json` | Master index mapping 326 patients to metadata, labels, report text, and image patch paths. | JSON | 13.1 MB | ✅ YES | Tracked in Git. Source of truth for patient metadata. |
| `results/actual_validation_keys.json` | Strict 66-patient holdout validation contract (50 MSS, 16 MSI). | JSON (list of strings) | 1,056 bytes | ✅ YES | Tracked in Git. Enforces zero data leakage for all evaluations. |
| `data/clean_tensors/*_vision.pt` | Precomputed Phikon-v2 embeddings (`[N, 1024]`). | PyTorch Tensor (`torch.save`) | ~1.2 GB (326 files) | ❌ NO | Excluded via `.gitignore`. Must be regenerated from raw Kaggle dataset using Phikon-v2 on GPU. |
| `data/clean_tensors/*_text.pt` | Precomputed ClinicalBERT embeddings (`[1, 768]`). | PyTorch Tensor (`torch.save`) | ~1.0 MB (326 files) | ❌ NO | Excluded via `.gitignore`. Can be regenerated from `patient_registry.json` reports using ClinicalBERT. |
| `data/MSIMUT_JPEG/` | Raw H&E patches for MSI cohort (75,039 images). | Directory of JPEGs | ~2.3 GB | ❌ NO (only `.gitkeep` & `README.txt`) | Excluded via `.gitignore`. Kaggle dataset: `joangibert/tcga_coad_msi_mss_jpg`. |
| `data/MSS_JPEG/` | Raw H&E patches for MSS cohort (117,273 images). | Directory of JPEGs | ~3.6 GB | ❌ NO (only `.gitkeep` & `README.txt`) | Excluded via `.gitignore`. Kaggle dataset: `joangibert/tcga_coad_msi_mss_jpg`. |

---

## 4. Pipeline Execution Scripts (`src/`)

| File Path | Role | Inputs | Outputs | Hardware / Execution Requirements |
| :--- | :--- | :--- | :--- | :--- |
| [`multimodal_dataset.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/multimodal_dataset.py) | PyTorch Dataset Loader | `patient_registry.json`, `data/clean_tensors/` | Batched dict: `patient_id`, `vision` `[1, N, 1024]`, `text` `[1, 768]`, `label` | CPU / Memory bound. Dynamic patch cap: random permutation to 1000 patches if $N > 1000$. |
| [`cross_attentiion_train.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attentiion_train.py) | Cross-Attention Training Loop | Clean tensors + Registry | Checkpoints in `models/`, loss trajectory plot in `results/`, validation keys | CPU / MPS / CUDA. 30 epochs, AdamW `lr=1e-4`, Weighted BCE `pos_weight=4.17`, `batch_size=1`. |
| [`cross_attention_eval.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py) | Holdout Evaluation Script | Clean tensors + `actual_validation_keys.json` + `fusion_model_v1_1.pth` | `Cross_Attention_Model_Evaluation_Report.txt`, `Confusion_Matrix_C.png`, `ROC_Curve_C.png`, `Real_Cross_Attention_Matrix.png` | CPU / MPS / CUDA. Runs strictly on 66 holdout validation patients with decision threshold $\tau = 0.83$. |
| [`late_fusion_train.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/late_fusion_train.py) | Baseline Training Script | Clean tensors + Registry | `models/late_fusion_model_v1_0.pth` | Same as cross-attention trainer. |
| [`late_fusion_eval.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/late_fusion_eval.py) | Baseline Evaluation Script | Clean tensors + `actual_validation_keys.json` + `late_fusion_model_v1_0.pth` | `Late_FusionModel_Evaluation_Report.txt`, `Confusion_Matrix_L.png`, `ROC_Curve_L.png` | Threshold $\tau = 0.70$. |
| [`verify_cross_attention.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/verify_cross_attention.py) | Dynamic Attention Perturbation Proof | Patient `TCGA-A6-6142` tensors + `fusion_model_v1_1.pth` | `Cross_Attention_Verification_Proof.png`, console verification log | Compares attention profile under real clinical text vs. random noise. Shifts prove dynamic search. |
| [`tensor_dim_check.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/tensor_dim_check.py) | Diagnostic Tensor Shape Checker | Single `.pt` file | Console output of tensor shape and type | Quick diagnostic tool for tensor format inspection. |

---

## 5. User Interface & Serving (`src/ui/`, root)

| File Path | Role | Dependencies | Entrypoint Command | Status |
| :--- | :--- | :--- | :--- | :--- |
| [`app.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/ui/app.py) | Primary Streamlit Clinical App | `streamlit`, `torch`, `matplotlib`, `PIL`, `requests` (Ollama) | `streamlit run src/ui/app.py --server.port 8501` | ✅ Verified working; loads weights & registry cleanly. Shows placeholder if tensors missing. |
| [`app_legacy.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/ui/app_legacy.py) | Legacy Pre-Hardening App | `streamlit`, `torch` | Archived reference | Retained as baseline reference; do not modify. |
| [`Dockerfile`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/Dockerfile) | Container Deployment | Base image `python:3.10-slim` | `docker build -t crc-msi . && docker run -p 8501:8501 crc-msi` | Includes health check on `/_stcore/health`. |
| [`.streamlit/config.toml`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/.streamlit/config.toml) | Streamlit Server Config | Streamlit | Auto-loaded by Streamlit | Configures port 8501, headless mode, CORS settings. |

---

## 6. Explainable AI (XAI) Stack Map

| Tier | Name | Script Location | Mechanism | Fallback Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Macroscopic Ranking** | [`generate_xai.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/generate_xai.py#L86), [`app.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/ui/app.py#L271) | Extracts cross-attention weights `[1, 1, N]`, averages over 8 heads, executes `argsort()[-5:][::-1]` to identify the Top-5 most clinically salient patches. | If raw JPEGs are missing, renders colored score box placeholders with attention values. |
| **Tier 2** | **Microscopic Cellular Heatmap** | [`generate_pixel_xai.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/generate_pixel_xai.py#L230) | Loads `owkin/phikon-v2` with `output_attentions=True`, computes attention rollout product across last 6 layers, creates a $14 \times 14$ grid, applies Otsu tissue masking and Gaussian blur, overlays Jet heatmap + yellow ROI contours. | If transformers or Phikon model is unavailable, gracefully falls back to hematoxylin density approximation. |
| **Tier 3** | **Linguistic Clinical Scribe** | [`app.py`](file:///d:/Projects/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/ui/app.py#L127) | Packages patient clinical notes, predicted MSI status, confidence score, and top attention scores into a deterministic prompt sent to local Ollama (`qwen2.5-coder:7b`). | If Ollama is offline or unreachable at `localhost:11434`, automatically generates a 3-paragraph clinically grounded template report. |

---

## 7. Documentation & Reference Materials (`docs/`, root)

| File | Type | Description |
| :--- | :--- | :--- |
| `PB05_Report_8thSem.pdf` | University Project Report (89 pp) | Comprehensive academic thesis providing technical rationale, clinical background, ablation studies, loss trajectory theory, and hyperparameter justifications. |
| `docs/ARCHITECTURE.md` | System Architecture Document | 14-section architectural specification covering data curation, frozen feature engineering, cross-attention mechanics, deployment, and file mapping. |
| `docs/HANDOFF.md` | Operational Playbook | Guide describing what works, known debt/limitations, data contracts, and quick-start instructions. |
| `docs/generate_diagrams.py` | Diagram Generator Script | Pure Python/Matplotlib script that programmatically renders all 10 architectural diagrams at 300 DPI. |
| `docs/diagrams/` | Architecture Diagrams | 10 vector SVG and 10 high-resolution PNG diagrams illustrating the complete system pipeline. |
| `UNPUSHED_ASSETS.md` | Forensic Asset Document | Explicit documentation of all heavy files (>100 MB) excluded from the GitHub push with original paths. |
| `HEAVY_FILES_MANIFEST.txt` | Asset Size Manifest | Text summary of excluded heavy files and sizes. |
| `data/DATASET.md` | Dataset Ingestion Guide | Kaggle links, extraction steps, and directory structure notes for `tcga_coad_msi_mss_jpg`. |
