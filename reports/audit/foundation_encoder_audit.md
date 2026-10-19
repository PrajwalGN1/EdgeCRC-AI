# Foundation Encoders Audit (Phikon-v2 & ClinicalBERT)

**Audit Date:** 2026-10-12  
**Target Encoders:** `owkin/phikon-v2` (Vision) • `emilyalsentzer/Bio_ClinicalBERT` (Text)  
**Status:** **CONFIGURATIONS PRESENT • FULL PRETRAINED WEIGHTS NOT LOCAL**  

---

## 1. Local Cache & Availability Classification

An inspection of the local Hugging Face cache (`~/.cache/huggingface/hub/`) establishes the following:

| Component | Identifier | Local Files Observed | Availability Category | Pretrained Weights Size |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Backbone** | `owkin/phikon-v2` | `config.json`, `refs/main` | **Category B** (Config present, weights missing) | ~1.2 GB (`model.safetensors`) |
| **Text Backbone** | `emilyalsentzer/Bio_ClinicalBERT` | `config.json`, `vocab.txt`, partial `.incomplete` blobs | **Category C** (Config & vocab present, partial download blobs) | ~436 MB (`pytorch_model.bin`) |

### Classification Legend:
* **Category A:** Complete pretrained weights available and loadable locally.
* **Category B:** Configuration and tokenizer files present, but pretrained weights missing.
* **Category C:** Partial downloads or incomplete snapshots present.
* **Category D:** Weights present but model loading unverified.
* **Category E:** Weights available only remotely.

> [!IMPORTANT]
> The full multi-gigabyte pretrained weights for Phikon-v2 and Bio_ClinicalBERT are **not stored locally on disk**. They were utilized during initial offline feature extraction (on a remote GPU workstation / Kaggle notebook) to generate the 652 cached `.pt` files.

---

## 2. In-Code Usage of Foundation Encoders

1. **Computational Profiling:**
   In [benchmark_computational_power.py](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_computational_power.py#L108-L109):
   ```python
   p_config = AutoConfig.from_pretrained("owkin/phikon-v2")
   p_model = AutoModel.from_config(p_config)
   ```
   * The benchmark instantiated the model graph directly from architecture configs (`AutoModel.from_config`) using synthetic tensors to profile FLOPs, parameter counts, and memory without requiring the 1.2 GB weights to be downloaded.
2. **Pixel-Level Explainable AI (XAI):**
   In [generate_pixel_xai.py](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/generate_pixel_xai.py#L243-L255):
   ```python
   processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2", trust_remote_code=True)
   model = AutoModel.from_pretrained("owkin/phikon-v2", output_attentions=True, trust_remote_code=True)
   ```
   * This module is programmed to fetch the weights from Hugging Face on-demand if invoked to compute attention rollouts on raw JPEG image tiles.

---

## 3. Preprocessing & Feature Extraction Traceability

* **Histology Image Tile Pipeline:**
  * Raw tiles are $224 \times 224$ pixels at $10\times$ magnification (~$1.0\ \mu\text{m}/\text{px}$).
  * Background filtering: Otsu thresholding removes tiles with $>50\%$ non-tissue background.
  * Color normalization: Reinhard stain normalization against a standard reference slide.
  * Phikon-v2 embedding: ViT-L/16 patch projection maps each tile to a 1024-D embedding.
* **Clinical Report Pipeline:**
  * Clinical notes undergo WordPiece tokenization via ClinicalBERT tokenizer (`max_length=512`).
  * Feature extraction extracts the `[CLS]` token embedding as a $1 \times 768$ summary vector.
* **Missing Artifact:**
  * The original batch extraction script (e.g., `extract_features.py` or Kaggle feature extraction notebook) that generated the 652 `.pt` files from the raw Kaggle dataset is **not committed in the repository**.
  * Running inference on a novel, un-preprocessed whole-slide scan currently requires writing this ingestion script or retrieving the original code from the senior team.
