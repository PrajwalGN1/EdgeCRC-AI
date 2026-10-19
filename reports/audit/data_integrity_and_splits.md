# Patient Registry, Labels, and Data Splits Audit

**Audit Date:** 2026-10-12  
**Target Files:** [`data/patient_registry.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/data/patient_registry.json), [`results/actual_validation_keys.json`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/results/actual_validation_keys.json)  

---

## 1. Patient Registry Schema & Content

`data/patient_registry.json` is a 13.1 MB JSON file serving as the central metadata registry:
* **Total Registered Patients:** **326**
* **Schema Per Record:**
  ```json
  "TCGA-NH-A5IV": {
    "label": 0,
    "class_name": "MSS",
    "report_text": "Surgical pathology report text excerpt...",
    "image_paths": ["/path/to/patch1.jpg", "..."],
    "patch_count": 482
  }
  ```
* **Label Integrity:**
  * Target variable is binary: `label = 0` (MSS - Microsatellite Stable) and `label = 1` (MSI - Microsatellite Instability).
  * **Class Breakdown:**
    * **MSS (Class 0):** **263 patients** (80.67%)
    * **MSI (Class 1):** **63 patients** (19.33%)
    * **Class Imbalance Ratio:** $263 / 63 \approx 4.17 : 1$. (This matches the loss weight `pos_weight = 4.17` used during Weighted BCE training).
* **Missing Data Scan:**
  * Missing report texts: **0**
  * Missing image path entries: **0**
  * Tensors mapped to registry: **326 of 326 (100.0%)**

---

## 2. Evaluation Split & Data Contract

### Validation Set Contract (`results/actual_validation_keys.json`)
* The file stores a list of exactly **66 isolated patient keys**.
* **Split Characteristics:**
  * **Total Size:** 66 patients (representing ~20.25% of the 326 total cohort).
  * **Class Distribution in Validation Set:**
    * **MSS (Class 0):** **50 patients** (75.76%)
    * **MSI (Class 1):** **16 patients** (24.24%)
  * **Integrity:** 100% of the 66 keys resolve to active records in both `patient_registry.json` and `data/clean_tensors/`.
  * **Patient-Level Isolation:** Splitting is strictly **patient-level**. All tiles/patches belonging to a patient reside solely within either the training partition or the validation partition. There is **zero patch leakage** across splits.

### Training Split Documentation Gap
* While the validation set is explicitly locked by `actual_validation_keys.json`, there is **no explicit `actual_train_keys.json` file** committed to the repository.
* The training cohort (260 patients: 213 MSS, 47 MSI) is currently inferred implicitly by taking the complement:
  $$\text{Train Keys} = \text{All Registry Keys} \setminus \text{Validation Keys}$$
* **Recommendation:** Formalize and save `train_keys.json` to prevent accidental divergence if future records are appended to the registry.

---

## 3. Stochastic Evaluation Nuance (Tile Subsampling)

In [multimodal_dataset.py](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/multimodal_dataset.py#L76-L80):
```python
if num_patches > self.max_patches:
    indices = torch.randperm(num_patches)[:self.max_patches]
    vision_tensor = vision_tensor[indices]
```
* For slides containing $>1,000$ patches, `torch.randperm` executes **without a fixed seed** during evaluation.
* Because the evaluation scripts ([cross_attention_eval.py](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py)) do not set `torch.manual_seed()` prior to the evaluation loop, slight run-to-run variations in test accuracy and ROC-AUC can be observed (e.g., accuracy varying between 87.88% and 89.39%, AUC between 0.9150 and 0.9250).
* **Fix for 100% Deterministic Reproducibility:** Call `torch.manual_seed(42)` immediately before initializing the evaluation DataLoader.
