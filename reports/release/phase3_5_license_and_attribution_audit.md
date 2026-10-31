# Phase 3.5 — Licensing, Model Provenance & Attribution Audit

- **Project:** EdgeCRC-AI — Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2
- **Repository:** `https://github.com/PrajwalGN1/EdgeCRC-AI`
- **Audit Gate Status:** **`BLOCKED — OWNER LICENSE DECISION REQUIRED`** *(Mandatory gate blocking public GitHub Release publication until owner selects and commits a repository `LICENSE` policy)*

---

## 1. Executive Gate Determination

1. **Repository `LICENSE` File Inspection:** Verified via Git tree inspection (`git ls-files "*LICENSE*" "*COPYING*"`) that **no `LICENSE` file exists** in the repository root or subdirectories. Under international copyright law and GitHub Terms of Service, absence of a license defaults to **All Rights Reserved** (others may view and fork on GitHub, but have no explicit license to reuse, modify, or redistribute code or weights).
2. **Critical Upstream Non-Commercial Constraint (`Owkin Phikon-v2`):** While EdgeCRC-AI's original Python/Kotlin source code (`src/`, `tools/`, `android/`) uses only permissive open-source libraries (MIT / Apache-2.0 / BSD-3), the precomputed vision patch embeddings (`reference_tensors/*.bin`) and the trained downstream cross-attention fusion weights (`models/fusion_model_v1_1.pth`, `models/fusion_model_fp32.onnx`, `models/fusion_model_fp16.onnx`) were trained on features extracted by **`owkin/phikon-v2`**, which is governed by the **Owkin Non-Commercial License**.
3. **Gate Outcome:** Per release engineering safety rules, no license was invented or automatically added. Public GitHub Release creation is held at **`BLOCKED — OWNER LICENSE DECISION REQUIRED`** pending the repository owner's selection from the compliant licensing options in Section 5.

---

## 2. Upstream Foundation Models, Datasets & Dependency License Matrix

### 2.1 Pretrained Foundation Models & Derived Model Artifacts

| Component | Identifier / Source | Upstream License | Redistributed in Repo? | Legal & Redistribution Impact on EdgeCRC-AI |
| :--- | :--- | :--- | :---: | :--- |
| **Owkin Phikon-v2** (Vision Backbone, ViT-L/16) | `owkin/phikon-v2` (Hugging Face; Filiot et al.) | **Owkin Non-Commercial License** (`LICENSE.pdf`) | **No** (Downloaded dynamically from Hugging Face in `src/inference/vision_encoder.py` only if raw slides are encoded) | **Restricts downstream model weights & features to non-commercial academic/research use.** Commercial or clinical diagnostic use requires a separate commercial agreement with Owkin. |
| **Bio_ClinicalBERT** (Clinical Text Encoder) | `emilyalsentzer/Bio_ClinicalBERT` (Alsentzer et al.) | **MIT License** | **No** (Downloaded dynamically in `src/inference/report_encoder.py`) | Permissive; requires retaining attribution to Alsentzer et al. (2019). |
| **Trained Cross-Attention Fusion Checkpoints & ONNX Graphs** | `models/fusion_model_v1_1.pth`, `models/fusion_model_fp32.onnx`, `models/fusion_model_fp16.onnx`, `models/late_fusion_model_v1_0.pth` | Authored by project team; **derived from Phikon-v2 + ClinicalBERT embeddings on TCGA-COAD** | **Yes** (`~61.6 MB` across `.pth` & `.onnx` files) | Can be shared publicly **for non-commercial academic research** with Owkin Phikon-v2 & TCGA attribution; **cannot** be licensed under unrestricted commercial terms (e.g., blanket MIT/Apache-2.0 covering weights) without an Owkin commercial license. |

### 2.2 Datasets & Cohort Metadata

| Dataset / File | Source | Applicable Terms | Redistributed in Repo? | Compliance Status |
| :--- | :--- | :--- | :---: | :--- |
| **TCGA-COAD Open-Access Cohort** | NCI / NIH The Cancer Genome Atlas (`https://www.cancer.gov/tcga`) & Kaggle curated tiles (`joangibert/tcga_coad_msi_mss_jpg`) | **NIH/NCI TCGA Open-Access Data Tier Guidelines** (all publication moratoria lifted; de-identified public cohort; requires TCGA Research Network acknowledgment; prohibits re-identification attempts) | Raw `12 GB` JPEGs & `1.2 GB` `clean_tensors/` are **excluded** (`.gitignore`). De-identified metadata (`data/patient_registry.json`), 66-case split (`actual_validation_keys.json`, `tcga_holdout_reference_labels.*`), and `7` benchmark `.bin` feature files (`9.98 MB`) **are included**. | **Compliant for academic research**, provided the standard NCI TCGA Research Network acknowledgment statement is maintained. |

### 2.3 Android Application Dependencies (`android/app/build.gradle.kts`)

| Dependency | Version | SPDX License | Scope | Redistribution Notes |
| :--- | :--- | :--- | :--- | :--- |
| `com.microsoft.onnxruntime:onnxruntime-android` | `1.18.0` | **MIT** | Runtime (`implementation`) | Permissive; include MIT copyright notice if distributing compiled APK binaries. |
| `androidx.core:core-ktx` | `1.13.1` | **Apache-2.0** | Runtime (`implementation`) | Permissive. |
| `androidx.appcompat:appcompat` | `1.6.1` | **Apache-2.0** | Runtime (`implementation`) | Permissive. |
| `com.google.android.material:material` | `1.12.0` | **Apache-2.0** | Runtime (`implementation`) | Permissive. |
| `org.jetbrains.kotlin:kotlin-stdlib` / plugin | `1.9.24` | **Apache-2.0** | Runtime / Build | Permissive. |
| `junit:junit` | `4.13.2` | **EPL-1.0** | Test only (`testImplementation`) | Not packaged in APK. |
| `androidx.test:runner` & `ext:junit` | `1.5.2` / `1.1.5` | **Apache-2.0** | Test only (`androidTestImplementation`) | Not packaged in release APK. |
| `org.json:json` | `20240303` | **JSON License** | Test only (`testImplementation`) | Used only in host JVM unit tests; Android runtime uses built-in `android.util` / `org.json` clean-room SDK stub. |

### 2.4 Python Dependencies (`requirements.txt`)

- **Core ML & UI Stack:** `torch==2.9.1`, `torchvision==0.24.1`, `numpy==2.2.6`, `pandas==2.3.3`, `scikit-learn` -> **BSD-3-Clause**; `huggingface_hub==1.2.3`, `safetensors==0.7.0`, `timm==1.0.22`, `monai==1.5.1`, `pytorch-lightning==2.6.0`, `opencv-python==4.12.0.88`, `streamlit==1.52.2`, `pyarrow==22.0.0`, `playwright==1.56.0` -> **Apache-2.0**; `segmentation_models_pytorch==0.5.0`, `attrs`, `jsonschema` -> **MIT**; `pillow==12.0.0` -> **MIT-CMU**.
- **Copyleft Check:** **Zero GPL or AGPL runtime dependencies** were detected in `requirements.txt` or `android/app/build.gradle.kts`.

---

## 3. Direct Answers to Mandatory Gate Questions

1. **Can the project be published as source code?**
   - **Yes (`PASS` — subject to owner's `LICENSE` selection).** All original Python (`src/`, `tools/`, `tests/`) and Android Kotlin/XML (`android/`) source files were authored for this project and link only against permissive Apache-2.0 / MIT / BSD-3 libraries.
2. **Can pretrained model weights (`fusion_model_*.pth` / `.onnx`) be redistributed?**
   - **Yes, strictly under Non-Commercial Research terms (`CONDITIONAL`).** Because the fusion weights were trained on features generated by **Owkin Phikon-v2** (`Owkin Non-Commercial License`), redistributing the `.pth` and `.onnx` weights on GitHub is permissible **only for non-commercial academic/research purposes** and must carry a clear non-commercial restriction notice. They must **not** be offered for commercial use without Owkin's written authorization.
3. **Can dataset-derived files (`patient_registry.json`, `tcga_holdout_reference_labels.*`, `7` `reference_tensors/*.bin`) be redistributed?**
   - **Yes, under Non-Commercial Research + TCGA Open-Access attribution (`CONDITIONAL`).** All `326` cases belong to the public, de-identified TCGA-COAD Open-Access cohort (all TCGA publication moratoria lifted), and the `7` `.bin` tensors inherit Phikon-v2's Non-Commercial condition.
4. **Is third-party attribution complete?**
   - **`README.md` Section 32** already credits academic mentor **Mr Nageshwara Guptha**, **Owkin Phikon-v2** (Filiot et al.), **Bio_ClinicalBERT** (Alsentzer et al.), **TCGA-COAD Research Network**, and **Microsoft ONNX Runtime**. Adding the exact NCI-recommended TCGA sentence (*"The results shown here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga."*) inside `LICENSE` / `NOTICE` when adding the license will make formal attribution 100% complete.
5. **Do any upstream restrictions affect the project?**
   - **Yes:** (a) **Owkin Non-Commercial License** prohibits commercial or clinical revenue-generating use of Phikon-v2-derived embeddings and models; (b) **TCGA Data Use Policy** prohibits any attempt to re-identify individual human subjects; and (c) **Medical Device Regulations** prohibit clinical diagnostic use without regulatory clearance.

---

## 4. Existing Author & Copyright Information Verified

- **Project Developer (`README.md` Section 31 & Git Commit History):** **Prajwal G N** (`PrajwalGN1`, `https://github.com/PrajwalGN1/`).
- **Academic Project Mentor (`PB05_Report_8thSem.pdf` & `README.md` Section 32):** **Mr Nageshwara Guptha**.
- *(Note: If the capstone project was completed under a university intellectual-property policy, the owner should confirm whether institutional co-ownership or sign-off applies before selecting a license.)*

---

## 5. Recommended Licensing Choices for Repository Owner Decision

| Option | Structure | Code License | Model Weights (`.pth`/`.onnx`) & `.bin` Tensors | Pros | Cons / Restrictions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Option 1 (Recommended Best Practice)** | **Dual / Split Licensing (`LICENSE` + `MODEL_LICENSE.md`)** | **Apache-2.0** (or **MIT**) for all `.py`, `.kt`, `.xml`, `.kts` source code | **CC-BY-NC-4.0 + Owkin Non-Commercial Notice** for `models/` and `reference_tensors/` | Standard in open-source ML research: allows unrestricted academic/software reuse of the Android workstation & Python pipeline code while legally respecting Owkin Phikon-v2's Non-Commercial constraint on derived weights/tensors. | Requires clearly separating code vs. model weight scope in `LICENSE` and `README.md`. |
| **Option 2 (Conservative Single-Policy)** | **Repository-Wide Non-Commercial Academic Research License (`CC-BY-NC-4.0` or Custom Academic Research License)** | Non-Commercial Research Only | Non-Commercial Research Only | Single uniform rule across the entire repository; zero risk of downstream users mistakenly using weights commercially. | Not an OSI-approved "open-source" code license because commercial use of code is also restricted. |
| **Option 3 (Unrestricted MIT / Apache-2.0 Across Whole Repo)** | Single permissive license covering **both** code and `models/` | MIT / Apache-2.0 | MIT / Apache-2.0 | Simplest single file. | **NOT RECOMMENDED (`CONFLICTS WITH PHIKON-V2`):** Granting commercial rights over Phikon-v2-derived `.onnx`/`.pth` weights and `.bin` tensors conflicts with the upstream **Owkin Non-Commercial License**. |

---

## 6. Exact Action Required from Repository Owner to Clear Gate

1. Select **Option 1** (Apache-2.0 or MIT for source code + Non-Commercial Research terms for `models/` and `reference_tensors/`) or **Option 2** (Repository-wide Non-Commercial Research license).
2. Authorize creation of the root `LICENSE` (and `NOTICE` / `MODEL_LICENSE.md` if Option 1) with `Copyright (c) 2026 Prajwal G N` and the NCI TCGA acknowledgment sentence.
3. Update Section 33 of `README.md` to reference the committed `LICENSE` file, then proceed with GitHub Release publication (`Phase I`).
