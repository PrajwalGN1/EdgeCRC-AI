# Foundation Model & Derived Model Artifact Licensing (`MODEL_LICENSE.md`)

> [!IMPORTANT]
> **Separate Upstream Governance Over Model Weights & Embeddings:**
> The root [`LICENSE`](LICENSE) file (**PolyForm Noncommercial License 1.0.0**) applies **only** to original project-authored software source code. It does **not** apply to, replace, or add conflicting terms onto foundation models, precomputed embeddings, or trained model checkpoints/ONNX graphs derived from upstream foundation models.

---

## 1. Summary Governance Matrix

| Asset Category | Repository Paths / Identifiers | Upstream Foundation Dependency | Applicable License / Governance | Commercial Use Permitted? |
| :--- | :--- | :--- | :--- | :---: |
| **Upstream Vision Backbone (`Phikon-v2`)** | `owkin/phikon-v2` *(not bundled in repository; downloaded dynamically in `src/inference/vision_encoder.py`)* | Owkin Inc. (`ViT-L/16` self-supervised pathology encoder) | **Owkin Phikon-v2 License Terms and Conditions** (`LICENSE.pdf`) | **No** (Requires separate written agreement with Owkin Inc.) |
| **Upstream Clinical Text Encoder (`Bio_ClinicalBERT`)** | `emilyalsentzer/Bio_ClinicalBERT` *(not bundled in repository; downloaded dynamically in `src/inference/report_encoder.py`)* | Emily Alsentzer et al. (MIT CSAIL / Clinical NLP) | **MIT License** | **Yes** (Subject to MIT copyright/permission notice) |
| **Trained Multimodal Fusion Checkpoints & Exported ONNX Graphs (`Results`)** | `models/fusion_model_v1_1.pth`, `models/late_fusion_model_v1_0.pth`, `models/fusion_model_fp32.onnx`, `models/fusion_model_fp16.onnx`, `android/app/src/main/assets/models/fusion_model_fp32.onnx` | Trained on `owkin/phikon-v2` `[1, N, 1024]` patch embeddings + `Bio_ClinicalBERT` `[1, 768]` `[CLS]` embeddings over `TCGA-COAD` | **Owkin Phikon-v2 License Terms and Conditions** (`Section 2(a)(1)(C)` & `Section 3(c)` ShareAlike on `Results`) + **MIT Notice** (`Bio_ClinicalBERT`) + **TCGA Attribution** | **No** (Non-commercial research by non-profit entities only; commercial use requires Owkin Inc. approval) |
| **Precomputed Multimodal Reference Embeddings (`Results`)** | `android/app/src/main/assets/reference_tensors/*.bin` (`7` `CRCT` v1 files) and exported `.crcbundle` / `.npy` feature bundles | Extracted via `owkin/phikon-v2` + `Bio_ClinicalBERT` from de-identified `TCGA-COAD` cases | **Owkin Phikon-v2 License Terms and Conditions** (`Results` ShareAlike) + **MIT Notice** + **TCGA Attribution** | **No** (Non-commercial research by non-profit entities only) |

---

## 2. Owkin Phikon-v2 — Upstream Terms & Derived Artifact Analysis

### 2.1 Original Model and Official Source
- **Model Identifier:** `owkin/phikon-v2` (`Dinov2Model` Vision Transformer `ViT-L/16`, `~307M` parameters, `1024`-dimensional patch embeddings)
- **Creator / Licensor:** **Owkin Inc.**, 831 Broadway, Unit 3R, New York, NY 10003, USA
- **Official Model Repository:** https://huggingface.co/owkin/phikon-v2
- **Applicable Version / Integration in EdgeCRC-AI:** Referenced in `src/inference/vision_encoder.py` via `AutoImageProcessor.from_pretrained("owkin/phikon-v2")` and `AutoModel.from_pretrained("owkin/phikon-v2")`. **EdgeCRC-AI does not store, modify, or redistribute the `owkin/phikon-v2` backbone weights.**

### 2.2 Upstream License and Official Link
- **License Title:** *Phikon-v2 — License Terms and Conditions* (`LICENSE.pdf`, 14 pages, including Annex 1 dataset notices)
- **Official Upstream License URL:** https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf
- **Governing Law:** State of New York, USA (`Section 9(d)`).

### 2.3 Non-Commercial & Non-Profit Use Requirement (`Preamble`, `Section 1`, `Section 2(a)(1)`)
Direct inspection of the official `owkin/phikon-v2` `LICENSE.pdf` establishes strict entity and purpose boundaries:
1. **Non-Profit Entity Scope (`Preamble`):**
   *"For the sake of clarity, this License is limited to grant non-profit entities with the rights defined hereunder. If You are a for-profit entity, You are invited to directly contact the Licensor for negotiating a separate agreement."*
2. **Definition of `"Non-Commercial"` (`Section 1`):**
   *"Means for non-commercial (research) purposes by non-profit entity only. Non-Commercial excludes any use, sale, lease, license, or other transfer of the Licensed Material, Derivative Works or Results by or to a for-profit organization. Non-Commercial shall also exclude uses of the Licensed Material, Derivative Works or Results by any organization, including You, to perform contract research, to screen compound libraries, to produce or manufacture product for general sales, or to conduct research activities that result in any sale, lease, license, or transfer of the Licensed Material, Derivative Works or Results to a for-profit organization."*
3. **No Patent Filing (`Section 2(a)(1)` & `Section 2(a)(3)`):**
   Neither the Licensed Material, Derivative Works, nor Results may be used in connection with any patent application, and no patent rights of Owkin Inc. are licensed.

### 2.4 Qualification of EdgeCRC-AI Assets: `Derivative Works` vs. `Results` (`Section 1`)
Under `Section 1` of the Owkin `LICENSE.pdf`:
- **`"Derivative Work"`** means any material subject to Copyright and Similar Rights in which the `Licensed Material` (`owkin/phikon-v2`) is translated, altered, transformed, modified, or included. Because EdgeCRC-AI neither modifies nor packages the `owkin/phikon-v2` weights, **no file in this repository is a `Derivative Work` of the Phikon-v2 model itself**.
- **`"Results"`** means *"any results, discovery or creation, of any kind and in any form whatsoever, whether patentable or not, which results from the use of the Licensed Material, excluding Derivative Works."*

Accordingly, the following EdgeCRC-AI artifacts qualify directly as **`Results`** produced through the use of `owkin/phikon-v2`:
1. **Precomputed Vision Embedding Tensors:** `android/app/src/main/assets/reference_tensors/*.bin` (`7` `CRCT` v1 benchmark specimens containing `[1, N, 1024]` Phikon-v2 patch embeddings) and any `.crcbundle` (`CRCB` v2), `.crcmanifest.json`, or `*_vision.npy` feature bundles exported via `tools/export_specimen_bundle.py`.
2. **Trained Downstream Fusion Checkpoints & Exported ONNX Models:** `models/fusion_model_v1_1.pth`, `models/late_fusion_model_v1_0.pth`, `models/fusion_model_fp32.onnx`, `models/fusion_model_fp16.onnx`, and `android/app/src/main/assets/models/fusion_model_fp32.onnx` (trained over Phikon-v2 `[1, N, 1024]` patch representations paired with Bio_ClinicalBERT `[1, 768]` clinical text representations).

### 2.5 Redistribution, ShareAlike, and Downstream-Recipient Requirements (`Section 2(a)(1)(C)`, `Section 2(a)(4)`, `Section 3(b)–(c)`)
- **Right to Share `Results` (`Section 2(a)(1)(C)`):** Non-profit licensees are granted a worldwide, royalty-free, non-sublicensable, non-exclusive license to *"produce, reproduce, and Share the Results for Non-Commercial purposes only. Any use of the Results other than Non-Commercial use shall be prior approved by the Licensor."*
- **Downstream-Recipient Direct License (`Section 2(a)(4)(C)`):** Every qualifying non-profit recipient of `Results` automatically receives an offer from the Licensor (`Owkin Inc.`) to exercise the rights granted under the Owkin License, subject to full compliance with its terms and conditions.
- **ShareAlike Obligation on `Results` (`Section 3(c)(1)–(3)`):** If a licensee Shares `Licensed Material`, `Derivative Works`, **or `Results`**, the Derived License applied **must have the same terms and conditions as the Owkin License** (that version or later), must prohibit any use other than `Non-Commercial`, and must include the text of or URI/hyperlink to the Owkin License (`https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf`).

### 2.6 Prohibition on Additional or Different Terms (`Section 3(c)(4)`)
`Section 3(c)(4)` of the Owkin `LICENSE.pdf` states:
> *"You may not offer or impose any additional or different terms or conditions on, or apply any Effective Technological Measures to, Derivative Works or Results that modify the rights granted under this License."*

**Why PolyForm Noncommercial 1.0.0 Is NOT Applied to Model Assets:**
Applying the PolyForm Noncommercial License 1.0.0 (which differs from Owkin's license in entity scope, personal-study provisions, and patent clauses) to `models/*` or `reference_tensors/*` would impose *"additional or different terms"* in conflict with `Section 3(c)(2)` and `Section 3(c)(4)`. Therefore, **PolyForm Noncommercial 1.0.0 is scoped strictly to original software source code in [`LICENSE`](LICENSE)**, and all Phikon-v2-derived `Results` (`models/*.pth`, `models/*.onnx`, `android/app/src/main/assets/models/*.onnx`, and `android/app/src/main/assets/reference_tensors/*.bin`) are governed **exclusively by the upstream Owkin Phikon-v2 License Terms and Conditions** (`https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf`).

### 2.7 Required Attribution (`Section 3(b)` & Annex 1)
Any permitted non-commercial sharing or publication involving Phikon-v2 or EdgeCRC-AI's Phikon-v2-derived `Results` must retain:
1. Identification of the upstream creators (**Owkin Inc.**; Filiot et al., 2024).
2. A URI/hyperlink to the Licensed Material (`https://huggingface.co/owkin/phikon-v2`) and to the upstream License (`https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf`), noting that EdgeCRC-AI's fusion weights and `.bin` reference embeddings are downstream `Results` generated using `owkin/phikon-v2`.
3. Upstream `PANCAN-XL` pretraining dataset notices recorded in Annex 1 of Owkin's `LICENSE.pdf` (`TCGA` Open Access, `CPTAC` CC-BY-4.0 / CC-BY-3.0, `GTEx` Open Access, and `TCIA` CC-BY-4.0 / CC-BY-3.0 / Restricted Licence).
4. Academic citation:
   ```bibtex
   @article{filiot2024phikonv2,
     title   = {Phikon-v2, A large and public feature extractor for biomarker prediction},
     author  = {Filiot, Alexandre and Jacob, Paul and Mac Kain, Alice and Saillard, Charlie},
     journal = {arXiv preprint arXiv:2409.09173},
     year    = {2024},
     url     = {https://arxiv.org/abs/2409.09173}
   }
   ```

### 2.8 Commercial-Use Permission Requirements
Any use of `owkin/phikon-v2` or EdgeCRC-AI's Phikon-v2-derived `Results` (`models/*`, `reference_tensors/*`, or compiled APKs containing `fusion_model_fp32.onnx`) by a **for-profit entity**, in **contract research**, in **clinical/commercial services**, or in **transfers to for-profit organizations** is **strictly prohibited** unless prior written permission and a separate commercial license are obtained directly from **Owkin Inc.** (`https://www.owkin.com` / `https://huggingface.co/owkin/phikon-v2`).

### 2.9 Documented Public Release Redistribution Uncertainty (`RELEASE GATE: BLOCKED`)
During release compliance verification for `v1.1.0`, the following upstream redistribution constraint was identified and documented transparently:
- Owkin's `LICENSE.pdf` limits licensees to **non-profit entities** (`Preamble`) and defines `"Non-Commercial"` (`Section 1`) to exclude *"any use, sale, lease, license, or other transfer of the Licensed Material, Derivative Works or Results by or to a for-profit organization."*
- Publishing an un-gated public GitHub Release (`v1.1.0`) under an individual GitHub account (`PrajwalGN1`) packages tracked repository assets into public release archives (`Source code (zip/tar.gz)`)—and, if an APK binary were attached, would bundle `assets/models/fusion_model_fp32.onnx` and `assets/reference_tensors/*.bin`—without institutional non-profit sign-off or access gating against downloads by for-profit organizations.
- **Compliance Disposition:**
  1. Existing tracked repository files were **not** automatically deleted so historical reproducibility is preserved.
  2. **No** restricted model weights (`.pth`, `.onnx`), reference embedding binaries (`.bin`), or compiled `.apk` packages are attached to any new public release.
  3. **Public `v1.1.0` tag and GitHub Release publication is held `BLOCKED`** until institutional non-profit eligibility and/or written clarification from Owkin Inc. is confirmed for public release distribution of `Results`.

---

## 3. Bio_ClinicalBERT — Verified Upstream License & Attribution

### 3.1 Original Model and Official Sources
- **Model Identifier:** `emilyalsentzer/Bio_ClinicalBERT` (`BertModel`, `~108.3M` parameters, `768`-dimensional `[CLS]` summary embeddings)
- **Official Hugging Face Model Card:** https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
- **Official Source Repository & License:** https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/LICENSE
- **Integration in EdgeCRC-AI:** Dynamically loaded in `src/inference/report_encoder.py` via `AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")` and `AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")`. **Bio_ClinicalBERT model weights are not bundled inside this repository or the Android APK.**

### 3.2 Verified Upstream License (`MIT License`)
Direct inspection of both the Hugging Face Model Card YAML metadata (`license: mit`) and the upstream GitHub `LICENSE` file verifies that **Bio_ClinicalBERT** is licensed under the **MIT License**:

```text
MIT License

Copyright (c) 2019 Emily Alsentzer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.3 Provenance & Required Academic Citations
Per the official `emilyalsentzer/Bio_ClinicalBERT` model card, Bio_ClinicalBERT was initialized from **BioBERT** (`BioBERT-Base v1.0 + PubMed 200K + PMC 270K`; Lee et al., 2019) and pretrained on de-identified clinical notes from **MIMIC-III v1.4** (Johnson et al., 2016) using **BERT** code (`Apache-2.0`; Devlin et al., 2019):

```bibtex
@inproceedings{alsentzer2019publicly,
  title     = {Publicly Available Clinical {BERT} Embeddings},
  author    = {Alsentzer, Emily and Murphy, John R. and Boag, Willie and Weng, Wei-Hung and Jin, Di and Naumann, Tristan and McDermott, Matthew B. A.},
  booktitle = {Proceedings of the 2nd Clinical Natural Language Processing Workshop},
  pages     = {72--78},
  year      = {2019},
  publisher = {Association for Computational Linguistics},
  url       = {https://arxiv.org/abs/1904.03323}
}
```
