# EdgeCRC-AI v1.1.0 — Android Edge Inference Research Workstation Release Notes

> [!WARNING]
> **INVESTIGATIONAL COMPUTATIONAL PATHOLOGY SOFTWARE — NOT FOR CLINICAL USE**
> **EdgeCRC-AI** is an academic computational pathology research prototype for investigating Microsatellite Instability (`MSI`) versus Microsatellite Stability (`MSS`) in colorectal adenocarcinoma. It has **not** been cleared, certified, or approved by the FDA, EMA, CE-IVDR, CDSCO, or any regulatory authority, and must **never** be used for clinical diagnosis, patient triage, or treatment selection.

---

## 1. Feature Summary

**EdgeCRC-AI `v1.1.0`** (`applicationId = "org.crc.multimodal"`, `versionName = "1.1.0"`, `versionCode = 4`) integrates the complete offline **Android ONNX Runtime Edge AI Research Workstation** with the upstream **Python Multimodal Cross-Attention Training, Explainability & Specimen Export Pipeline** on `main`:

- **13-Screen Native Android Research Workstation** with Material Design 3 Day/Night theming, custom `Canvas` decision-boundary visualization (`ProbabilityThresholdBarView`), and a 4-tab navigation rail (`Workspace`, `Cases`, `New Analysis`, `Settings`).
- **Dual Local SQLite Databases (`crc_workstation.db` v4 & `crc_batch_runs.db` v3)** with additive, non-destructive schema migrations and full audit/provenance logging.
- **Cryptographically Checked Multimodal Specimen Ingestion** supporting `.crcbundle` (`CRCB` v2), `.crcmanifest.json` (`v2.0`), paired NumPy `.npy` tensors (`v1.0`/`v2.0`), and seeded `CRCT` v1 reference specimens.
- **100-Specimen Batch Analysis Coordinator** with isolated per-specimen error handling, automatic Protocol 2 routing for `N > 1000` patches, and bulk **EXPORT PDF**, **ARCHIVE / RESTORE**, and **DELETE** management controls.
- **66-Case TCGA-COAD Holdout Cohort Evaluation Engine** computing confusion matrices, clinical classification metrics, trapezoidal **ROC-AUC**, and step-wise **PR-AUC** completely offline.
- **Cohort Research Analytics & Margin Profiling** with probability histograms, near-threshold (`|P - 0.83| <= 0.05`) margin brackets, latency/patch distributions, and unified specimen filtering.
- **Multi-Page A4 PDF Research Reporting** (`PdfReportGenerator` v2.0.0 & `BatchReportGenerator`) with SHA-256 document digests, versioned `report_history`, SAF export, and `FileProvider` sharing.
- **Formal Non-Commercial Source-Available & Upstream Model Licensing Architecture** (`LICENSE`, `MODEL_LICENSE.md`, `DATA_ATTRIBUTION.md`, `THIRD_PARTY_NOTICES.md`).

---

## 2. Android Architecture

- **Execution Split (Upstream Encoders vs. On-Device Cross-Attention Fusion):**
  - **Upstream Python Pipeline (`src/`, `tools/`):** Frozen **Owkin Phikon-v2** (`ViT-L/16`, `~307M` params) extracts `[1, N, 1024]` patch embeddings from H&E histopathology tiles, and frozen **Bio_ClinicalBERT** (`~108.3M` params) extracts `[1, 768]` `[CLS]` embeddings from clinical pathology text.
  - **On-Device Android Workstation (`android/`):** Loads precomputed Float32 feature bundles and executes the 8-head **Text-to-Patch Cross-Attention Fusion Network** (`SummaryToPatchCrossAttention`, `5,250,561` trainable parameters) locally on CPU via **ONNX Runtime Android `1.18.0`**.
- **Zero-Permission Air-Gapped Security:** `AndroidManifest.xml` declares **zero `<uses-permission>` entries** (no `android.permission.INTERNET`), disables cloud backup (`android:allowBackup="false"`), scopes `FileProvider` strictly to internal `reports/` directories, enforces canonical Zip-Slip/file-size/NaN/Inf checks (`SafeFileValidator`), and scrubs sensitive tokens from logs (`SensitiveLogRedactor`).
- **Dual SQLite Storage:**
  - `crc_workstation.db` (`v4`, `8` tables): `cases`, `specimen_inputs`, `inference_runs`, `review_notes`, `research_reports`, `report_history`, `reference_labels`, `audit_trail`.
  - `crc_batch_runs.db` (`v3`, `2` tables): `batch_runs`, `batch_items`.

---

## 3. ONNX Inference Contract

| Direction | Tensor Name | Runtime Shape | Data Type | Specification |
| :--- | :--- | :--- | :--- | :--- |
| **Input 0** | `vision_seq` | `[1, N, 1024]` | `FLOAT` (`FP32`) | Dynamic patch sequence (`N >= 1`), finite-checked |
| **Input 1** | `text_summary` | `[1, 768]` | `FLOAT` (`FP32`) | Clinical `[CLS]` embedding vector, finite-checked |
| **Output 0** | `logit` | `[1, 1]` | `FLOAT` (`FP32`) | Pre-sigmoid scalar score |
| **Output 1** | `probability` | `[1, 1]` | `FLOAT` (`FP32`) | In-graph sigmoid `P(MSI)` in `[0.0, 1.0]` |
| **Output 2** | `attention_weights` | `[1, 1, N]` | `FLOAT` (`FP32`) | Head-averaged cross-attention weights summing to `1.0` |

- **Bundled Model Asset:** `fusion_model_fp32.onnx` (`21,015,452` B, uncompressed for direct memory mapping; SHA-256: `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`).
- **Frozen Operating Threshold:** **`0.83`** (`P >= 0.83 -> Class 1 [MSI]`; `P < 0.83 -> Class 0 [MSS]`).
- **Evaluation Protocols:**
  - **Protocol 1 (`PROTOCOL_1_CAPPED`):** Accepts `1 <= N <= 1000` patches; explicitly rejects `N > 1000` with zero silent truncation.
  - **Protocol 2 (`PROTOCOL_2_UNCAPPED`):** Accepts full sequence `N >= 1` using dynamic ONNX sequence axes (`batch_size`, `num_patches`).

---

## 4. Specimen Ingestion

- **`.crcbundle` (`CRCB` v2 Binary Container):** Upstream exporter (`tools/export_specimen_bundle.py`) and Android `AuthenticBundleLoader` verify Little-Endian `CRCB` magic headers, JSON metadata, and per-modality SHA-256 digests over raw Float32 tensor payloads before execution.
- **`.crcmanifest.json` (`v2.0` JSON Bundle):** Human-inspectable UTF-8 JSON tensor representation.
- **Paired `.npy` Arrays (`*_vision.npy` + `*_text.npy`):** `NpyTensorLoader` parses NumPy `v1.0`/`v2.0` headers (`<f4`, `fortran_order=False`) imported via Android's Storage Access Framework (`SAF`).
- **Seeded `CRCT` v1 Reference Specimens:** First launch seeds `TCGA-D5-6540` (`MSI`, `N=345`) and `TCGA-CM-6675` (`MSS`, `N=541`) for immediate offline verification; `7` golden `CRCT` benchmark cases are bundled under `assets/reference_tensors/`.

---

## 5. Batch Processing

- **`BatchCoordinator` & `BatchRepository`:** Processes up to `100` imported `.crcbundle` files sequentially per run on a background executor with live progress callbacks and isolated per-specimen failure capture.
- **Automatic Protocol 2 Escalation:** Specimens with `N > 1000` patches automatically execute under **Protocol 2 (Uncapped)** with recorded protocol provenance (`protocol_reason`).
- **Specimen Traceability & Bulk Actions:** Filter chips (`All`, `MSI Predicted`, `MSS Predicted`, `Rejected/Failed`, `Archived`), live search, and multi-select **EXPORT PDF**, **ARCHIVE / RESTORE**, and **DELETE** controls.

---

## 6. Cohort Evaluation

- **`CohortEvaluationEngine`:** Matches batch or workstation predictions against the bundled `66`-case TCGA-COAD holdout ground-truth table (`assets/reference_labels/tcga_holdout_reference_labels.json`: `50` `MSS`, `16` `MSI`) using canonical TCGA barcode normalization.
- **Verified Metrics:** Computes `2x2` confusion matrix (`TP`, `TN`, `FP`, `FN`), Accuracy, Balanced Accuracy, Precision (`PPV`), Sensitivity (`Recall`), Specificity (`TNR`), F1 Score, `NPV`, trapezoidal **ROC-AUC**, and step-wise **PR-AUC** (`Average Precision`).

---

## 7. Analytics

- **`ResearchAnalyticsEngine`:** Aggregates workstation and batch runs into cohort MSI/MSS prediction ratios, 5-bin probability distribution histograms, near-threshold (`|P - 0.83| <= 0.05`) margin profiling, CPU latency/patch statistics, and a filterable Specimen Prediction Explorer.

---

## 8. PDF Reporting

- **Single-Case Reports (`PdfReportGenerator` v2.0.0):** Multi-page A4 (`595 x 842 pt`) PDF reports generated via `android.graphics.pdf.PdfDocument`, capturing predicted class, `P(MSI)`, signed threshold margin (`Delta = P - 0.83`), top-attended patch indices, protocol metadata, SHA-256 integrity footers, and non-clinical research disclaimers, tracked across revisions in `report_history`.
- **Batch & Cohort Reports (`BatchReportGenerator`):** Compiles multi-specimen batch summaries, confusion matrices, and per-specimen tables into exportable PDF and JSON research packages.

---

## 9. Technical Requirements

- **Development / Build Host:** Windows 10/11, Linux, or macOS; **JDK 17**; **Android SDK Platform 34** (`Build-Tools 34.0.0`); Gradle `8.10.2` wrapper; Python `3.10`–`3.13` for upstream tooling and `pytest`.
- **Android Target Device:** Android 7.0 (`API 24`) through Android 14+ (`API 34+`); ABIs `arm64-v8a`, `armeabi-v7a`, `x86_64`.

```powershell
git clone https://github.com/PrajwalGN1/EdgeCRC-AI.git
Set-Location EdgeCRC-AI\android
.\gradlew.bat assembleDebug
adb install -r "app\build\outputs\apk\debug\app-debug.apk"
```

---

## 10. Regression Evidence

- **Holdout Benchmark Parity (`66` TCGA-COAD Cases, Protocol 1, `tau = 0.83`):** Accuracy `89.39%` (`TP=12, TN=47, FP=3, FN=4`), MSI Precision `0.8000`, MSI Sensitivity `0.7500`, Specificity `0.9400`, F1 `0.7742`, ROC-AUC `0.92625`, PR-AUC `0.82205`.
- **ONNX Numerical Parity (`7/7` Golden `CRCT` Cases + Dynamic `N=1500`/`N=4305`):** `max_logit_diff = 9.54e-07`, `max_prob_diff = 4.62e-07`, **`0` classification label flips**; root and Android `fusion_model_fp32.onnx` verified bit-identical (`0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`); `fusion_model_v1_1.pth` verified unchanged (`e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453`).
- **Python Test Suite (`python -m pytest tests -v`):** **`66/66` passed** across `9` test modules (`0` failures).
- **Android Build & JVM Unit Tests (`assembleDebug` & `testDebugUnitTest`):** `assembleDebug` **`PASS`** (`versionCode = 4`, `versionName = "1.1.0"`, `113,023,985` B debug APK); `testDebugUnitTest` **`229/229` passed** across `23` JUnit suites (`0` failures). *(Connected physical-device `androidTest` instrumentation suites were `NOT RUN` during headless release packaging.)*

---

## 11. Model & Data Attribution

- **Author:** **Prajwal G N** (`https://github.com/PrajwalGN1/`)
- **Academic Guidance:** Mr Nageshwara Guptha
- **Owkin Phikon-v2 (`owkin/phikon-v2`):** Filiot et al. (2024), *Phikon-v2, A large and public feature extractor for biomarker prediction* (`https://huggingface.co/owkin/phikon-v2`).
- **Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`):** Alsentzer et al. (2019), *Publicly Available Clinical BERT Embeddings*, NAACL Clinical NLP Workshop (`https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT`).
- **TCGA Research Network:** *"The results shown here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga."*
- **ONNX Runtime:** Microsoft Corporation (`onnxruntime` / `onnxruntime-android` `1.18.0`, MIT License).

---

## 12. License Details

EdgeCRC-AI uses a component-scoped non-commercial licensing architecture (it is **source-available under non-commercial terms, not OSI-approved open source**, and is **not** uniformly licensed under a single file):

1. **Original Software Code ([`LICENSE`](../../LICENSE)):** Licensed by `Copyright (c) 2026 Prajwal G N` under the **PolyForm Noncommercial License 1.0.0** (`https://polyformproject.org/licenses/noncommercial/1.0.0`), scoped strictly to original project-authored software code.
2. **Derived Model Checkpoints, ONNX Graphs & Embeddings ([`MODEL_LICENSE.md`](../../MODEL_LICENSE.md)):** Phikon-v2-derived `Results` (`models/*.pth`, `models/*.onnx`, `android/app/src/main/assets/models/*.onnx`, and `android/app/src/main/assets/reference_tensors/*.bin`) are governed **exclusively** by the upstream **Owkin Phikon-v2 License Terms and Conditions** (`https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf` — non-commercial research by non-profit entities only; ShareAlike; no additional/conflicting PolyForm terms applied; separate written Owkin Inc. permission required for any commercial or for-profit use). Bio_ClinicalBERT is governed by the **MIT License**.
3. **Dataset Attribution ([`DATA_ATTRIBUTION.md`](../../DATA_ATTRIBUTION.md)):** Governed by **NIH/NCI TCGA Open-Access Data Tier Guidelines** (attribution required; re-identification attempts strictly prohibited).
4. **Third-Party Libraries ([`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)):** All Python and Android dependencies retain their respective upstream licenses.

*(Note: Restricted model weights, `.bin` tensors, and compiled `.apk` binaries bundling Phikon-v2-derived `Results` must not be attached to un-gated public releases unless institutional non-profit / Owkin redistribution clearance is verified; see [`MODEL_LICENSE.md`](../../MODEL_LICENSE.md) Section 2.9.)*

---

## 13. Research-Use Disclaimer

EdgeCRC-AI is an **investigational academic research prototype** provided *"as is"* without warranty of any kind. It is **not** clinically validated, certified, or approved as an in-vitro diagnostic (`IVD`) or Software as a Medical Device (`SaMD`), and its outputs must **never** be used to diagnose disease, guide immunotherapy or chemotherapy decisions, or replace clinical mismatch-repair (`MMR`) Immunohistochemistry (`IHC`), `PCR`, or `NGS` testing.
