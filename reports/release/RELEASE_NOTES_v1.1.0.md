# EdgeCRC-AI v1.1.0 — Android Edge Inference Research Workstation Release

> [!WARNING]
> **INVESTIGATIONAL COMPUTATIONAL PATHOLOGY SOFTWARE — NOT FOR CLINICAL USE**
> EdgeCRC-AI is an academic research workstation for multimodal colorectal cancer (`MSI` vs. `MSS`) biomarker investigation. It is **not** a certified or FDA/CE/CDSCO-cleared medical device and must **never** be used for patient diagnosis, prognosis, or treatment selection.

---

## 1. Release Overview

**EdgeCRC-AI `v1.1.0`** delivers the complete offline **Android ONNX Runtime Edge AI Research Workstation** alongside the verified **Python Multimodal Cross-Attention Research & Bundle Export Pipeline**, accompanied by comprehensive source-verified documentation across both `main` and `edge-deployment`.

*(Versioning Note: Following earlier Android releases `v0.1.0`, `v1.0.1`, and `v1.0.2`, `v1.1.0` marks the minor feature expansion from the single-screen `v1.0.2` prototype to the full 13-screen SQLite-backed multi-case, batch, cohort-benchmarking, analytics, and A4 PDF reporting workstation. Current `android/app/build.gradle.kts` carries `versionName = "1.0.2"`, `versionCode = 3`; bump to `"1.1.0"` / `versionCode = 4` if publishing a freshly compiled release APK asset, or tag as `v1.0.3` if retaining `1.0.2` series numbering.)*

---

## 2. Major Features Added Since `v1.0.2`

1. **13-Screen Native Android Research Workstation (`org.crc.multimodal`):**
   - Material Design 3 Day/Night interface built with Android Views (`Fragment` + XML + custom `Canvas` `ProbabilityThresholdBarView`) and a 4-tab bottom navigation rail (`Workspace`, `Cases`, `New Analysis`, `Settings`).
2. **Cryptographically Verified Specimen Ingestion (`.crcbundle` `CRCB` v2 & `.npy`):**
   - Upstream exporter (`tools/export_specimen_bundle.py`) and on-device parser (`AuthenticBundleLoader`, `NpyTensorLoader`) supporting Little-Endian `CRCB` v2 binary bundles (with per-modality SHA-256 verification), `CRCT` v1 benchmark tensors, JSON v2.0 manifests, and NumPy `.npy` v1.0/v2.0 tensor pairs via Storage Access Framework (`SAF`).
3. **Dual SQLite Persistence & Non-Destructive Schema Migrations:**
   - `crc_workstation.db` (`v4`, `8` tables: `cases`, `specimen_inputs`, `inference_runs`, `review_notes`, `research_reports`, `report_history`, `reference_labels`, `audit_trail`) and `crc_batch_runs.db` (`v3`, `2` tables: `batch_runs`, `batch_items`).
4. **Batch Specimen Analysis (`BatchCoordinator`):**
   - Sequential evaluation of up to `100` `.crcbundle` specimens per batch with automatic routing to **Protocol 2 (Uncapped)** when `N > 1000`, status filter chips, and multi-select bulk **EXPORT PDF**, **ARCHIVE** / **RESTORE**, and **DELETE** actions.
5. **66-Case TCGA Holdout Cohort Evaluation (`CohortEvaluationEngine`):**
   - Built-in TCGA-COAD holdout ground-truth loader (`tcga_holdout_reference_labels.json`), canonical patient/specimen ID matching, 2x2 confusion matrix (`TP`, `TN`, `FP`, `FN`), Accuracy, Balanced Accuracy, Precision, Sensitivity, Specificity, F1, NPV, trapezoidal **ROC-AUC**, and step-wise **PR-AUC** (verified against `scikit-learn`).
6. **Research Analytics & Margin Profiling (`ResearchAnalyticsEngine`):**
   - Cohort MSI/MSS prediction ratios, 5-bin probability histogram, near-threshold (`|P - 0.83| <= 0.05`) margin brackets, latency/patch distributions, and unified Specimen Prediction Explorer.
7. **On-Device A4 PDF Research Report Compilation (`PdfReportGenerator` v2.0.0 & `BatchReportGenerator`):**
   - Native `android.graphics.pdf.PdfDocument` multi-page A4 reports with SHA-256 provenance footers, versioned `report_history`, SAF export, and `FileProvider` sharing.

---

## 3. Android & Multimodal Architecture Summary

- **Clear Encoder vs. Fusion Execution Boundary:**
  - **Upstream (Python / GPU Workstation):** Frozen **Owkin Phikon-v2** (`ViT-L/16`, `~307M` params) extracts `[1, N, 1024]` patch embeddings from H&E tiles, and frozen **Bio_ClinicalBERT** (`~108.3M` params) extracts `[1, 768]` `[CLS]` embeddings from pathology reports.
  - **On-Device (Android Edge Workstation):** Imports precomputed Float32 feature bundles and executes the exported 8-head **Text-to-Patch Cross-Attention Fusion Network** (`SummaryToPatchCrossAttention`, `5,250,561` trainable parameters) locally via **ONNX Runtime Android `1.18.0` (`CPUExecutionProvider`)**.
- **Air-Gapped Privacy:** `AndroidManifest.xml` declares **zero `<uses-permission>` tags** (no `INTERNET` permission; `android:allowBackup="false"`).

---

## 4. ONNX Inference Contract & Protocols

| Direction | Tensor Name | Shape | Data Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Input 0** | `vision_seq` | `[1, N, 1024]` | `FLOAT` (FP32) | Dynamic patch count `N >= 1`; checked finite |
| **Input 1** | `text_summary` | `[1, 768]` | `FLOAT` (FP32) | Clinical summary vector; checked finite |
| **Output 0** | `logit` | `[1, 1]` | `FLOAT` (FP32) | Pre-sigmoid scalar |
| **Output 1** | `probability` | `[1, 1]` | `FLOAT` (FP32) | In-graph sigmoid `P(MSI)` in `[0.0, 1.0]` |
| **Output 2** | `attention_weights` | `[1, 1, N]` | `FLOAT` (FP32) | Head-averaged spatial patch attention weights |

- **Bundled Model Asset:** `fusion_model_fp32.onnx` (`21,015,452` bytes, SHA-256: `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`).
- **Frozen Operating Threshold:** **`0.83`** (`P >= 0.83 -> Class 1 [MSI]`; `P < 0.83 -> Class 0 [MSS]`).
- **Evaluation Protocols:**
  - **Protocol 1 (Capped):** `1 <= N <= 1000` patches (explicitly rejects `N > 1000` with zero silent truncation).
  - **Protocol 2 (Uncapped):** Full sequence `N >= 1` using dynamic ONNX sequence axes.

---

## 5. Supported Specimen Import Formats

1. **`.crcbundle` (`CRCB` v2 Binary Container)** — Recommended single-file format with per-modality SHA-256 verification.
2. **`.crcmanifest.json` (`v2.0` JSON Feature Bundle)** — UTF-8 JSON tensor bundle for human-readable inspection.
3. **Paired `.npy` Arrays (`*_vision.npy` + `*_text.npy`)** — Little-Endian C-contiguous `float32` NumPy arrays (`v1.0` / `v2.0`).
4. **Built-in `CRCT` v1 Reference Cases** — `TCGA-D5-6540` (MSI, `N=345`) and `TCGA-CM-6675` (MSS, `N=541`) auto-seeded on first launch.

---

## 6. Technical Requirements & Build Instructions

- **Build Host:** JDK `17`, Android SDK Platform `34`, Gradle `8.10.2` wrapper.
- **Target Android Device:** Android 7.0 (`API 24`) through Android 14+ (`API 34+`); `arm64-v8a`, `armeabi-v7a`, or `x86_64`.

```powershell
git clone https://github.com/PrajwalGN1/EdgeCRC-AI.git
Set-Location EdgeCRC-AI\android
.\gradlew.bat assembleDebug
adb install -r "app\build\outputs\apk\debug\app-debug.apk"
```

---

## 7. Validation Evidence

- **Holdout Benchmark Parity (`66` TCGA-COAD Cases, Protocol 1, `tau = 0.83`):** Accuracy `89.39%` (`TP=12, TN=47, FP=3, FN=4`), MSI Precision `0.80`, MSI Sensitivity `0.75`, Specificity `0.94`, F1 `0.7742`, ROC-AUC `0.92625`, PR-AUC `0.82205`.
- **PyTorch FP32 vs. ONNX Runtime Parity:** `0` classification label flips across all `66` holdout cases (`max probability diff < 3e-6`).
- **Automated Test Suites:** `23` Android JVM unit test suites (`229/229` tests `PASS`), `9` Python `pytest` modules (`66/66` tests `PASS`), and `7` physical-device Android instrumentation test suites (`androidTest/`, `NOT RUN` during headless release packaging).

---

## 8. Known Limitations

- Raw Whole Slide Image (`.svs`/`.tif`) tiling/Phikon-v2 extraction and raw free-text Bio_ClinicalBERT encoding run upstream in Python and are **not** executed on the Android device.
- The optional *"Remote Inference Server Base URL"* field in `Settings` is an unwired UI placeholder; all inference runs 100% locally.
- Theme selection (`System` / `Light` / `Dark`) applies to the active app process session and resets on cold process restart.

---

## 9. Attribution & License Status

- **Author:** **Prajwal G N** (`https://github.com/PrajwalGN1/`)
- **Academic Guidance:** Mr Nageshwara Guptha
- **Upstream Attributions:**
  - **Owkin Phikon-v2** (`owkin/phikon-v2`, Filiot et al.) — *Owkin Non-Commercial License* (restricts derived features and trained fusion weights to non-commercial academic research).
  - **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`, Alsentzer et al.) — *MIT License*.
  - **Cohort Data:** *The results shown here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.*
  - **Inference Runtime:** Microsoft *ONNX Runtime* (*MIT License*).
- **Repository License Status:** Pending explicit `LICENSE` file addition by the repository owner (all rights reserved by default until a root `LICENSE` file is committed; Phikon-v2-derived model weights are restricted to non-commercial academic research).
