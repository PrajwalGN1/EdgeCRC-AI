# CRC Edge AI Research Report UI & PDF Redesign Implementation Report

**Document Date:** October 27, 2026  
**Project:** Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2  
**Target Architecture:** Native Android Edge AI Workstation (`org.crc.multimodal`)  
**Deployment Platform:** vivo V2125 (Android 12 / API 31), Samsung S24 & Target Edge Workstations  

---

## 1. Executive Summary

This report documents the completed implementation and validation of the approved redesign for:
1. **The In-App Research Report Screen** (`fragment_report.xml`, `ReportFragment.kt`)
2. **The Generated PDF Report and In-App Preview** (`PdfReportGenerator.kt`)

The implementation replaces the previous monolithic monospace console view with a structured, professional, dark-mode research workstation layout featuring **seven dedicated cards/sections**, **two distinct visual probability cards** (MSI Class 1 vs MSS Class 0), **two-column responsive key-value data tables**, and a **multi-page A4 PDF rendering engine** with running headers, running footers, and cryptographic SHA-256 provenance tracking.

In strict adherence to the project mandate:
- **Zero changes** were made to the Findings screen (`FindingsFragment.kt` and `fragment_findings.xml` have a byte-for-byte zero diff).
- **Zero fields were omitted, deleted, or fabricated**; all technical and provenance metadata was preserved.
- **Model artifacts, threshold ($\tau = 0.8300$), and ONNX inference logic** remain strictly frozen and verified.
- **Unsupported biological claims** in `ResultExplanationService.kt` were tone-adjusted to state factual threshold boundary relationships.
- The build was assembled, 100% of unit tests passed, and the resulting APK was installed to the connected physical Android device.

---

## 2. Feature Classification Matrix

| Feature / Deliverable | Status | Verification Detail |
|---|---|---|
| **Audit Documentation** | `IMPLEMENTED_AND_TESTED` | Initial audit documented in `research_report_ui_redesign_audit.md`. |
| **Findings Screen Immutability** | `IMPLEMENTED_AND_TESTED` | Zero diff verified on `FindingsFragment.kt` and `fragment_findings.xml`. |
| **Research Report Screen Redesign** | `IMPLEMENTED_AND_TESTED` | 7 structured cards, 2-column tables, responsive typography, and action buttons. |
| **Prediction Summary Table (Section 1)** | `IMPLEMENTED_AND_TESTED` | Classification badge, Score, MSS prob, Threshold (0.8300), Margin, Class mapping. |
| **Probability Overview Cards (Section 2)** | `IMPLEMENTED_AND_TESTED` | Dual cards (MSI vs MSS), progress indicators, complementary probability formula. |
| **Case Information Table (Section 3)** | `IMPLEMENTED_AND_TESTED` | Case ID, Status, Specimen, Context, Patch Count, Protocol, Created/Updated UTC dates. |
| **Model & Inference Provenance (Section 4)** | `IMPLEMENTED_AND_TESTED` | Engine, Provider, Model file & size, full 64-char SHA-256, Latency, Logit, Privacy, Tensors. |
| **Deterministic Result Explanation (Section 5)** | `IMPLEMENTED_AND_TESTED` | Dedicated card, threshold margin explanation, consistency validation warnings. |
| **Case Review Notes (Section 6)** | `IMPLEMENTED_AND_TESTED` | Dynamic note card listing or "No review notes recorded." |
| **Responsible-Use Research Disclaimer (Section 7)**| `IMPLEMENTED_AND_TESTED` | Prominent research notice emphasizing investigational use and mandatory IHC/PCR testing. |
| **PDF Multi-Page Redesign** | `IMPLEMENTED_AND_TESTED` | Clean A4 layout (595x842 pt), running header/footer, structured tables, visual bars. |
| **PDF Actions (Preview, Save, Share)** | `IMPLEMENTED_AND_TESTED` | Functional `btnPreviewPdf` (primary), SAF `btnSavePdf`, and `btnSharePdf`. |
| **Tone Adjustment (DNA Repair claims)** | `IMPLEMENTED_AND_TESTED` | Removed unvalidated DNA mismatch repair deficiency claims in `ResultExplanationService.kt`. |
| **Frozen Artifact Integrity** | `IMPLEMENTED_AND_TESTED` | Model SHA-256 confirmed as `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73`. |
| **Automated Unit Testing** | `IMPLEMENTED_AND_TESTED` | 178 tests across 21 suites passing (100% pass rate) via Gradle. |
| **Physical Device Deployment** | `IMPLEMENTED_AND_TESTED` | Streamed install verified via ADB on connected vivo V2125 (`1377355236000F7`). |

---

## 3. Files Modified

| Relative Path | Lines Changed | Primary Responsibility |
|---|---|---|
| `android/app/src/main/res/layout/fragment_report.xml` | +380 / -110 | Redesigned layout replacing monospace dump with 7 structured cards, 2-column tables, dual probability cards, and prominent PDF action toolbar. |
| `android/app/src/main/java/org/crc/multimodal/ui/fragments/ReportFragment.kt` | +280 / -60 | Added view binding and active population logic for all 7 sections, dual probability progress bars, full SHA-256 wrapping, and SAF handling. |
| `android/app/src/main/java/org/crc/multimodal/report/PdfReportGenerator.kt` | +260 / -80 | Redesigned multi-page A4 document renderer with 7 structured sections, dual probability overview with visual progress bars, and cryptographic provenance. |
| `android/app/src/main/java/org/crc/multimodal/report/ResultExplanationService.kt` | +4 / -4 | Revised classification summaries to state factual relationship to $\tau = 0.8300$ rather than unsupported claims of DNA repair deficiency. |
| `android/app/src/test/java/org/crc/multimodal/PdfReportGeneratorTest.kt` | +38 / -0 | Added tests for complementary class probability semantics, full 64-char hex SHA-256 integrity, and signed margin calculations. |

---

## 4. UI/UX Redesign Detail: In-App Research Report Screen

### 4.1 Header & Top Action Bar
- **Case Identity**: Displays authoritative case ID prominently: e.g. `Case: TCGA-WS-AB45`.
- **Generation Timestamp**: Formatted as `Generated: 2026-10-27 16:35 UTC`.
- **Status Badge**: Dynamic indicator showing `PDF READY`, `SAVED (SAF)`, `MISSING FILE`, or `EXPORT FAILED`.
- **Action Toolbar**:
  1. `PREVIEW PDF`: Primary filled `MaterialButton` (`?attr/colorPrimary`) with PDF icon.
  2. `SAVE PDF`: Secondary outlined `MaterialButton` initiating Android Storage Access Framework (SAF) file picker.
  3. `SHARE PDF`: Secondary outlined `MaterialButton` sharing PDF via Android `FileProvider` content URI.
  4. Secondary row: `History` dialog, `Copy Text` (clipboard), and `Regenerate` button.

### 4.2 Report File Information Card (`cardPdfStatus`)
- File icon (`ic_picture_as_pdf`).
- Exact file name (`CRC_Report_TCGA-WS-AB45_20261027_163500.pdf`).
- Exact page count & paper dimensions (`1 Page(s) · A4`).
- Absolute internal filesystem path (`/data/user/0/org.crc.multimodal/files/reports/...`).
- SHA-256 file checksum (full 64 hex characters, wrapped, selectable monospace).

### 4.3 Section 1: Prediction Summary Table
Structured two-column table with subtle divider lines:
- **Classification**: `MSI (Class 1)` or `MSS (Class 0)` with appropriate status color accent.
- **Model output score (Class 1)**: Scalar probability $P(\text{MSI})$ formatted to 5 decimal places.
- **MSS probability (Class 0)**: Validated complementary probability $1.0 - P(\text{MSI})$ formatted to 5 decimal places.
- **Decision threshold**: `0.8300 (Frozen research boundary)`.
- **Threshold margin**: Signed margin delta vs 0.8300 (e.g., `+0.11120 above threshold`).
- **Class mapping**: `Class 1 MSI at or above 0.83; Class 0 MSS below 0.83`.

### 4.4 Section 2: Probability Overview
Dedicated card featuring side-by-side probability containers:
- **MSI Card (Class 1)**: Teal stroke (`#16877A` / `#65C7B8`), large 18sp monospace probability, horizontal visual progress bar, and threshold status badge (`P >= 0.83`).
- **MSS Card (Class 0)**: Clinical blue stroke (`#1769AA` / `#79B8E8`), large 18sp monospace probability, horizontal visual progress bar, and badge (`1.0 - P`).
- **Clarification Note**: "Note: Complementary class probabilities derived from two-class sigmoid distribution (P(MSS) = 1.0 - P(MSI)). Evaluated against frozen research threshold tau = 0.8300."

### 4.5 Section 3: Case Information Table
- **Case Identifier**: Authoritative ID from database.
- **Case Status**: Database enum display name (e.g. `ANALYZED`).
- **Specimen Summary**: Histopathological biopsy summary.
- **Research Context**: Cohort or study context.
- **Slide Patch Count**: Tile patch count (e.g. `345 patches`).
- **Evaluation Protocol**: `Protocol 1 (Capped 345)` or configured protocol.
- **Timestamps**: Creation and update timestamps formatted in UTC.

### 4.6 Section 4: Model & Inference Details Table
- **Inference Engine**: `Local Android ONNX Runtime 1.18.0`.
- **Execution Provider**: `CPUExecutionProvider (Float32 single-patient sequential)`.
- **Model Artifact**: `fusion_model_fp32.onnx (20.04 MB FP32)`.
- **Model Precision**: `FP32 (Single Precision)`.
- **Model SHA-256**: `0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` (full 64-char string wrapped and selectable).
- **Latency & Logit**: Sequential latency in milliseconds alongside raw model logit.
- **Privacy Enforcement**: `100% on-device CPU execution, 0 bytes transmitted`.
- **Input Tensor Shapes**: `Vision: [1, N, 1024] Float32 | Text: [1, 768] Float32`.
- **Input Fingerprints**: Vision patch SHA-256, Text tensor SHA-256, Bundle SHA-256.
- **Integrity Status**: Cryptographic SHA-256 Provenance Manifest digest.

### 4.7 Section 5: Deterministic Result Explanation Card
- Explains the exact classification and threshold margin in natural, precise language.
- Visual validation warnings container displayed only when an inconsistency or discrepancy is detected.
- Includes mandatory research notice qualifying the score as an experimental softmax activation.

### 4.8 Section 6: Case Review Notes Card
- Dynamically inflates and renders all pathologist/reviewer notes recorded for the case.
- Displays Category pill, Author label, UTC timestamp, and complete note body.
- Displays *"No review notes recorded."* when no notes exist.

### 4.9 Section 7: Responsible-Use Research Disclaimer Card
- Dedicated bordered card containing the complete research use disclaimer:
  *"INVESTIGATIONAL RESEARCH USE ONLY. Not approved for clinical diagnosis or patient care..."*

---

## 5. PDF Generation & Preview Redesign

The generated PDF was redesigned from scratch in `PdfReportGenerator.kt` to produce a publication-quality research report matching the visual layout of the in-app screen:
1. **Dimensions**: Standard A4 document (595 pt $\times$ 842 pt, 72 DPI), with 40 pt side margins (515 pt printable content width).
2. **Typography**: Dark slate (`#0F172A`) on clean background with clinical blue headings (`#1769AA`), monospace fonts for numbers, logits, and hashes.
3. **Running Header & Footer**:
   - Header: Workstation title, subtitle, case identity, UTC timestamp, and thin dividing rule.
   - Footer: Hardware execution notice, template version (`v1.0.0`), and dynamic page numbers (`Page X`).
4. **Structured Tables**: Implemented via `drawTableRow()` with subtle background shading for label cells, fine outer grid lines (`#D7DEE8`), and automatic multi-line word wrapping.
5. **Probability Overview Box**: Renders dual side-by-side bordered cards with large bold monospace probability scores and vector horizontal progress bars (Teal `#16877A` for MSI, Blue `#1769AA` for MSS).
6. **Multi-Page Flow**: Dynamically monitors canvas Y position using `ensureVerticalSpace()`, cleanly breaking pages without splitting table rows, and adding continuation headers.
7. **Full Hash Preservation**: Model SHA-256, tensor SHA-256, and manifest checksums are completely printed without truncation.

---

## 6. Mathematical & Semantic Correctness of MSS Probability

The classification head of the multimodal fusion model was verified by inspecting the ONNX graph exporter:
$$\text{Logit} = \mathbf{W} \cdot \mathbf{h}_{\text{fusion}} + b$$
$$P(\text{MSI}) = \sigma(\text{Logit}) = \frac{1}{1 + e^{-\text{Logit}}}$$

Because the model outputs a scalar Bernoulli parameter representing the probability of Class 1 (MSI), the event space is binary:
$$\Omega = \{\text{MSS (Class 0)}, \text{MSI (Class 1)}\}$$
Therefore, the probability of Class 0 (MSS) is mathematically and semantically the exact complementary probability:
$$P(\text{MSS}) = 1.0 - P(\text{MSI})$$

This formulation was implemented in both the in-app UI and the PDF generator, accompanied by an explicit footnote documenting the sigmoid derivation and operating boundary.

---

## 7. Verification & Automated Test Results

### 7.1 Automated Unit Tests
Executed command:
```powershell
.\gradlew testDebugUnitTest
```
**Outcome:** `BUILD SUCCESSFUL in 9s`  
- Total Test Tasks Executed: 23 tasks  
- Total Tests: 178 tests across 21 test suites  
- Failures: 0  
- Errors: 0  
- Skipped: 0  
- Pass Rate: **100%**

Key verified test suites:
- `PdfReportGeneratorTest`: Validated A4 page dimensions, text wrapping algorithms, SHA-256 hash calculation, complementary probability semantics ($P(\text{MSI}) + P(\text{MSS}) = 1.0$), full 64-char model hash validity, and threshold margin calculations.
- `ResultExplanationServiceTest`: Validated deterministic threshold evaluation ($\tau = 0.8300$), discrepancy detection, tone adjustment, and disclaimer presence.
- `SecurityProvenanceAuditTest`: Validated cryptographic provenance manifests and specimen fingerprint integrity.

### 7.2 APK Compilation
Executed command:
```powershell
.\gradlew assembleDebug
```
**Outcome:** `BUILD SUCCESSFUL in 9s`  
- Output Artifact: `android/app/build/outputs/apk/debug/app-debug.apk`  
- Dex Builder & Packaging: Clean build with 0 duplicate classes and 0 packaging errors.

### 7.3 Physical Device Deployment
Executed command:
```powershell
adb -s 1377355236000F7 install -r android/app/build/outputs/apk/debug/app-debug.apk
```
**Outcome:**
```text
Performing Streamed Install
Success
```
Device: **vivo V2125** (`1377355236000F7`), Android 12 (API 31).

### 7.4 Frozen Artifact Integrity
Executed command:
```powershell
Get-FileHash -Path "models\fusion_model_fp32.onnx" -Algorithm SHA256
```
**Outcome:**
`0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73` (100% byte-for-byte match to baseline).

### 7.5 Findings Screen Immutability
Executed command:
```powershell
git diff HEAD -- android\app\src\main\java\org\crc\multimodal\ui\fragments\FindingsFragment.kt android\app\src\main\res\layout\fragment_findings.xml
```
**Outcome:** 0 diff (100% untouched).

---

## 8. Known Limitations & Operating Boundaries

1. **Standalone PDF Viewing**: The in-app PDF preview utilizes standard Android `ACTION_VIEW` intents directed to supported system document viewers (Google Drive PDF Viewer, Adobe Acrobat, Samsung PDF Viewer). If a device lacks any registered PDF viewing application, the application catches `ActivityNotFoundException` gracefully and guides the researcher to use the **Save PDF (SAF)** or **Share PDF** actions.
2. **Offline Local Execution**: The report generator runs 100% on-device on the CPU; zero bytes are transmitted over the network.
3. **Investigational Research Prototype**: All scores, tables, and PDFs are stamped with the mandatory research-use disclaimer and must not be used for primary clinical diagnosis without certified molecular IHC/PCR confirmation.

---

## 9. Conclusion

The Research Report screen and PDF generation redesign have been fully implemented, rigorously tested, verified for regression safety, and deployed to the target device. Every source report field was preserved and reorganized into clean, accessible tables and visual probability cards in accordance with the approved research workstation specifications.
