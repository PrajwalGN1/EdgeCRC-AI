# Research Report Screen & PDF Redesign — Architectural & UI/UX Audit

**Project**: Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2  
**Target Areas**: (1) In-App Research Report Screen, (2) Generated PDF Report & Preview  
**Status**: AUDIT COMPLETE — PROCEEDING WITH IMPLEMENTATION  
**Date**: October 27, 2026  
**Auditor**: Senior Android UI/UX Engineer, Kotlin Developer, PDF-Generation Engineer, Software Quality Auditor  

---

## 1. Executive Summary & Scope Boundary Enforcement

In strict compliance with Section 1 of the Master Directive:
- **FINDINGS SCREEN IS COMPLETELY PRESERVED**: `FindingsFragment.kt` and `fragment_findings.xml` remain 100% untouched. No colors, typography, buttons, margins, or prediction cards on the Findings screen are modified.
- **RESTRICTED REDESIGN SCOPE**:
  1. **Research Report Screen**: `ReportFragment.kt` and `fragment_report.xml`
  2. **PDF Report Template & Preview**: `PdfReportGenerator.kt`
  3. **Result Explanation Tone Correction**: `ResultExplanationService.kt` (only unsupported clinical claims removed)
- **ZERO LOSS OF REPORT DATA**: Every existing technical field currently emitted in the plaintext console view is preserved and mapped into human-readable, accessible, structured two-column tables and dedicated cards.

---

## 2. Inventory of Current Components and File Locations

| Component | File Path | Current Role | Planned Redesign Action |
|---|---|---|---|
| **Research Report Layout** | `android/app/src/main/res/layout/fragment_report.xml` | Monospace console `tvReportContentText` inside card | Redesign with 7 structured cards, 2-column key-value tables, and dual probability cards |
| **Research Report Fragment** | `android/app/src/main/java/org/crc/multimodal/ui/fragments/ReportFragment.kt` | Loads data, builds string, launches PDF actions | Bind new structured sections, format data rows, maintain SAF save/preview/share |
| **PDF Report Generator** | `android/app/src/main/java/org/crc/multimodal/report/PdfReportGenerator.kt` | Native Android `PdfDocument` renderer | Redesign A4 layout with structured tables, probability overview, section numbers, full metadata |
| **Result Explanation Engine** | `android/app/src/main/java/org/crc/multimodal/report/ResultExplanationService.kt` | Deterministic explanation text | Tone adjustment to remove unsupported clinical DNA repair claims |
| **Plaintext Report Utility** | `android/app/src/main/java/org/crc/multimodal/report/ReportGenerator.kt` | Canonical plaintext string generator | Preserved as backing data formatter and for clipboard copy |
| **Navigation Host** | `android/app/src/main/java/org/crc/multimodal/ui/WorkstationNavigationHost.kt` | Interface for screen transitions | Preserved (`navigateToReport(caseId, runId)`) |
| **Main Activity** | `android/app/src/main/java/org/crc/multimodal/ui/MainActivity.kt` | Fragment container & toolbar host | Preserved |
| **Unit Test Suite** | `android/app/src/test/java/org/crc/multimodal/PdfReportGeneratorTest.kt` | Tests wrapping, constants, SHA-256 | Extend with section coverage and table formatting tests |

---

## 3. Data Model & Semantics Audit

### 3.1 Case and Inference Entities
- `ResearchCase`: `caseIdentifier`, `specimenDescription`, `researchContext`, `status` (`CaseStatus`), `patchCount`, `protocol` (`InferenceProtocol`), `createdAt`, `updatedAt`.
- `InferenceRun`: `predictedClass` (0 or 1), `predictedLabel` ("MSS" or "MSI"), `probability` ($[0.0, 1.0]$), `logit`, `threshold` (0.83f), `margin` ($\text{probability} - 0.83$), `latencyMs`, `numPatches`, `diagnosticsSummary`, `executionTimestamp`.
- `ReviewNote`: `category`, `authorLabel`, `content`, `createdAt`.
- `AuthenticFeatureBundle`: `specimenId`, `description`, `patchCount`, `visionEmbedding`, `textEmbedding`, `visionSha256`, `textSha256`, `schemaVersion`.
- `PersistentReportRecord`: `reportId`, `filename`, `localFilePath`, `sha256Hash`, `pageCount`, `exportStatus`.

### 3.2 MSS Probability Mathematical Provenance & Verification
- **Model Graph Verification**: Inspection of `models/fusion_model_fp32.onnx` and `CrossAttentionExportWrapper` in `src/eval/export_onnx.py`:
  - Classifier head: `classifier[3] = Linear(256, 1)`, `classifier[4] = Sigmoid()`.
  - The model outputs a single scalar $\sigma(\text{logit}) \in [0.0, 1.0]$ representing Class 1 ($P(\text{MSI})$).
- **Complementary Class Semantics**:
  - In a standard binary logistic Bernoulli model, the class probabilities partition the sample space:
    $$P(\text{MSI}) = \sigma(\text{logit})$$
    $$P(\text{MSS}) = 1.0 - P(\text{MSI}) = \sigma(-\text{logit})$$
  - Therefore, $P(\text{MSS})$ is mathematically valid and complementary ($P(\text{MSI}) + P(\text{MSS}) = 1.0$).
  - **Qualification Required**: The report and UI will explicitly state: *"Model outputs single-logit Sigmoid activation score; MSS is the complementary probability $P(\text{MSS}) = 1.0 - P(\text{MSI})$."*

### 3.3 Explanatory Text Tone Revision
- `ResultExplanationService.kt` currently states:
  *"MSI (Microsatellite Instability) — High probability of DNA mismatch repair deficiency indicated by multimodal fusion."*
- **Audit Revision**: Replace unsupported biological claims with strict empirical model evaluation statements:
  *"MSI (Microsatellite Instability) — Evaluated model output score is at or above the research decision threshold (tau = 0.8300)."*
  *"MSS (Microsatellite Stable) — Evaluated model output score is below the research decision threshold (tau = 0.8300)."*

---

## 4. UI/UX Hierarchy & Screen Architecture (Research Report Screen)

The redesigned `fragment_report.xml` replaces the monospace dump with a structured, high-contrast, modern dark-mode card flow:

1. **Top Header & Status Bar**:
   - Case ID: `Case: <caseIdentifier>`
   - Subtitle: `Generated <timestamp>`
   - Status Badge: `PDF READY` (visible when PDF exists) or `READY`
2. **Action Buttons**:
   - `PREVIEW PDF` (Filled primary button, `@drawable/ic_picture_as_pdf`)
   - `SAVE PDF` (Outlined secondary button, `@drawable/ic_file_download`)
   - `SHARE PDF` (Outlined secondary button, `@drawable/ic_share`)
   - `HISTORY` & `COPY` buttons retained for full functional parity
3. **Report File Metadata Card**:
   - PDF icon, actual filename, page count (`1 Page (A4)`), paper dimensions (`A4: 595 × 842 pt`), actual file path, SHA-256 digest
4. **Section 1: Prediction Summary**:
   - Structured 2-column table: Classification badge, Model output score (Class 1), MSS probability (Class 0), Decision threshold (0.8300), Threshold margin, Class mapping
5. **Section 2: Probability Overview**:
   - Dual side-by-side cards:
     - Card A: `MSI Probability (Class 1)` in clinical teal (`#65C7B8` / `#16877A`) with visual bar
     - Card B: `MSS Probability (Class 0)` in clinical blue (`#79B8E8` / `#1769AA`) with visual bar
     - Complementary probability notation
6. **Section 3: Case Information**:
   - 2-column table: Case ID, Case Status, Specimen Summary, Research Context, Patch Count, Protocol, Timestamps
7. **Section 4: Model and Inference Details**:
   - 2-column table: Engine (`ONNX Runtime 1.18.0`), Execution Provider (`CPUExecutionProvider`), Model Artifact (`fusion_model_fp32.onnx`), Model Precision (`FP32`), Model SHA-256 (full text wrapped/selectable), Tensor Shapes, Text/Vision Fingerprints, Latency, Privacy
8. **Section 5: Deterministic Result Explanation**:
   - Clean explanation card with threshold rationale and research notice
9. **Section 6: Case Review Notes**:
   - Review notes list or `"No review notes recorded."`
10. **Section 7: Responsible-Use Research Disclaimer**:
    - Complete standard investigational research disclaimer card

---

## 5. PDF Template Redesign Architecture

The redesigned `PdfReportGenerator.kt` will render:
1. **Header & Branding**: "CRC MULTIMODAL PATHOLOGY RESEARCH WORKSTATION", subtitle, Report ID, Generated Date, running header divider.
2. **Investigational Warning Banner**: Full width, light tint background, clear warning title.
3. **Section 1 — Prediction Summary**: Clean table with alternating row background, bold field labels, classification badge.
4. **Section 2 — Probability Overview**: Distinct dual probability boxes for MSI ($P$) and MSS ($1-P$), visual gauge representation, complementary probability note.
5. **Section 3 — Case & Specimen Specification**: Two-column key-value table.
6. **Section 4 — Model Provenance & Technical Details**: Complete table including model file size, exact SHA-256 fingerprint, runtime engine, execution provider, tensor dimensions, latency, privacy statement.
7. **Section 5 — Deterministic Result Explanation**: Paragraphs wrapped cleanly, margin explanation.
8. **Section 6 — Case Review Notes**: Notes entries or empty notice.
9. **Section 7 — Responsible-Use Research Disclaimer**: Complete research disclaimer.
10. **Running Footer**: Page numbering ("Page X of Y"), template version, timestamp.

---

## 6. Implementation Risk & Blocker Assessment
- **Blockers**: None. Build compiles cleanly, existing 21 test suites pass (175 tests).
- **Risks**:
  - Long SHA-256 strings clipping in tables: Mitigated by `android:fontFamily="monospace"` and `break-word` / wrapping in both UI XML and PDF text wrapping.
  - Narrow screen button wrapping: Mitigated by `HorizontalScrollView` or flexible wrapping layout for action buttons.
  - Findings screen regression: Guarded by leaving `FindingsFragment.kt` and `fragment_findings.xml` completely untouched.

**VERDICT**: AUDIT APPROVED. Proceeding immediately to implementation.
