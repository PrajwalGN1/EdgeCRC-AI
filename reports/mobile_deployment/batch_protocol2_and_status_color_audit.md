# AUDIT REPORT: CRC EDGE AI BATCH ANALYSIS STATUS COLORS AND PROTOCOL 2 SUPPORT

**Document Path**: `reports/mobile_deployment/batch_protocol2_and_status_color_audit.md`  
**Execution Timestamp**: 2026-10-27T18:41:00+05:30  
**Target Device**: vivo V2125 (`1377355236000F7`, Android 12, API 31)  
**Application Package**: `org.crc.multimodal` (VersionCode: `3`, VersionName: `1.0.2`)  
**Audit Status**: **`AUDIT_COMPLETE`**  

---

## 1. Executive Summary

This audit investigates two operational issues observed during multi-specimen batch evaluation on the CRC Multimodal Pathology Edge AI Workstation:

1. **Status Badge Color Conflation**: The batch specimen results screen displayed MSI classifications, processing failures, and policy rejections with the identical red color (`@color/status_failed`), failing to provide visual distinction between diagnostic findings, execution errors, and administrative exclusions.
2. **Batch Protocol 1 Cap Failure on Large Slides**: Authentic specimen bundles with more than 1,000 slide patches failed during batch inference with `IllegalArgumentException: Protocol 1 enforces a maximum cap of 1000 patches, but received N`. Specifically:
   - `TCGA-NH-A50T`: 2,296 patches -> failed under Protocol 1.
   - `TCGA-NH-A5IV`: 2,888 patches -> failed under Protocol 1.
   - `TCGA-QG-A5Z2`: 3,682 patches -> failed under Protocol 1.
   - Separately, `TCGA-G4-6311` was rejected during ingestion due to bundle overwrite protection.

Both root causes were traced to specific lines of code in the Android application and confirmed against the physical device's SQLite database (`crc_batch_runs.db`).

---

## 2. Status Color System Audit

### 2.1 Current Implementation & Color Conflation

Inspection of `BatchSpecimenAdapter.kt` revealed that status backgrounds were hardcoded to reuse `R.color.status_failed`:

```kotlin
// In BatchSpecimenAdapter.kt lines 96-98:
val chipBg = if (isMsi) R.color.status_failed else R.color.status_success
chipClass.chipBackgroundColor = ColorStateList.valueOf(ContextCompat.getColor(context, chipBg))

// In BatchSpecimenAdapter.kt lines 111-118:
val statusBg = when (item.status) {
    BatchSpecimenStatus.VALIDATED -> R.color.primary_dark
    BatchSpecimenStatus.INFERRING -> R.color.status_pending
    BatchSpecimenStatus.REJECTED -> R.color.status_failed  // <-- MAPPED TO RED!
    BatchSpecimenStatus.FAILED -> R.color.status_failed    // <-- MAPPED TO RED!
    BatchSpecimenStatus.CANCELLED -> R.color.text_secondary_light
    else -> R.color.text_secondary_light
}
```

Because `R.color.status_failed` resolves to `#B42318` (in light mode) or `#F87171` (in dark mode):
- **MSI** resolved to red.
- **FAILED** resolved to red.
- **REJECTED** resolved to red.

This made it impossible for laboratory personnel to quickly differentiate between a positive research finding (`MSI`), an engine or tensor runtime crash (`FAILED`), and an administrative or bundle policy exclusion (`REJECTED`).

### 2.2 Approved Healthcare Palette & Contrast Analysis

The approved palette defines four distinct colors on the dark navy/slate background (`#101820` / `#18232E`):

| Status | Approved Color | Semantic Role | Relative Luminance ($L$) | Contrast vs Dark Navy (`#101820`) | Contrast vs Black (`#000000`) | Contrast vs White (`#FFFFFF`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MSS** | `#2ECC9A` (Emerald Green) | Model classification (Class 0, $P < 0.83$) | 0.457 | **8.5:1 (AAA)** | **10.1:1 (AAA)** | 2.1:1 (Fail) |
| **MSI** | `#F56B70` (Coral Red) | Model classification (Class 1, $P \ge 0.83$) | 0.260 | **6.2:1 (AA)** | **6.2:1 (AA)** | 3.4:1 (Fail) |
| **FAILED** | `#F59E0B` (Amber Orange) | Execution / processing exception | 0.430 | **8.1:1 (AAA)** | **9.6:1 (AAA)** | 2.2:1 (Fail) |
| **REJECTED** | `#D6A83D` (Gold / Ochre) | Policy exclusion / overwrite protection | 0.410 | **7.8:1 (AAA)** | **9.2:1 (AAA)** | 2.3:1 (Fail) |

**Key Design Finding**:
All four approved badge colors have relatively high luminance ($L \in [0.26, 0.46]$). When rendered with dark text (`#101820` or `#000000`), the contrast ratio satisfies **WCAG 2.1 AA and AAA** requirements (6.2:1 to 10.1:1). In contrast, white text fails WCAG standards on `#2ECC9A`, `#F59E0B`, and `#D6A83D`. Therefore, badges must use dark text (`#101820`) for optimal clinical readability.

---

## 3. Batch Protocol Selection Audit

### 3.1 Trace of Batch Execution

The execution chain for batch inference proceeds as follows:
```
[User taps "Run Batch Analysis"]
            │
            ▼
BatchAnalysisFragment.startBatchExecution()
            │
            ▼
BatchCoordinator.executeBatchInference()
            │
            ▼
Loop over valid items:
classifier.predictAuthentic(
    visionData = bundle.visionData,
    numPatches = bundle.patchCount,
    textData = bundle.textData,
    caseId = bundle.specimenId,
    provenance = AUTHENTIC_IMPORTED
    // protocol omitted -> DEFAULTED TO PROTOCOL_1_CAPPED
)
            │
            ▼
CrcMultimodalClassifier.predict()
            │
            ▼
InputValidator.validateVision(visionData, numPatches, protocol = PROTOCOL_1_CAPPED)
            │
            ▼
require(numPatches <= 1000)
            │
            ├── N <= 1000: PASS -> Run ONNX Session -> COMPLETED (MSS / MSI)
            │
            └── N > 1000: THROW IllegalArgumentException -> Catch in BatchCoordinator -> FAILED
```

### 3.2 Evidence from Physical Device Database

Direct query of `crc_batch_runs.db` pulled from the connected `vivo V2125` confirmed the exact error states in batch `BATCH-20261027-160953-901698`:

```text
Specimen ID    | Patches | Status    | Error Details / Validation Error
---------------+---------+-----------+-------------------------------------------------------------
TCGA-G4-6311   |       0 | REJECTED  | A bundle for specimen 'TCGA-G4-6311' already exists. Overwrite not permitted.
TCGA-G4-6315   |     125 | COMPLETED | None (MSS, P = 0.0510)
TCGA-G4-6320   |     913 | COMPLETED | None (MSS, P = 0.1464)
TCGA-G4-6323   |     250 | COMPLETED | None (MSS, P = 0.0676)
TCGA-G4-6586   |     730 | COMPLETED | None (MSI, P = 0.9945)
TCGA-G4-6588   |     523 | COMPLETED | None (MSI, P = 0.9949)
TCGA-G4-6626   |      89 | COMPLETED | None (MSS, P = 0.7028)
TCGA-G4-6628   |     701 | COMPLETED | None (MSI, P = 0.9949)
TCGA-G5-6641   |      90 | COMPLETED | None (MSI, P = 0.9845)
TCGA-NH-A50T   |    2296 | FAILED    | Protocol 1 enforces a maximum cap of 1000 patches, but received 2296
TCGA-NH-A50U   |     449 | COMPLETED | None (MSI, P = 0.9539)
TCGA-NH-A5IV   |    2888 | FAILED    | Protocol 1 enforces a maximum cap of 1000 patches, but received 2888
TCGA-NH-A6GA   |     327 | COMPLETED | None (MSS, P = 0.1514)
TCGA-NH-A6GB   |     650 | COMPLETED | None (MSS, P = 0.4738)
TCGA-NH-A8F8   |     259 | COMPLETED | None (MSS, P = 0.8257)
TCGA-QG-A5YV   |     665 | COMPLETED | None (MSS, P = 0.0039)
TCGA-QG-A5Z2   |    3682 | FAILED    | Protocol 1 enforces a maximum cap of 1000 patches, but received 3682
TCGA-QL-A97D   |     349 | COMPLETED | None (MSS, P = 0.0935)
TCGA-SS-A7HO   |     851 | COMPLETED | None (MSS, P = 0.0049)
TCGA-WS-AB45   |     343 | COMPLETED | None (MSI, P = 0.9942)
```

### 3.3 Root-Cause Analysis: Over-Limit Specimen Failures

1. **Protocol 2 is Fully Implemented in Engine**:
   - `CrcMultimodalClassifier.predict()` natively supports `InferenceProtocol.PROTOCOL_2_UNCAPPED`.
   - `InputValidator.validateVision()` with `PROTOCOL_2_UNCAPPED` enforces positive patch counts and memory allocation bounds while skipping the 1,000-patch cap.
   - ONNX Runtime dynamically binds the input shape `[1, numPatches, 1024]` via native direct NIO `FloatBuffer`.
   - `AuthenticBundleLoader` already loads and validates bundles of arbitrary size using `PROTOCOL_2_UNCAPPED`.
2. **Missing Routing in `BatchCoordinator.kt`**:
   - `BatchCoordinator.executeBatchInference()` called `classifier.predictAuthentic()` without supplying a `protocol` parameter, which defaulted to `InferenceProtocol.PROTOCOL_1_CAPPED`.
   - As a result, specimens with $N > 1000$ patches threw validation errors during execution and were marked `FAILED`.
3. **Missing Protocol Metadata in Batch Persistence**:
   - `batch_items` table in SQLite had no `protocol` or `protocol_reason` column.
   - `BatchSpecimenItem` lacked fields to record the actual protocol used and the rationale for protocol selection.
4. **Bundle Overwrite Protection**:
   - `BundleStorageManager.saveBundle(context, bundle, overwrite = false)` correctly throws an `IllegalArgumentException` when attempting to import a duplicate bundle without explicit authorization.
   - In `BatchCoordinator.ingestUris()`, this exception was caught and the item was set to `REJECTED` with the exact message `"A bundle for specimen 'TCGA-G4-6311' already exists. Overwrite not permitted."`. This behavior is correct and preserves specimen immutability, but its visual representation was erroneously red rather than gold.

---

## 4. Planned Remediation Architecture

### 4.1 Status Color System
- Define explicit color resources in `res/values/colors.xml` and `res/values-night/colors.xml`:
  - `batch_status_mss`: `#2ECC9A`
  - `batch_status_msi`: `#F56B70`
  - `batch_status_failed`: `#F59E0B`
  - `batch_status_rejected`: `#D6A83D`
- In `BatchSpecimenAdapter.kt`:
  - Render `MSS` chip with `#2ECC9A` background and `#101820` text.
  - Render `MSI` chip with `#F56B70` background and `#101820` text.
  - Render `FAILED` chip with `#F59E0B` background and `#101820` text.
  - Render `REJECTED` chip with `#D6A83D` background and `#101820` text.
  - Update `tvItemError` text color to `#F59E0B` (for runtime failures) or `#D6A83D` (for policy rejections).

### 4.2 Batch Protocol Selection & Fallback
- In `BatchModels.kt`:
  - Add `protocol: String = "PROTOCOL_1_CAPPED"` and `protocolReason: String? = null` to `BatchSpecimenItem`.
- In `BatchCoordinator.kt`:
  - Before invoking inference, inspect each specimen's `bundle.patchCount`.
  - If `patchCount <= 1000`: execute under `PROTOCOL_1_CAPPED`.
  - If `patchCount > 1000`: route to `PROTOCOL_2_UNCAPPED` with explicit record:
    `protocol = "PROTOCOL_2_UNCAPPED"`, `protocolReason = "Exceeded Protocol 1 cap ($patchCount > 1000); routed to Protocol 2 Uncapped"`.
  - Pass the resolved protocol to `classifier.predictAuthentic(..., protocol = targetProtocol)`.
- In `BatchRepository.kt`:
  - Upgrade SQLite helper to version 2 with `ALTER TABLE batch_items ADD COLUMN protocol TEXT DEFAULT 'PROTOCOL_1_CAPPED'` and `ALTER TABLE batch_items ADD COLUMN protocol_reason TEXT`.
- In `item_batch_specimen.xml` & `BatchSpecimenAdapter.kt`:
  - Display the protocol used (`Protocol 1` vs `Protocol 2`) in the specimen metrics row or badge.
  - Display the auto-routing note when `protocolReason` is present.
- In `BatchReportGenerator.kt`:
  - Record the actual protocol used in the JSON report (`"protocol": "${item.protocol}"`) and PDF report table.
