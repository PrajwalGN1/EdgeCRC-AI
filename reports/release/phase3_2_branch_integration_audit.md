# Phase 3.2 — Branch Difference, Change Inventory & Integration Safety Audit

- **Project:** EdgeCRC-AI — Multimodal Fusion for Detection of Colorectal Cancer Using ClinicalBERT and Phikon-v2
- **Repository:** `https://github.com/PrajwalGN1/EdgeCRC-AI.git`
- **Local Project Root:** `<REPO_ROOT>` (`MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`)
- **Audit Timestamp:** `2026-10-31T14:17:30+05:30`
- **Comparison Baseline (`base...head`):** Local `main` (`189b6638c31d80344ecc8c45133c7058c3e47fa1`) vs. Live `origin/edge-deployment` (`db72e5f046b92a3a408cb8b3ca9f6e49a4c9e039`)
- **Audit Mode:** Strictly read-only repository inspection (zero commits, merges, fetches, branch switches, resets, stagings, or edits to existing tracked files)

---

## 1. Executive Summary

| Audit Dimension | Status | Summary Finding |
| :--- | :---: | :--- |
| **Overall Integration Safety** | **PASS** | `origin/edge-deployment` (`db72e5f`) is a **strictly linear, conflict-free fast-forward descendant** (`9` commits ahead, `0` commits behind) of local `main` (`189b663`). |
| **Upstream ML & Split Integrity** | **PASS** | Primary weights (`fusion_model_v1_1.pth`), baseline weights (`late_fusion_model_v1_0.pth`), 66-case split (`actual_validation_keys.json`), `patient_registry.json`, and `SummaryToPatchCrossAttention` Python code are **100% bit-for-bit identical** between `main` and `origin/edge-deployment`. |
| **ONNX & Threshold Contract** | **PASS** | `fusion_model_fp32.onnx` (`21,015,452` B, SHA-256 `0402f6c5...702d73`) matches across `models/` and `android/.../assets/models/`, enforcing `vision_seq [1,N,1024]`, `text_summary [1,768]` -> `logit`, `probability`, `attention_weights` at frozen threshold `0.83` (`0 = MSS`, `1 = MSI`). |
| **Android Workstation Integrity** | **PASS** | Zero manifest permissions, scoped `FileProvider`, uncompressed ONNX asset packaging, additive SQLite migrations (`v1->v4` and `v1->v3`), and truthful upstream-encoded tensor documentation verified. |
| **Secrets, Credentials & PII** | **PASS** | `0` API keys, tokens, passwords, keystores, `.apk` binaries, `.db` dumps, `.crcbundle` files, or patient PII across all `334` changed files. (`18` historical Markdown/CSV audit logs contain non-secret local path strings — logged as `NOTE`). |
| **Release Tag Monotonicity** | **NOTE** | Tags `v0.1.0`, `v1.0.1`, and `v1.0.2` already exist on `origin`. Next release should use **`v1.0.3`** or **`v1.1.0`** rather than `v1.0.0`. |

---

## 2. Verified Repository State (Phase 3.2.1)

Executed read-only verification commands confirmed that all Phase 3.1 conditions remain intact:

- **Active Branch (`git branch --show-current`):** `edge-deployment`
- **Local `HEAD`:** `db72e5f046b92a3a408cb8b3ca9f6e49a4c9e039`
- **Remote URL (`git remote -v`):** `origin -> https://github.com/PrajwalGN1/EdgeCRC-AI.git` (fetch & push)
- **Preserved Working-Tree State (`git status --short`):**
  ```text
   M README.md
  ?? reports/release/
  ```
  - ` M README.md`: Preserved untouched (single leading newline at line 1).
  - `?? reports/release/phase3_1_repository_state_audit.md`: Preserved untouched.

---

## 3. Local and Remote Branch Inventory

| Reference Scope | Ref Name | Target Commit SHA-1 | Live Status |
| :--- | :--- | :--- | :--- |
| **Local Head** | `refs/heads/edge-deployment` | `db72e5f046b92a3a408cb8b3ca9f6e49a4c9e039` | Active checked-out branch; synchronized with live `origin/edge-deployment` |
| **Local Head** | `refs/heads/main` | `189b6638c31d80344ecc8c45133c7058c3e47fa1` | Pre-Android Python/Streamlit baseline (`9` commits behind `edge-deployment`) |
| **Live Remote (`ls-remote`)** | `HEAD` (Default Branch Symref) | `ref: refs/heads/edge-deployment` (`db72e5f...`) | **Active GitHub default landing branch** |
| **Live Remote (`ls-remote`)** | `refs/heads/edge-deployment` | `db72e5f046b92a3a408cb8b3ca9f6e49a4c9e039` | **Sole branch currently existing on GitHub `origin`** |
| **Live Remote (`ls-remote`)** | `refs/heads/main` | *(Does not exist on remote)* | Absent on GitHub |
| **Local Cached Remote Ref** | `refs/remotes/origin/edge-deployment` | `db72e5f046b92a3a408cb8b3ca9f6e49a4c9e039` | Current |
| **Local Cached Remote Ref** | `refs/remotes/origin/main` | `189b6638c31d80344ecc8c45133c7058c3e47fa1` | **Stale local tracking ref** (not used as authoritative target) |
| **Local Cached Remote Ref** | `refs/remotes/origin/HEAD` | `189b6638c31d80344ecc8c45133c7058c3e47fa1` | **Stale local symref** |

---

## 4. Commit Ancestry Analysis (Phase 3.2.2)

### Complete 12-Commit Repository Graph (`git log --graph --oneline --decorate --all -30`)
```text
* db72e5f (HEAD -> edge-deployment, origin/edge-deployment) docs: add comprehensive EdgeCRC-AI documentation and sync Android workstation
* 5465a90 chore(release): add automation script for v1.0.2 GitHub release
* 2ba253a (tag: v1.0.2) feat(mobile): enterprise-grade UI/UX redesign and interactive research workspace (v1.0.2)
* 62e4226 chore(release): add automation scripts for v1.0.1 GitHub release and asset upload
* f8519f5 (tag: v1.0.1) fix(mobile): resolve startup crash on Samsung Galaxy S24 with Material theme and resources (v1.0.1)
* d3c68eb docs(release): publish v0.1.0 Android APK distribution audit and S24 installation guide
* 19b27f4 (tag: v0.1.0) fix(docs): resolve Mermaid flowchart render error and add tcga-coad notebook
* dff4959 fix(docs): resolve GitHub Mermaid rendering error in architecture flowchart
* b790e51 Add Android edge deployment and validation artifacts
* 189b663 (origin/main, origin/HEAD, main) chore: remove roadmap/next-team and authors, keep mentor Mr Nageshwara Guptha — professional public version
* ac27a8f chore: exclude generated heatmaps, fix .gitignore
* 5a9cd2a first commit
```

### Ancestry Verification Findings
- **Common Ancestor (`git merge-base main origin/edge-deployment`):** `189b6638c31d80344ecc8c45133c7058c3e47fa1` (the exact tip of local `main`).
- **Fast-Forward Descendant:** **Yes (`PASS`).** `origin/edge-deployment` is a direct, single-parent linear descendant of `main`.
- **Unrelated Histories or Divergent Commits:** **None (`PASS`).** Zero merge commits and zero conflicting edits exist between `main` and `origin/edge-deployment`.

---

## 5. Commit Divergence Counts

- **`git rev-list --left-right --count main...origin/edge-deployment`:** `0	9`
  - **Commits exclusive to local `main` (`origin/edge-deployment..main`):** **`0`**
  - **Commits exclusive to `origin/edge-deployment` (`main..origin/edge-deployment`):** **`9`**

| # | Commit | Timestamp (IST) | Author | Files Touched | Summary |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 1 | `b790e51` | `2026-10-19 15:35:37` | `PrajwalGN1` | `107` (`100 A`, `7 M`) | Initial ONNX export (`fp32`/`fp16`), Python inference wrappers, Phase 0–4 mobile audits, initial Android skeleton |
| 2 | `dff4959` | `2026-10-19 16:11:22` | `PrajwalGN1` | `1` (`1 M`) | Mermaid flowchart syntax fix in `README.md` |
| 3 | `19b27f4` | `2026-10-19 16:11:41` | `PrajwalGN1` | `1` (`1 A`) | Added `tcga-coad (1).ipynb` (`tag: v0.1.0`) |
| 4 | `d3c68eb` | `2026-10-19 16:45:34` | `PrajwalGN1` | `6` (`4 A`, `2 M`) | Added `release/INSTALLATION.md`, `scripts/publish_release.py`, Phase 4.2 distribution report |
| 5 | `f8519f5` | `2026-10-19 20:47:10` | `PrajwalGN1` | `14` (`7 A`, `7 M`) | Resolved Samsung Galaxy S24 startup crash with Material theme & resources (`tag: v1.0.1`) |
| 6 | `62e4226` | `2026-10-19 20:56:34` | `PrajwalGN1` | `2` (`2 A`) | Added `scripts/publish_release_v1_0_1.py` and `scripts/upload_asset_v1_0_1.py` |
| 7 | `2ba253a` | `2026-10-21 18:56:05` | `PrajwalGN1` | `23` (`15 A`, `8 M`) | Enterprise UI/UX redesign and `release/release_manifest.json` (`tag: v1.0.2`) |
| 8 | `5465a90` | `2026-10-21 19:02:15` | `PrajwalGN1` | `1` (`1 A`) | Added `scripts/publish_release_v1_0_2.py` |
| 9 | `db72e5f` | `2026-10-31 13:55:41` | `PrajwalGN1` | `217` (`197 A`, `20 M`) | Comprehensive 33-section `README.md`, full 13-screen Android workstation, SQLite `v4`/`v3`, Batch/Cohort/Analytics/PDF engines, unit/device tests, `.gitignore` hardening |

---

## 6. File-Level Change Inventory (`main...origin/edge-deployment`) (Phase 3.2.3)

- **Cumulative Diff Summary (`git diff --shortstat main...origin/edge-deployment`):**
  `334 files changed, 81,458 insertions(+), 463 deletions(-)` (`327` Added, `7` Modified, `0` Deleted, `0` Renamed).
- **Independent Verification of Commit `db72e5f`:** Confirmed exact `217` files (`197` Added, `20` Modified, `0` Deleted); `38` of those `20 + 18` files overlapped earlier files introduced in commits `b790e51`–`5465a90`, yielding `334` net unique file changes relative to `main`.

### 12-Category Classification Matrix

| # | Category | Added (`A`) | Modified (`M`) | Deleted (`D`) | Renamed (`R`) | Net Size on `edge-deployment` | Integration Risk Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Root documentation** (`README.md`) | `0` | `1` | `0` | `0` | `73,516 B` (`0.070 MB`) | **Low** — Replaces pre-Android 305-line `README.md` with verified 944-line full-project `README.md`. |
| **2** | **Android application source** (`AndroidManifest.xml`, `CrcApplication.kt`, `MainActivity.kt`, `WorkstationNavigationHost.kt`) | `3` | `0` | `0` | `0` | `12,795 B` (`0.012 MB`) | **None** — Pure additive Android entry points (`android/` did not exist on `main`). *(Note: `CrcApplication.kt` is counted under Kotlin source below; `MainActivity`, `WorkstationNavigationHost`, and `AndroidManifest` here).* |
| **3** | **Android resources and layouts** (`android/app/src/main/res/**`) | `68` | `0` | `0` | `0` | `410,559 B` (`0.392 MB`) | **None** — Material Day/Night themes, `23` layouts, `27` drawables, `7` color selectors, `file_paths.xml`, navigation menu. |
| **4** | **Android build configuration** (`android/*.kts`, `gradle.properties`, `gradlew*`, `gradle-wrapper.*`, `proguard-rules.pro`) | `7` | `0` | `0` | `0` | `49,471 B` (`0.047 MB`) | **Low** — Standard Gradle `8.10.2` wrapper (`gradle-wrapper.jar` `43.4 KB`) and AGP `8.4.2` / JDK `17` configuration; no local SDK paths (`local.properties` gitignored). |
| **5** | **Kotlin/Java source & tests** (`android/app/src/{main,test,androidTest}/java/**`, excl. persistence/entry) | `82` | `0` | `0` | `0` | `937,713 B` (`0.894 MB`) | **None** — Inference, batch, cohort evaluation, analytics, PDF report generator, security validators, `23` JVM test suites, and `7` `androidTest` suites. |
| **6** | **Python inference & evaluation tools** (`src/inference/*`, `src/eval/*`, `tools/*`, `tests/*`, `src/ui/app.py`) | `31` | `1` | `0` | `0` | `269,582 B` (`0.257 MB`) | **Low** — Only `src/ui/app.py` was modified (`M`, adding a Raw-Input Diagnostic tab alongside the existing Cohort Explorer tab). All existing training/model files untouched. |
| **7** | **ONNX model deployment assets** (`models/fusion_model_fp32.onnx`, `models/fusion_model_fp16.onnx`, `android/.../assets/models/fusion_model_fp32.onnx`) | `3` | `0` | `0` | `0` | `52,550,558 B` (`50.116 MB` logical; `30.07 MB` unique Git blobs) | **Low** — Individual files (`20.04 MB` and `10.03 MB`) are well under GitHub's `100 MB` file limit. Identical FP32 ONNX blobs (`0b3d9baa1a77`) are stored once in Git packfiles. |
| **8** | **Validation manifests & fixtures** (`assets/reference_labels/*.json`, `assets/reference_tensors/*.bin`, `data/reference_labels/*.csv`, `reports/mobile_deployment/*.{json,csv}`) | `31` | `0` | `0` | `0` | `11,196,047 B` (`10.677 MB`) | **None** — Required by `CrcApplication` seeding, `CohortEvaluationFragment`, and JVM parity/metric tests (`max` single `.bin` file is `4.10 MB`). |
| **9** | **Database & persistence implementation** (`data/*.kt`, `BatchRepository.kt`, `BundleStorageManager.kt`) | `13` | `0` | `0` | `0` | `99,538 B` (`0.095 MB`) | **None** — Self-contained `SQLiteOpenHelper` repositories (`crc_workstation.db` v4 & `crc_batch_runs.db` v3) with additive `onUpgrade` migrations. |
| **10** | **Reports, notebooks & documentation** (`reports/{audit,edge_deployment,mobile_deployment}/**/*.md`, `tcga-coad (1).ipynb`) | `79` | `0` | `0` | `0` | `1,047,657 B` (`0.999 MB`) | **Low (`NOTE`)** — Lightweight engineering audit trail; a subset contains historical local workspace paths (see Section 10). |
| **11** | **Configuration & release scripts** (`.gitignore`, `scripts/publish_release*.py`, `scripts/upload_asset_v1_0_1.py`) | `4` | `1` | `0` | `0` | `31,405 B` (`0.030 MB`) | **None** — `.gitignore` strictly expands ignore coverage without removing existing rules; release scripts read credentials from `git credential fill`. |
| **12** | **Generated evaluation outputs & handoff docs** (`PROJECT_HANDOFF_AUDIT.md`, `REPOSITORY_FILE_MAP.md`, `release/*`, `results/computational_power_fp32.*`, `results/*.{png,txt}`) | `6` | `4` | `0` | `0` | `548,105 B` (`0.523 MB`) | **Low (`NOTE`)** — `results/Cross_Attention_Model_Evaluation_Report.txt` & plots were regenerated in `b790e51` (explained in Section 7). |
| — | **TOTAL** | **`327`** | **`7`** | **`0`** | **`0`** | **`67,226,946 B` (`64.11 MB` logical)** | **Zero Deletions, Zero Conflicts** |

### Complete List of All `7` Modified (`M`) Files Across `main...origin/edge-deployment`
1. `.gitignore` (`b790e51`, `d3c68eb`, `db72e5f`) — Appended Android, `.apk`, `.crcbundle`, crash log, scratch, and local helper exclusions.
2. `README.md` (`b790e51`, `dff4959`, `db72e5f`) — Upgraded to comprehensive 33-section EdgeCRC-AI documentation.
3. `src/ui/app.py` (`b790e51`) — Added Tab 1 (*Live Raw-Input Diagnostic*) using `MultimodalPredictor` while preserving Tab 2 (*Cohort Explorer & Tensor Benchmark*).
4. `results/Cross_Attention_Model_Evaluation_Report.txt` (`b790e51`) — Regenerated evaluation summary (see Section 7).
5. `results/Confusion_Matrix_C.png` (`b790e51`) — Regenerated plot matching #4.
6. `results/ROC_Curve_C.png` (`b790e51`) — Regenerated plot matching #4.
7. `results/Real_Cross_Attention_Matrix.png` (`b790e51`) — Regenerated plot matching #4.

---

## 7. ML Model Integrity Findings (Phase 3.2.4 A) — `PASS`

### 7.1 Cryptographic & Git Blob Verification of Frozen Baseline Artifacts

| Artifact Path | Git Blob (`main`) | Git Blob (`origin/edge-deployment`) | Parity Status |
| :--- | :--- | :--- | :---: |
| `models/fusion_model_v1_1.pth` | `a747ae1e1d55` | `a747ae1e1d55` | **PASS (100% Identical)** |
| `models/late_fusion_model_v1_0.pth` | `3dedb05771bb` | `3dedb05771bb` | **PASS (100% Identical)** |
| `results/actual_validation_keys.json` (66 holdout IDs) | `265f060b3528` | `265f060b3528` | **PASS (100% Identical)** |
| `data/patient_registry.json` (326 patients & GT labels) | `3934fbcfbf1d` | `3934fbcfbf1d` | **PASS (100% Identical)** |
| `src/architectures/cross_attention_fusion.py` | `8e31a3e0483d` | `8e31a3e0483d` | **PASS (100% Identical)** |
| `src/architectures/late_fusion_model.py` | `e22fe00fe7f1` | `e22fe00fe7f1` | **PASS (100% Identical)** |
| `src/cross_attentiion_train.py` | `8fce7967eb38` | `8fce7967eb38` | **PASS (100% Identical)** |
| `src/cross_attention_eval.py` | `d938054fb078` | `d938054fb078` | **PASS (100% Identical)** |

### 7.2 ONNX Artifact & Mathematical Contract Verification
- **Root Export (`models/fusion_model_fp32.onnx`):** `21,015,452` bytes | Blob `0b3d9baa1a77` | SHA-256 `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`.
- **Android Bundled Asset (`android/app/src/main/assets/models/fusion_model_fp32.onnx`):** `21,015,452` bytes | Blob `0b3d9baa1a77` | SHA-256 `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` (**exact bit-for-bit match**).
- **Verified Frozen Contract:**
  - `vision_seq`: `[1, N, 1024]` (`FLOAT` / FP32)
  - `text_summary`: `[1, 768]` (`FLOAT` / FP32)
  - Outputs: `logit [1, 1]`, `probability [1, 1]`, `attention_weights [1, 1, N]`
  - Decision Threshold: **`0.83`** (`ThresholdLogic.DECISION_THRESHOLD = 0.83f` & `MultimodalPredictor.DECISION_THRESHOLD = 0.83`)
  - Class Mapping: **`0 = MSS`**, **`1 = MSI`** (`probability >= 0.83f -> MSI`, `< 0.83f -> MSS`)
  - Protocols: `PROTOCOL_1_CAPPED` (`1..1000` patches; throws explicit `IllegalArgumentException` if `N > 1000` — **zero silent truncation**); `PROTOCOL_2_UNCAPPED` (`N >= 1` full sequence).

### 7.3 Explanation of `results/Cross_Attention_Model_Evaluation_Report.txt` Diff (`NOTE`)
- In commit `b790e51`, `results/Cross_Attention_Model_Evaluation_Report.txt` changed from `Accuracy: 0.8939 / ROC-AUC: 0.9150` (`FP=3`) to `Accuracy: 0.8788 / ROC-AUC: 0.9250` (`FP=4`) because `MultimodalDataset` used unseeded `torch.randperm` when subsampling slides with `N > 1000` patches.
- This exact stochasticity was subsequently audited in `reports/mobile_deployment/phase1_5_1_sampling_reproducibility_audit.md` and resolved for mobile deployment by locking deterministic indices in `protocol1_patch_selection_manifest.json` and `cohort_evaluation_reference_metrics.json` (`Accuracy = 89.3939%`, `TP=12, TN=47, FP=3, FN=4`, `ROC-AUC = 0.92625`, `PR-AUC = 0.82205`). No model weights or ground-truth labels changed.

---

## 8. Android Implementation Integrity Findings (Phase 3.2.4 B & C) — `PASS`

- **Manifest & Zero-Network Guarantee (`AndroidManifest.xml`):** Declares **zero `<uses-permission>` elements**, sets `android:allowBackup="false"`, configures `MainActivity` (`exported="true"`), and restricts `androidx.core.content.FileProvider` (`org.crc.multimodal.fileprovider`, `exported="false"`, `grantUriPermissions="true"`) to `@xml/file_paths` (`filesDir/reports/` and `cacheDir/reports/`).
- **Gradle & Asset Packaging (`android/app/build.gradle.kts`):** `namespace = "org.crc.multimodal"`, `compileSdk = 34`, `minSdk = 24`, `targetSdk = 34`, `versionCode = 3`, `versionName = "1.0.2"`, JDK `17`, `onnxruntime-android:1.18.0`, and `noCompress.addAll(listOf("onnx"))`. `ReferenceTensorLoader` reads `.bin` assets via stream (`assets.open(...)`), so it does not depend on uncompressed file descriptors (`openFd`), preventing AAPT compression crashes.
- **SQLite Migration Integrity:**
  - `SqliteWorkstationRepository` (`crc_workstation.db`, `DATABASE_VERSION = 4`): `onUpgrade` applies additive `CREATE TABLE IF NOT EXISTS` and guarded `ALTER TABLE` steps across `v1 -> v2 -> v3 -> v4` with zero `DROP TABLE` destructive fallbacks.
  - `BatchRepository` (`crc_batch_runs.db`, `DATABASE_VERSION = 3`): `onUpgrade` applies guarded `addColumnIfMissing` migrations (`is_archived` on `v2`; `protocol_used` and `status_reason` on `v3`) preserving existing batch records.
- **Architectural Honesty (`PASS`):** Both `README.md` and `ProvenanceAuditManager.kt` (`encoderStatus = "UPSTREAM_PRECOMPUTED_UNVERIFIED"`) explicitly state that Phikon-v2 and Bio_ClinicalBERT run upstream in Python and **do not** execute on raw WSI slides or raw clinical text on Android.

---

## 9. `README.md` and Documentation Difference Review (Phase 3.2.6) — `PASS`

| Aspect | `main:README.md` (`305` lines, `15.9 KB`) | `origin/edge-deployment:README.md` (`944` lines, `73.5 KB`) | Audit Verdict |
| :--- | :--- | :--- | :--- |
| **Upstream Model & Holdout Metrics** | Covered (`SummaryToPatchCrossAttention`, `89.39%` acc, `75%` MSI recall) | Covered with exact parameter counts (`5,250,561`), layer shapes, and Protocol 1 & 2 contracts | **Preserved & Expanded** |
| **Android Edge Workstation** | **Completely absent** (`0` lines) | Complete (`13` screens, ONNX Runtime `1.18.0`, `.crcbundle` v2 byte spec, dual SQLite schemas, Batch/Cohort/Analytics/PDF) | **Essential Addition** |
| **Streamlit & XAI Stack** | Covered (`streamlit run`, Ollama `qwen2.5-coder:7b`, Docker, 3-tier XAI) | Covered (Section 1, Section 4, Section 18, Section 21.6 Streamlit launch + direct ONNX snippet) | **Preserved** |
| **Historical Training / Kaggle Resync Commands** | Includes Kaggle dataset URL (`joangibert/tcga_coad_msi_mss_jpg`), `kaggle datasets download` CLI, `docker build/run`, and `AdamW lr=1e-4, pos_weight=4.17` | Detailed in untouched `docs/ARCHITECTURE.md`, `docs/HANDOFF.md`, `data/DATASET.md`, and `Dockerfile` (linked from `README.md`) | **No Repo Information Lost** *(Optional: a 4-line Kaggle/Docker pointer could be added if desired, but is already in `data/DATASET.md` & `docs/HANDOFF.md`)* |
| **Link & Asset Validity** | Cited `src/architectures/late_fusion_model.py:5` and old repo folder `Colorectal_cancer_detect` | `33/33` TOC anchors, `8/8` relative links, `45/45` inline paths, and `3/3` Mermaid diagrams verified on disk | **`edge-deployment` Superior & Valid** |

**Consolidation Recommendation:** Adopt `origin/edge-deployment:README.md` directly as the canonical root `README.md` without manually splicing obsolete sections from `main:README.md`, since `docs/ARCHITECTURE.md`, `docs/HANDOFF.md`, and `data/DATASET.md` remain intact in the repository and preserve every low-level upstream training/resync note.

---

## 10. Sensitive-Data and Repository-Hygiene Audit (Phase 3.2.5) — `PASS` (`NOTE`)

Read-only automated scanning was executed across all `334` changed/added blobs in `main...origin/edge-deployment`:

- **API Keys / GitHub PATs / Cloud Keys / Private Keys:** **`0` findings (`PASS`).** (Inspected `scripts/publish_release*.py` and `upload_asset_v1_0_1.py`: authentication tokens are retrieved dynamically at runtime via `git credential fill`, with zero hardcoded secrets.)
- **Keystores / Signing Credentials (`.jks`, `.keystore`) / `.env` Files:** **`0` findings (`PASS`).**
- **SQLite Database Dumps (`.db`, `.sqlite`), `.apk`/`.aab` Binaries, Crash `.log` Files, `.crcbundle` Exports:** **`0` findings (`PASS`).**
- **Patient-Identifiable Information (PHI/PII):** **`0` findings (`PASS`).** Only public de-identified `TCGA-XX-XXXX` case identifiers appear.
- **Files >= `1 MB` (`7` total across `334` files):**
  1. `models/fusion_model_fp32.onnx` (`20.04 MB`, blob `0b3d9baa1a77`)
  2. `android/app/src/main/assets/models/fusion_model_fp32.onnx` (`20.04 MB`, identical blob `0b3d9baa1a77` — deduplicated in Git packfile)
  3. `models/fusion_model_fp16.onnx` (`10.03 MB`, blob `c50417b3985d`)
  4. `android/app/src/main/assets/reference_tensors/tcga_cm_6674.bin` (`3.91 MB`)
  5. `android/app/src/main/assets/reference_tensors/tcga_cm_6675.bin` (`2.12 MB`)
  6. `android/app/src/main/assets/reference_tensors/tcga_aa_3845.bin` (`1.96 MB`)
  7. `android/app/src/main/assets/reference_tensors/tcga_d5_6540.bin` (`1.35 MB`)
  *(All 7 are legitimate model/benchmark assets well below GitHub's `100 MB` limit.)*
- **Informational Hygiene Observation — Historical Local Path Strings in Audit Markdown/CSV Reports (`NOTE`, Low Severity):**
  - Zero local paths exist in `README.md`, `android/`, `src/`, `tools/`, `tests/`, or `scripts/`.
  - `18` historical audit/manifest files under `PROJECT_HANDOFF_AUDIT.md` and `reports/` (spanning commits `b790e51` through `db72e5f`) record workspace/environment paths from verification runs:
    - `<LOCAL_WORKSPACE_DIR>` (`68` occurrences across `15` report files, including `40` rows in `reports/mobile_deployment/patient_tensor_transfer_manifest.csv`).
    - `<WINDOWS_USER_HOME>\...` (`5` occurrences: `2` HuggingFace cache paths in `reports/edge_deployment/raw_input_readiness_audit.md` [`b790e51`] and `3` Android SDK paths in `reports/mobile_deployment/phase4_2_issue_register.md` [`f8519f5`]).
    - `<LINUX_USER_HOME>/...` (`2` occurrences in `PROJECT_HANDOFF_AUDIT.md` [`b790e51`]).
  - **Impact:** Contains no credentials, tokens, or sensitive personal data (only generic OS folder names and local workspace root). Rewriting historical commits (`b790e51`, `f8519f5`) is prohibited (`No history rewrite`); if desired later, paths in head-level `.md`/`.csv` reports could be normalized in a follow-up docs commit, though not a security blocker.

---

## 11. Existing Tag and Release Versioning Analysis (Phase 3.2.8)

| Tag | Tag Type | Tag / Commit SHA-1 | Commit Date | Android Version | Release Role (`release_manifest.json`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`v0.1.0`** | Lightweight | `19b27f49c168a34275b51e03a000301e2d46576d` | `2026-10-19` | `1.0.0` (`code 1`) | Initial Android test pre-release (`archived_startup_crash`) |
| **`v1.0.1`** | Lightweight | `f8519f538844baa4564633ec7ace1c674cdb66e7` | `2026-10-19` | `1.0.1` (`code 2`) | Samsung Galaxy S24 Material theme startup crash fix (`remediated_baseline_ui`) |
| **`v1.0.2`** | Annotated (`27267803...`) | `2ba253a66b2c7695c0ec071b9c546d11034e6f15` | `2026-10-21` | `1.0.2` (`code 3`) | Enterprise UI/UX Redesign (`CRC Multimodal Edge AI v1.0.2`) |

### Release Version Evaluation (`v1.0.0` vs. `v1.0.3` vs. `v1.1.0`)
1. **Why `v1.0.0` Should NOT Be Used:** `v1.0.1` and `v1.0.2` are already published on `origin`. Creating `v1.0.0` on a descendant of `v1.0.2` violates Semantic Versioning (`SemVer 2.0.0`) ordering and causes tooling/CLI sorters (`git tag --sort=-version:refname`) to rank `v1.0.2` above `v1.0.0`.
2. **Comparison of Valid Next Versions:**
   - **`v1.1.0` (Recommended Primary SemVer Choice):** Commit `db72e5f` introduces major backward-compatible workstation functionality beyond `v1.0.2` (`13` fragments, dual SQLite databases, `.crcbundle` v2 & `.npy` SAF import, 100-item Batch Analysis, 66-case TCGA Cohort Evaluation with ROC/PR-AUC, Research Analytics, A4 PDF reporting, and complete project documentation). Under SemVer, substantial new backward-compatible features warrant a **minor version bump (`v1.1.0`)**. *(Note: If building a matching APK later, `versionName` can be bumped to `"1.1.0"` / `versionCode = 4`, or release notes can note that source tag `v1.1.0` packages workstation schema `v4`/`v3` with current Gradle `versionName = "1.0.2"` if strict zero-Gradle-edit is preferred.)*
   - **`v1.0.3` (Recommended Conservative Patch Alternative):** Appropriate if framing this release strictly as the documentation finalization, repository synchronization, and Batch UI button visibility refinement over the `1.0.x` Android series without starting a `1.1.x` line.

---

## 12. Integration Options Evaluation and Recommendation (Phase 3.2.7)

| Criterion | Option A: Fast-Forward Local `main` to `edge-deployment` (+ Push `main`) | Option B: Create Remote `main` Directly from `origin/edge-deployment` | Option C: Dedicated `release/edgecrc-ai-documentation` Branch + PR into `main` | Option D: Keep `edge-deployment` Only as Default Branch (No Remote `main`) |
| :--- | :--- | :--- | :--- | :--- |
| **Benefits** | Zero merge-commit noise; `main` and `edge-deployment` share identical SHA (`db72e5f`); establishes standard `main` on GitHub. | Single server-side ref push (`git push origin origin/edge-deployment:refs/heads/main`) without even switching local branches. | Provides a GitHub Pull Request review UI and explicit `--no-ff` merge audit record if a remote `main` baseline (`189b663`) is pushed first. | Zero Git branch changes needed; GitHub landing page already renders `db72e5f:README.md` right now. |
| **Risks** | Requires switching/updating local `main` (must clean 1-line `README.md` diff first). Doesn't change GitHub default branch until toggled in Settings. | Local `main` (`189b663`) remains behind unless also fast-forwarded locally (`git branch -f main origin/edge-deployment`). | Requires pushing old `189b663` as remote `main` first just to open a PR against it, creating a synthetic merge commit when history is already 100% linear. | External users/tools expecting a `main` branch (`git clone -b main`) will not find `refs/heads/main` on GitHub. |
| **Exact Git Operations** | 1. Clean working tree (`git checkout -- README.md`)<br>2. `git fetch --prune origin`<br>3. `git checkout main`<br>4. `git merge --ff-only origin/edge-deployment`<br>5. `git push -u origin main` | 1. `git branch -f main origin/edge-deployment`<br>2. `git push -u origin main`<br>3. `git remote set-head origin -a` | 1. `git push -u origin main` (at `189b663`)<br>2. `git checkout -b release/edgecrc-ai-documentation origin/edge-deployment`<br>3. `git push -u origin release/edgecrc-ai-documentation`<br>4. Open & merge PR `release/... -> main` on GitHub | None required (optional: `git fetch --prune origin && git remote set-head origin -a` locally). |
| **GitHub Settings Change Needed?** | Optional: switch Default Branch from `edge-deployment` to `main` **after** `main` is pushed at `db72e5f`. | Optional: switch Default Branch from `edge-deployment` to `main` **after** push. | Optional: switch Default Branch to `main` **after** PR is merged. | **No** (`edge-deployment` is already default). |
| **Affects Existing Default Branch?** | **No** (`edge-deployment` remains untouched at `db72e5f`). | **No** (`edge-deployment` remains untouched at `db72e5f`). | **No** (`edge-deployment` remains untouched). | **No**. |
| **Preserves All Existing Commits?** | **Yes (`100%` — all `12` commits & `3` tags preserved).** | **Yes (`100%` — all `12` commits & `3` tags preserved).** | **Yes (`100%` + 1 merge commit).** | **Yes (`100%`).** |

### Recommended Integration Strategy
**Recommend Combined Option A / B (Fast-Forward `main` to `origin/edge-deployment` and Publish `refs/heads/main` to `origin`, Preserving `edge-deployment`):**
- Because `merge-base(main, origin/edge-deployment) == main` (`189b663`, `0` divergent commits on `main`), fast-forwarding local `main` to `db72e5f` and pushing `refs/heads/main` to `origin` is mathematically lossless, creates **zero merge conflicts**, preserves `edge-deployment` untouched, keeps `v0.1.0`/`v1.0.1`/`v1.0.2` reachable from **both** `main` and `edge-deployment`, and ensures both branches serve the exact verified `944`-line `README.md`.
- *(Alternative if a formal GitHub PR UI review is desired by the owner: **Option C** — push `main` (`189b663`) and `release/edgecrc-ai-documentation` (`db72e5f`) to `origin` and open a Pull Request from `release/edgecrc-ai-documentation` or `edge-deployment` into `main`.)*

---

## 13. Risk Register

| ID | Risk Description | Severity | Likelihood | Mitigation |
| :--- | :--- | :---: | :---: | :--- |
| **R-01** | Switching GitHub default branch to `main` *before* `main` is updated to `db72e5f` would temporarily revert the landing page to the 305-line pre-Android `README.md`. | Medium | Low | Push `main` at `db72e5f` (or merge PR) **first** and verify `origin/main` SHA equals `db72e5f` before changing GitHub's Default Branch setting. |
| **R-02** | Uncommitted 1-line leading blank newline in working-tree `README.md` (` M README.md`) could block `git checkout main` or pollute a commit. | Low | High | Explicitly discard (`git checkout -- README.md`) at the start of Phase 3.3 once owner authorizes Phase 3.3. |
| **R-03** | Creating a new GitHub Release tagged `v1.0.0` would sort below existing remote releases `v1.0.1` and `v1.0.2`. | Medium | High | Use `v1.1.0` (feature release) or `v1.0.3` (patch/docs release) in Phase 3.7 per owner decision. |
| **R-04** | No `LICENSE` file exists in repository (`All rights reserved` by default), and Phikon-v2 (`Owkin Non-Commercial` / gated terms) imposes downstream model-weight usage constraints. | High (Gate) | Certain | Enforce **Phase 3.5 License & Attribution Gate** before publishing any public GitHub Release in Phase 3.7. |
| **R-05** | Untracked `reports/release/*.md` audit deliverables exist in the working tree. | Low | Certain | Decide in Phase 3.3/3.7 whether to commit `reports/release/` as formal release documentation or add `reports/release/` to `.gitignore`. |
| **R-06** | `18` historical report files under `reports/` and `PROJECT_HANDOFF_AUDIT.md` contain local path strings (`<LOCAL_WORKSPACE_DIR>`, `<WINDOWS_USER_HOME>\...`). | Low (`NOTE`) | Present | Informational only (zero secrets/PII); optional path redaction in head-level `.md`/`.csv` files if desired by owner without rewriting history. |

---

## 14. Exact Next Actions for Phase 3.3 (Pending Owner Authorization)

1. **Owner Confirmation on Integration Workflow & Working-Tree Cleanup:**
   - Confirm whether to discard the accidental 1-line leading newline in `README.md` (`git checkout -- README.md`).
   - Confirm preferred integration path:
     - **Path 1 (Recommended — Clean Fast-Forward `main`):** Fast-forward local `main` (`189b663 -> db72e5f`) and push `refs/heads/main` to `origin` while keeping `edge-deployment` intact.
     - **Path 2 (PR-Based — `release/edgecrc-ai-documentation` -> `main`):** Push baseline `main` (`189b663`) and `release/edgecrc-ai-documentation` (`db72e5f`) to `origin` and open a GitHub Pull Request into `main`.
2. **Execute Approved Phase 3.3 Branch Integration** without force-pushing, deleting branches, or altering model/tensor/threshold artifacts, and record full verification evidence in `reports/release/phase3_3_main_branch_integration_report.md`.
