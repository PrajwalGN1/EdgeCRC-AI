# CRC Edge AI Patient Tensor Export and Android Transfer Report

## Executive Summary

- **Deployment Status**: **100% SUCCESS** — All 40 approved patient tensors transferred individually and verified byte-for-byte and SHA-256 identical.
- **Total Approved Patients**: 20 (20 complete pairs)
- **Total Files Transferred & Verified**: 40 / 40
- **Total Payload Transferred**: 100,012,656 bytes (95.38 MB)
- **No-Archive Policy**: **ENFORCED** — 0 archives (ZIP/TAR/etc.) created; all files transferred individually via ADB.
- **Source & Frozen Artifact Integrity**: **PRESERVED** — 0 source files or frozen models modified.

## 1. System and Hardware Environment

- **Host OS**: Windows (PowerShell)
- **Repository Root**: `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`
- **Source Directory**: `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2\data\clean_tensors`
- **Target Android Device Model**: `vivo V2125` (Product: `2020iM`)
- **Device Serial**: `1377355236000F7`
- **Android Version**: Android 12 (SDK API level 31)
- **Target Destination Path**: `/sdcard/Download/CRC_Tensors/`
- **Available Storage on Device**: ~1.6 GB free (sufficient for 95.38 MB transfer)

## 2. Approved Patient Roster

The 20 user-approved patient IDs:
1. `TCGA-G4-6311`
2. `TCGA-G4-6315`
3. `TCGA-G4-6320`
4. `TCGA-G4-6323`
5. `TCGA-G4-6586`
6. `TCGA-G4-6588`
7. `TCGA-G4-6626`
8. `TCGA-G4-6628`
9. `TCGA-G5-6641`
10. `TCGA-NH-A50T`
11. `TCGA-NH-A50U`
12. `TCGA-NH-A5IV`
13. `TCGA-NH-A6GA`
14. `TCGA-NH-A6GB`
15. `TCGA-NH-A8F8`
16. `TCGA-QG-A5YV`
17. `TCGA-QG-A5Z2`
18. `TCGA-QL-A97D`
19. `TCGA-SS-A7HO`
20. `TCGA-WS-AB45`


## 3. Transfer and Verification Statistics

- **Expected File Count**: 40
- **Actual Transferred File Count**: 40
- **Missing Files**: 0
- **Destination Filename Conflicts**: 0
- **Failed Commands / Errors**: 0
- **Retries Required**: 0
- **Final Verification Status Breakdown**:
  - `TRANSFERRED_AND_HASH_VERIFIED`: 40
  - `TRANSFERRED_SIZE_VERIFIED_HASH_UNAVAILABLE`: 0
  - `ALREADY_PRESENT_AND_IDENTICAL`: 0
  - `FAILED`: 0
  - `BLOCKED_BY_CONFLICT`: 0

## 4. Comprehensive File-by-File Audit & Transfer Ledger

| # | Patient ID | Modality | Filename | Source Size (B) | Dest Size (B) | Source SHA-256 | Destination SHA-256 | Verification Status |
|---|---|---|---|---|---|---|---|---|
| 01 | `TCGA-G4-6311` | text | `TCGA-G4-6311_text.pt` | 1,574,511 | 1,574,511 | `2974f2e3a2e411bb...` | `2974f2e3a2e411bb...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 02 | `TCGA-G4-6311` | vision | `TCGA-G4-6311_vision.pt` | 2,217,597 | 2,217,597 | `1f537afa9b21a67e...` | `1f537afa9b21a67e...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 03 | `TCGA-G4-6315` | text | `TCGA-G4-6315_text.pt` | 1,574,511 | 1,574,511 | `a9bee20db10025f8...` | `a9bee20db10025f8...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 04 | `TCGA-G4-6315` | vision | `TCGA-G4-6315_vision.pt` | 513,661 | 513,661 | `dc58ae48487e2fb8...` | `dc58ae48487e2fb8...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 05 | `TCGA-G4-6320` | text | `TCGA-G4-6320_text.pt` | 1,574,511 | 1,574,511 | `e7d1b6db777dfadd...` | `e7d1b6db777dfadd...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 06 | `TCGA-G4-6320` | vision | `TCGA-G4-6320_vision.pt` | 3,741,309 | 3,741,309 | `ee8d39e5a52bf1fc...` | `ee8d39e5a52bf1fc...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 07 | `TCGA-G4-6323` | text | `TCGA-G4-6323_text.pt` | 1,574,511 | 1,574,511 | `7b97e1b9a07f0922...` | `7b97e1b9a07f0922...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 08 | `TCGA-G4-6323` | vision | `TCGA-G4-6323_vision.pt` | 1,025,661 | 1,025,661 | `369fa05b8a4ec1b0...` | `369fa05b8a4ec1b0...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 09 | `TCGA-G4-6586` | text | `TCGA-G4-6586_text.pt` | 1,574,511 | 1,574,511 | `f41f28260bddf2be...` | `f41f28260bddf2be...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 10 | `TCGA-G4-6586` | vision | `TCGA-G4-6586_vision.pt` | 2,991,741 | 2,991,741 | `b3ead5eb0c8bf201...` | `b3ead5eb0c8bf201...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 11 | `TCGA-G4-6588` | text | `TCGA-G4-6588_text.pt` | 1,574,511 | 1,574,511 | `fa873c71a0df0c34...` | `fa873c71a0df0c34...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 12 | `TCGA-G4-6588` | vision | `TCGA-G4-6588_vision.pt` | 2,143,869 | 2,143,869 | `7886524fb3fb847d...` | `7886524fb3fb847d...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 13 | `TCGA-G4-6626` | text | `TCGA-G4-6626_text.pt` | 1,574,511 | 1,574,511 | `5f49c1511a5827d8...` | `5f49c1511a5827d8...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 14 | `TCGA-G4-6626` | vision | `TCGA-G4-6626_vision.pt` | 366,205 | 366,205 | `17f8ed069d5333af...` | `17f8ed069d5333af...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 15 | `TCGA-G4-6628` | text | `TCGA-G4-6628_text.pt` | 1,574,511 | 1,574,511 | `20d151801a6604df...` | `20d151801a6604df...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 16 | `TCGA-G4-6628` | vision | `TCGA-G4-6628_vision.pt` | 2,872,957 | 2,872,957 | `e450938d0cdc8103...` | `e450938d0cdc8103...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 17 | `TCGA-G5-6641` | text | `TCGA-G5-6641_text.pt` | 1,574,511 | 1,574,511 | `798d7efef2f6f588...` | `798d7efef2f6f588...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 18 | `TCGA-G5-6641` | vision | `TCGA-G5-6641_vision.pt` | 370,301 | 370,301 | `c4432b3b972493ea...` | `c4432b3b972493ea...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 19 | `TCGA-NH-A5IV` | text | `TCGA-NH-A5IV_text.pt` | 1,574,511 | 1,574,511 | `2745e5a429bfda6e...` | `2745e5a429bfda6e...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 20 | `TCGA-NH-A5IV` | vision | `TCGA-NH-A5IV_vision.pt` | 11,830,909 | 11,830,909 | `976a97808261e985...` | `976a97808261e985...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 21 | `TCGA-NH-A6GA` | text | `TCGA-NH-A6GA_text.pt` | 1,574,511 | 1,574,511 | `68188849828c2da4...` | `68188849828c2da4...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 22 | `TCGA-NH-A6GA` | vision | `TCGA-NH-A6GA_vision.pt` | 1,341,053 | 1,341,053 | `121c8550fdb4ed2b...` | `121c8550fdb4ed2b...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 23 | `TCGA-NH-A6GB` | text | `TCGA-NH-A6GB_text.pt` | 1,574,511 | 1,574,511 | `c9e7f534114376e0...` | `c9e7f534114376e0...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 24 | `TCGA-NH-A6GB` | vision | `TCGA-NH-A6GB_vision.pt` | 2,664,061 | 2,664,061 | `b5ab8f3d7440d1d2...` | `b5ab8f3d7440d1d2...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 25 | `TCGA-NH-A8F8` | text | `TCGA-NH-A8F8_text.pt` | 1,574,511 | 1,574,511 | `dd9e4696e782fe92...` | `dd9e4696e782fe92...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 26 | `TCGA-NH-A8F8` | vision | `TCGA-NH-A8F8_vision.pt` | 1,062,525 | 1,062,525 | `158c1b00edbc8c36...` | `158c1b00edbc8c36...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 27 | `TCGA-NH-A50T` | text | `TCGA-NH-A50T_text.pt` | 1,574,511 | 1,574,511 | `c552070430346ae5...` | `c552070430346ae5...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 28 | `TCGA-NH-A50T` | vision | `TCGA-NH-A50T_vision.pt` | 9,406,077 | 9,406,077 | `4e8ba45cc2c8d1ba...` | `4e8ba45cc2c8d1ba...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 29 | `TCGA-NH-A50U` | text | `TCGA-NH-A50U_text.pt` | 1,574,511 | 1,574,511 | `7daaae9fdd2f0d58...` | `7daaae9fdd2f0d58...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 30 | `TCGA-NH-A50U` | vision | `TCGA-NH-A50U_vision.pt` | 1,840,765 | 1,840,765 | `055e4fed53f8c0af...` | `055e4fed53f8c0af...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 31 | `TCGA-QG-A5YV` | text | `TCGA-QG-A5YV_text.pt` | 1,574,511 | 1,574,511 | `070d08ca800becd9...` | `070d08ca800becd9...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 32 | `TCGA-QG-A5YV` | vision | `TCGA-QG-A5YV_vision.pt` | 2,725,501 | 2,725,501 | `57e05e885b6a306d...` | `57e05e885b6a306d...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 33 | `TCGA-QG-A5Z2` | text | `TCGA-QG-A5Z2_text.pt` | 1,574,511 | 1,574,511 | `e6680769df3eea97...` | `e6680769df3eea97...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 34 | `TCGA-QG-A5Z2` | vision | `TCGA-QG-A5Z2_vision.pt` | 15,083,133 | 15,083,133 | `6d7e16ef583dd625...` | `6d7e16ef583dd625...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 35 | `TCGA-QL-A97D` | text | `TCGA-QL-A97D_text.pt` | 1,574,511 | 1,574,511 | `e50cf131dc541abf...` | `e50cf131dc541abf...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 36 | `TCGA-QL-A97D` | vision | `TCGA-QL-A97D_vision.pt` | 1,431,165 | 1,431,165 | `d4cb475255e4281f...` | `d4cb475255e4281f...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 37 | `TCGA-SS-A7HO` | text | `TCGA-SS-A7HO_text.pt` | 1,574,511 | 1,574,511 | `2be322aec8ba7d4e...` | `2be322aec8ba7d4e...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 38 | `TCGA-SS-A7HO` | vision | `TCGA-SS-A7HO_vision.pt` | 3,487,357 | 3,487,357 | `7d08212bb69fa7fd...` | `7d08212bb69fa7fd...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 39 | `TCGA-WS-AB45` | text | `TCGA-WS-AB45_text.pt` | 1,574,511 | 1,574,511 | `dadb79dd397ddeed...` | `dadb79dd397ddeed...` | **TRANSFERRED_AND_HASH_VERIFIED** |
| 40 | `TCGA-WS-AB45` | vision | `TCGA-WS-AB45_vision.pt` | 1,406,589 | 1,406,589 | `58445e6261d2f1b6...` | `58445e6261d2f1b6...` | **TRANSFERRED_AND_HASH_VERIFIED** |

## 5. Frozen Artifacts Preservation Verification

The following baseline assets were verified against pre-transfer SHA-256 checksums to guarantee strict read-only compliance:

| Artifact Path | Role | Verified SHA-256 Hash | Preservation Status |
|---|---|---|---|
| `models/fusion_model_fp32.onnx` | Frozen ONNX Fusion Model | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **UNMODIFIED** |
| `models/best_fusion_model.pth` | PyTorch Checkpoint | `fe408061100cf8d56d5a61fd7a5ede145f83eb31e95d802cf87d1c4275a05c0d` | **UNMODIFIED** |
| `data/patient_registry.json` | Master Patient Registry | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | **UNMODIFIED** |
| `data/clean_tensors/*.pt` (40 files) | Source Tensor Assets | All 40 source files re-hashed and confirmed identical | **UNMODIFIED** |

## 6. Manual Instructions for Viewing Files on Android Device

### Method 1: Using the Files App on Device (vivo V2125)
1. On the vivo phone, open the **Files** (or **vivo File Manager**) app.
2. Navigate to **Phone Storage** (or **Internal Storage**).
3. Tap on the **Download** folder.
4. Open the **`CRC_Tensors`** folder.
5. You will see all 40 individual `.pt` files corresponding to the 20 approved patients.

### Method 2: Via ADB Shell Command Line
To inspect the directory contents from host PowerShell:

```powershell
& "C:\Android\Sdk\platform-tools\adb.exe" shell ls -la /sdcard/Download/CRC_Tensors/
```

To verify sha256 sums on-device:

```powershell
& "C:\Android\Sdk\platform-tools\adb.exe" shell sha256sum /sdcard/Download/CRC_Tensors/*
```
