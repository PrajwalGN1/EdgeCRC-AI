# CRC Edge AI Patient CRCBUNDLE Export and Android Transfer Report

## Executive Summary

- **Deployment Status**: **100% SUCCESS — FULLY EXPORTED, TRANSFERRED & VERIFIED**
- **Approved Patient Roster**: 20 patient pairs
- **Total `.crcbundle` Containers**: 20 individual files
- **Total Bytes Transferred**: 68,553,436 bytes (65.38 MB)
- **Format Specification**: Native CRCB Binary Container (v2), Little-Endian Float32, cryptographic SHA-256 manifest
- **Target Android Device**: `vivo V2125` (Product: `2020iM`, Serial: `1377355236000F7`, Android 12 / SDK 31)
- **On-Device Destination**: `/sdcard/Download/CRCBundle/`
- **No-Archive Policy**: **ENFORCED** — 0 combined ZIP/TAR archives created. All 20 files pushed and verified individually.
- **Frozen Artifact Integrity**: **PRESERVED** — FP32 ONNX model, PyTorch checkpoint, patient registry, and source tensors remain completely unmodified.

## 1. Verified Patient Roster

All 20 user-approved patient IDs:
1. `TCGA-G4-6311`
2. `TCGA-G4-6315`
3. `TCGA-G4-6320`
4. `TCGA-G4-6323`
5. `TCGA-G4-6586`
6. `TCGA-G4-6588`
7. `TCGA-G4-6626`
8. `TCGA-G4-6628`
9. `TCGA-G5-6641`
10. `TCGA-NH-A5IV`
11. `TCGA-NH-A6GA`
12. `TCGA-NH-A6GB`
13. `TCGA-NH-A8F8`
14. `TCGA-NH-A50T`
15. `TCGA-NH-A50U`
16. `TCGA-QG-A5YV`
17. `TCGA-QG-A5Z2`
18. `TCGA-QL-A97D`
19. `TCGA-SS-A7HO`
20. `TCGA-WS-AB45`


## 2. Container Format and Compatibility

The exported bundles strictly reuse the project's existing authentic feature bundle specification implemented in `tools/export_specimen_bundle.py` and ingested by `AuthenticBundleLoader.kt` on Android:
- **Container Magic**: `CRCB` (0x43, 0x52, 0x43, 0x42)
- **Schema Version**: `2` (Int32 Little-Endian)
- **Specimen Identifier**: UTF-8 encoded string matching patient ID
- **Tensor Dimensions**: Text: `[1, 768]` Float32, Vision: `[1, N, 1024]` Float32 (where $N$ is authentic patch count)
- **Payload Integrity**: Internal 32-byte SHA-256 digests for vision and text binary payloads
- **Value Finiteness**: Zero tolerance for NaN / Inf values enforced during export and verified during round-trip validation

## 3. Comprehensive Patient Export and Transfer Ledger

| # | Patient ID | Patches ($N$) | Source Vision Size (B) | Source Text Size (B) | Bundle Size (B) | Bundle SHA-256 | Android Status |
|---|---|---|---|---|---|---|---|
| 01 | `TCGA-G4-6311` | 541 | 2,217,597 | 1,574,511 | 2,219,147 | `8a476df1a180a11b...` | **TRANSFERRED_AND_VERIFIED** |
| 02 | `TCGA-G4-6315` | 125 | 513,661 | 1,574,511 | 515,211 | `3ef3bd2d3f16c82e...` | **TRANSFERRED_AND_VERIFIED** |
| 03 | `TCGA-G4-6320` | 913 | 3,741,309 | 1,574,511 | 3,742,859 | `0a973ddd81509c72...` | **TRANSFERRED_AND_VERIFIED** |
| 04 | `TCGA-G4-6323` | 250 | 1,025,661 | 1,574,511 | 1,027,211 | `c4d8247cb016b802...` | **TRANSFERRED_AND_VERIFIED** |
| 05 | `TCGA-G4-6586` | 730 | 2,991,741 | 1,574,511 | 2,993,291 | `2999e8c7c5b3ef29...` | **TRANSFERRED_AND_VERIFIED** |
| 06 | `TCGA-G4-6588` | 523 | 2,143,869 | 1,574,511 | 2,145,419 | `174edd2db92dffd6...` | **TRANSFERRED_AND_VERIFIED** |
| 07 | `TCGA-G4-6626` | 89 | 366,205 | 1,574,511 | 367,755 | `0f4717f3c73f73fb...` | **TRANSFERRED_AND_VERIFIED** |
| 08 | `TCGA-G4-6628` | 701 | 2,872,957 | 1,574,511 | 2,874,507 | `16f43ca08cc66250...` | **TRANSFERRED_AND_VERIFIED** |
| 09 | `TCGA-G5-6641` | 90 | 370,301 | 1,574,511 | 371,851 | `cb95dd284732a4e3...` | **TRANSFERRED_AND_VERIFIED** |
| 10 | `TCGA-NH-A5IV` | 2888 | 11,830,909 | 1,574,511 | 11,832,459 | `ea0541287a4b4299...` | **TRANSFERRED_AND_VERIFIED** |
| 11 | `TCGA-NH-A6GA` | 327 | 1,341,053 | 1,574,511 | 1,342,603 | `2277d53698efd81b...` | **TRANSFERRED_AND_VERIFIED** |
| 12 | `TCGA-NH-A6GB` | 650 | 2,664,061 | 1,574,511 | 2,665,611 | `0db45082cc87aaf0...` | **TRANSFERRED_AND_VERIFIED** |
| 13 | `TCGA-NH-A8F8` | 259 | 1,062,525 | 1,574,511 | 1,064,075 | `6bcb3bb6970fb623...` | **TRANSFERRED_AND_VERIFIED** |
| 14 | `TCGA-NH-A50T` | 2296 | 9,406,077 | 1,574,511 | 9,407,627 | `f8ae865f208ed72b...` | **TRANSFERRED_AND_VERIFIED** |
| 15 | `TCGA-NH-A50U` | 449 | 1,840,765 | 1,574,511 | 1,842,315 | `14079e1c5bb1b25a...` | **TRANSFERRED_AND_VERIFIED** |
| 16 | `TCGA-QG-A5YV` | 665 | 2,725,501 | 1,574,511 | 2,727,051 | `14df9a91b95ac6f0...` | **TRANSFERRED_AND_VERIFIED** |
| 17 | `TCGA-QG-A5Z2` | 3682 | 15,083,133 | 1,574,511 | 15,084,683 | `c2654bfd05749c50...` | **TRANSFERRED_AND_VERIFIED** |
| 18 | `TCGA-QL-A97D` | 349 | 1,431,165 | 1,574,511 | 1,432,715 | `11b51857a667f6ba...` | **TRANSFERRED_AND_VERIFIED** |
| 19 | `TCGA-SS-A7HO` | 851 | 3,487,357 | 1,574,511 | 3,488,907 | `af3c179be65c3a76...` | **TRANSFERRED_AND_VERIFIED** |
| 20 | `TCGA-WS-AB45` | 343 | 1,406,589 | 1,574,511 | 1,408,139 | `f0f3122dd3825c06...` | **TRANSFERRED_AND_VERIFIED** |

## 4. Verification and Integrity Results

1. **Source Pair Validation**: 20/20 patient pairs found in `data/clean_tensors/`. Shapes `(1, 768)` and `(N, 1024)` validated with PyTorch.
2. **Zero Numerical Loss**: Every `.crcbundle` was verified post-export by extracting binary tensors and computing absolute difference against upstream PyTorch tensors; maximum difference = `0.0`.
3. **Device Destination Audit**: Pre-transfer check established `/sdcard/Download/CRCBundle/` had 0 conflicts. Storage space: 1.6 GB available.
4. **Individual ADB Push**: All 20 files pushed one-by-one via ADB without archive creation.
5. **On-Device Size Verification**: `stat -c %s` on Android returned exact byte-for-byte size matches for all 20 files.
6. **On-Device SHA-256 Verification**: `/system/bin/sha256sum` executed on the Android device confirmed 20/20 matching SHA-256 digests.
7. **Independent Enumeration**: Post-transfer enumeration of `/sdcard/Download/CRCBundle/` confirmed exactly 20 `.crcbundle` files and 0 extraneous files.

## 5. Frozen Artifacts Preservation Proof

| Baseline Asset | Path | Baseline Hash | Post-Transfer Hash | Status |
|---|---|---|---|---|
| Frozen ONNX Model | `models/fusion_model_fp32.onnx` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | **UNMODIFIED** |
| PyTorch Checkpoint | `models/best_fusion_model.pth` | `fe408061100cf8d56d5a61fd7a5ede145f83eb31e95d802cf87d1c4275a05c0d` | `fe408061100cf8d56d5a61fd7a5ede145f83eb31e95d802cf87d1c4275a05c0d` | **UNMODIFIED** |
| Patient Registry | `data/patient_registry.json` | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2` | **UNMODIFIED** |
| Source Tensors | `data/clean_tensors/*.pt` (40 files) | Pre-transfer SHA-256 verified | Re-hashed post-transfer | **UNMODIFIED** |

## 6. Exact Commands Executed

```powershell
# Export and verification
python scratch/export_and_verify_crcbundles.py

# Check device connection
& "C:\Android\Sdk\platform-tools\adb.exe" devices -l

# Create target directory
& "C:\Android\Sdk\platform-tools\adb.exe" -s 1377355236000F7 shell mkdir -p /sdcard/Download/CRCBundle

# Push individual bundle (example)
& "C:\Android\Sdk\platform-tools\adb.exe" -s 1377355236000F7 push "exports/crcbundles/TCGA-G4-6311.crcbundle" /sdcard/Download/CRCBundle/TCGA-G4-6311.crcbundle

# Verify on-device size and SHA-256
& "C:\Android\Sdk\platform-tools\adb.exe" -s 1377355236000F7 shell stat -c %s /sdcard/Download/CRCBundle/TCGA-G4-6311.crcbundle
& "C:\Android\Sdk\platform-tools\adb.exe" -s 1377355236000F7 shell sha256sum /sdcard/Download/CRCBundle/TCGA-G4-6311.crcbundle
```

## 7. Android Application Import Instructions

### Ingesting via Android App UI
1. Launch the **CRC Multimodal Edge AI Workstation** app on the vivo phone.
2. Navigate to **New Specimen** (`+` button).
3. Under **Input Provenance / Feature Bundle**, tap **Choose Bundle File** (`btnSelectBundleFile`).
4. In the Android Storage Access Framework (SAF) system picker, open **Downloads** > **CRCBundle**.
5. Select any `.crcbundle` file (e.g. `TCGA-G4-6311.crcbundle`).
6. The app parses the `CRCB` magic header, validates SHA-256 checksums, displays the specimen ID and authentic patch count, and enables immediate on-device inference.
