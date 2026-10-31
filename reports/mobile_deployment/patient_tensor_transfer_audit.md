# Patient Tensor Transfer Audit (Phase A)

## 1. Audit Overview

- **Repository Root**: `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`
- **Authoritative Source Tensor Directory**: `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2\data\clean_tensors`
- **Total Approved Patients**: 20
- **Total Expected Files**: 40 (20 Text tensors + 20 Vision tensors)
- **Total Files Found**: 40
- **Missing Files Count**: 0
- **Duplicate / Ambiguous Files Count**: 0
- **Case Mismatch Count**: 0
- **Registry Verification**: All 20 patient IDs verified present in `data/patient_registry.json`
- **Integrity Status**: All 40 tensors verified readable via PyTorch CPU loader (shapes & dtypes validated), SHA-256 hashes generated.

## 2. Source Tensor Inventory Table

| Patient ID | Text file | Vision file | Source size | Integrity status |
|---|---|---|---|---|
| `TCGA-G4-6311` | `TCGA-G4-6311_text.pt` | `TCGA-G4-6311_vision.pt` | 1,574,511 B (text) / 2,217,597 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G4-6315` | `TCGA-G4-6315_text.pt` | `TCGA-G4-6315_vision.pt` | 1,574,511 B (text) / 513,661 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G4-6320` | `TCGA-G4-6320_text.pt` | `TCGA-G4-6320_vision.pt` | 1,574,511 B (text) / 3,741,309 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G4-6323` | `TCGA-G4-6323_text.pt` | `TCGA-G4-6323_vision.pt` | 1,574,511 B (text) / 1,025,661 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G4-6586` | `TCGA-G4-6586_text.pt` | `TCGA-G4-6586_vision.pt` | 1,574,511 B (text) / 2,991,741 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G4-6588` | `TCGA-G4-6588_text.pt` | `TCGA-G4-6588_vision.pt` | 1,574,511 B (text) / 2,143,869 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G4-6626` | `TCGA-G4-6626_text.pt` | `TCGA-G4-6626_vision.pt` | 1,574,511 B (text) / 366,205 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G4-6628` | `TCGA-G4-6628_text.pt` | `TCGA-G4-6628_vision.pt` | 1,574,511 B (text) / 2,872,957 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-G5-6641` | `TCGA-G5-6641_text.pt` | `TCGA-G5-6641_vision.pt` | 1,574,511 B (text) / 370,301 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-NH-A5IV` | `TCGA-NH-A5IV_text.pt` | `TCGA-NH-A5IV_vision.pt` | 1,574,511 B (text) / 11,830,909 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-NH-A6GA` | `TCGA-NH-A6GA_text.pt` | `TCGA-NH-A6GA_vision.pt` | 1,574,511 B (text) / 1,341,053 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-NH-A6GB` | `TCGA-NH-A6GB_text.pt` | `TCGA-NH-A6GB_vision.pt` | 1,574,511 B (text) / 2,664,061 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-NH-A8F8` | `TCGA-NH-A8F8_text.pt` | `TCGA-NH-A8F8_vision.pt` | 1,574,511 B (text) / 1,062,525 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-NH-A50T` | `TCGA-NH-A50T_text.pt` | `TCGA-NH-A50T_vision.pt` | 1,574,511 B (text) / 9,406,077 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-NH-A50U` | `TCGA-NH-A50U_text.pt` | `TCGA-NH-A50U_vision.pt` | 1,574,511 B (text) / 1,840,765 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-QG-A5YV` | `TCGA-QG-A5YV_text.pt` | `TCGA-QG-A5YV_vision.pt` | 1,574,511 B (text) / 2,725,501 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-QG-A5Z2` | `TCGA-QG-A5Z2_text.pt` | `TCGA-QG-A5Z2_vision.pt` | 1,574,511 B (text) / 15,083,133 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-QL-A97D` | `TCGA-QL-A97D_text.pt` | `TCGA-QL-A97D_vision.pt` | 1,574,511 B (text) / 1,431,165 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-SS-A7HO` | `TCGA-SS-A7HO_text.pt` | `TCGA-SS-A7HO_vision.pt` | 1,574,511 B (text) / 3,487,357 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |
| `TCGA-WS-AB45` | `TCGA-WS-AB45_text.pt` | `TCGA-WS-AB45_vision.pt` | 1,574,511 B (text) / 1,406,589 B (vision) | VERIFIED_OK (text) / VERIFIED_OK (vision) |

## 3. Comprehensive File-by-File Integrity Details

| Filename | Modality | Size (Bytes) | Tensor Shape | Dtype | SHA-256 Hash |
|---|---|---|---|---|---|
| `TCGA-G4-6311_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `2974f2e3a2e411bb36d87e385c37e38416063398ac839e4d45853a9d1f8d536a` |
| `TCGA-G4-6311_vision.pt` | vision | 2217597 | [541, 1024] | torch.float32 | `1f537afa9b21a67e5afe7a047f14077e0f1572778d8974443eba41b3719e8a8b` |
| `TCGA-G4-6315_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `a9bee20db10025f8d8f1a88fb72bbf83bff9ca89ccd518603ac9363c4791cbd5` |
| `TCGA-G4-6315_vision.pt` | vision | 513661 | [125, 1024] | torch.float32 | `dc58ae48487e2fb8a0c68bb22cd8ed1ed01a18028b0239076d1bc6f8b2affef5` |
| `TCGA-G4-6320_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `e7d1b6db777dfadd058f94a022698c10217cc75b802d4df6de40a93a33141eb9` |
| `TCGA-G4-6320_vision.pt` | vision | 3741309 | [913, 1024] | torch.float32 | `ee8d39e5a52bf1fc555cc4ce89f96de8fc416b95cfc4d5a6b9d1c48080d5fa90` |
| `TCGA-G4-6323_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `7b97e1b9a07f09220613bae71cd7a2accdbd18f78c331e3fd7072aa8cbe23b08` |
| `TCGA-G4-6323_vision.pt` | vision | 1025661 | [250, 1024] | torch.float32 | `369fa05b8a4ec1b0a4947e9ef89692ec73ac21045c30505c180974117a7564cf` |
| `TCGA-G4-6586_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `f41f28260bddf2bed2bbe322a5d63ee7010db854d53bb9000056655ba9d9d00c` |
| `TCGA-G4-6586_vision.pt` | vision | 2991741 | [730, 1024] | torch.float32 | `b3ead5eb0c8bf20158cc6fb49ada86d2a767a0457e0271d5a254526b6b3a8e76` |
| `TCGA-G4-6588_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `fa873c71a0df0c348cb86ca6b958aeeead2eceb26fee2f2d7e6360d7fbf42806` |
| `TCGA-G4-6588_vision.pt` | vision | 2143869 | [523, 1024] | torch.float32 | `7886524fb3fb847d757032c5186fe3afed2658d87c16d4235c378ccbc948be81` |
| `TCGA-G4-6626_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `5f49c1511a5827d8c8e5ec61209aaa834c690f9f28ef0795ed443956410b1d6b` |
| `TCGA-G4-6626_vision.pt` | vision | 366205 | [89, 1024] | torch.float32 | `17f8ed069d5333afeecffbd5b5a7478b65f9011239a3fc0e21cbe00b8c41f245` |
| `TCGA-G4-6628_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `20d151801a6604dfebf87e6002b9b1f48d3ad83e029e0b766be896496d0aca6d` |
| `TCGA-G4-6628_vision.pt` | vision | 2872957 | [701, 1024] | torch.float32 | `e450938d0cdc81035ebd3ee74b9906ee8ced5998c164a28a90b337fec80a6f8d` |
| `TCGA-G5-6641_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `798d7efef2f6f5884eb37eed69132222dfc6a4c0a70583bd6c88c23d16a339a5` |
| `TCGA-G5-6641_vision.pt` | vision | 370301 | [90, 1024] | torch.float32 | `c4432b3b972493eaf4e0c5b16eb2bc8c3577341aea00f12a47b67303103212a1` |
| `TCGA-NH-A5IV_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `2745e5a429bfda6e2916460d5ef1ab0b17eef94961fe8f4003dac492abfb9654` |
| `TCGA-NH-A5IV_vision.pt` | vision | 11830909 | [2888, 1024] | torch.float32 | `976a97808261e9858c1fddb4e1ba0feb11827d43ee3c4e0f3f0dd72cbcaaf6f1` |
| `TCGA-NH-A6GA_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `68188849828c2da4c253dae40a2d3d49554634cf006fe0f0b158c17a0de40d6d` |
| `TCGA-NH-A6GA_vision.pt` | vision | 1341053 | [327, 1024] | torch.float32 | `121c8550fdb4ed2b6b6dc7129c0dd6ee57ebe6c53b32bc4611cb2644e2147377` |
| `TCGA-NH-A6GB_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `c9e7f534114376e02aba0819ab8c6f46d80b5f22503aa23dfcadc2d565e289cd` |
| `TCGA-NH-A6GB_vision.pt` | vision | 2664061 | [650, 1024] | torch.float32 | `b5ab8f3d7440d1d2ade320c578bba61df1e9d3507aebc64467f4d014b09053cf` |
| `TCGA-NH-A8F8_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `dd9e4696e782fe921cebe5d0e0ba190d81ecab78f329e1f34d43ad575ea76a2e` |
| `TCGA-NH-A8F8_vision.pt` | vision | 1062525 | [259, 1024] | torch.float32 | `158c1b00edbc8c369fdbf8f53a2f2c92087256dd875512eb4b1111926f77f9c0` |
| `TCGA-NH-A50T_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `c552070430346ae5f7d60061f7a025c55058ea322eaf4225f2c9a6498b7d27be` |
| `TCGA-NH-A50T_vision.pt` | vision | 9406077 | [2296, 1024] | torch.float32 | `4e8ba45cc2c8d1bad56638b45803b852f158ac7b8d41d26c94bf4809482a55bb` |
| `TCGA-NH-A50U_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `7daaae9fdd2f0d5879911eb0cb8e2a9be586ff3356d00f39987bf91d3e35f353` |
| `TCGA-NH-A50U_vision.pt` | vision | 1840765 | [449, 1024] | torch.float32 | `055e4fed53f8c0afd4affb647c9c73bd9acaae273b3d3be660bd2c5912af799f` |
| `TCGA-QG-A5YV_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `070d08ca800becd91882e9e04cdac13fbf01692ef2e5f003500c1e1b55a6185e` |
| `TCGA-QG-A5YV_vision.pt` | vision | 2725501 | [665, 1024] | torch.float32 | `57e05e885b6a306da769ea90f92f812aea39072fad8e9128c1ba6c203c640fd1` |
| `TCGA-QG-A5Z2_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `e6680769df3eea97ddc75128149960c699137a8d08d059a918682136057c59f4` |
| `TCGA-QG-A5Z2_vision.pt` | vision | 15083133 | [3682, 1024] | torch.float32 | `6d7e16ef583dd6257018d357ccfc61dbd7ff1ef25527995352a2b3152138637e` |
| `TCGA-QL-A97D_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `e50cf131dc541abfa7c31c7c16e29b25ed5d14644302b6572fe9e38bb8f41b06` |
| `TCGA-QL-A97D_vision.pt` | vision | 1431165 | [349, 1024] | torch.float32 | `d4cb475255e4281fe72c5924b96d1d6914dadd2da25a71390e7617f0223801f5` |
| `TCGA-SS-A7HO_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `2be322aec8ba7d4e522d629b9678da40d01d8928acb601948bb0e1957d0dc725` |
| `TCGA-SS-A7HO_vision.pt` | vision | 3487357 | [851, 1024] | torch.float32 | `7d08212bb69fa7fd4f58f1f51f157d9cb2c3a9be5be667c677dc712c136069b6` |
| `TCGA-WS-AB45_text.pt` | text | 1574511 | [1, 768] | torch.float32 | `dadb79dd397ddeed92a1e5c754468acbb17dd96b53a24fae32ee34de2a19a0de` |
| `TCGA-WS-AB45_vision.pt` | vision | 1406589 | [343, 1024] | torch.float32 | `58445e6261d2f1b6bc1e6e93c4b7fd661629ee229e95bb3a19a5a80448ea414c` |
