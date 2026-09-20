# Dataset — TCGA-COAD MSI/MSS JPG

## Canonical Kaggle Link

**Primary (from `data/Download_DataSet.pdf:2`):**
```
https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg
```

> The PDF truncates as `joangibert/tcga_coad_msi_mss_jpg` (full slug `tcga_coad_msi_mss_jpg`, file `tcga_coad_msi_mss_jpg.zip`). If Kaggle returns 404, search `tcga_coad_msi_mss` on Kaggle or use the TCGA-COAD GDC directly.

**Alternative mirrors (if primary is delisted):**
- Search: `https://www.kaggle.com/datasets/search?query=tcga+coad+msi`
- TCGA GDC: `https://portal.gdc.cancer.gov/projects/TCGA-COAD` (raw SVS + reports, requires tiling to 224×224)

## What the Kaggle Dataset Contains

| Content | Path in Kaggle zip | Size |
|---------|-------------------|------|
| MSI patches (JPEG, 224×224 @10×) | `MSIMUT_JPEG/` | ~192k images total, ~63 patients |
| MSS patches | `MSS_JPEG/` | ~263 patients |
| Clinical reports (aligned) | `clinical_reports/` or `metadata.csv` | 9236 raw → 326 aligned via TCGA barcode |
| This project's processed form | `data/clean_tensors/` + `data/patient_registry.json` | 653 tensors (326×2), already committed |

## What You Already Have (No Download Needed for Inference)

```
data/
├── patient_registry.json       # 326 patients, 5 fields: label, class_name, report_text, image_paths, patch_count
├── clean_tensors/              # 653 frozen tensors
│   ├── TCGA-AA-3877_vision.pt # [N×1024] Phikon-v2
│   └── TCGA-AA-3877_text.pt   # [1×768] ClinicalBERT
└── Download_DataSet.pdf        # original link + "extract in data/" instruction
```

**Inference (`src/ui/app.py`) uses ONLY the processed tensors + registry** — no Kaggle download required to run the app.

## When You DO Need Kaggle Raw

- To show real JPEG gallery (vs placeholder) in Streamlit Top-5
- To retrain with new tiling, stain norm, or patch size
- To audit patch quality (remove pen marks, folds)

## Download & Setup — Step by Step

### Option A: Kaggle CLI (recommended)

```bash
# 1. Get Kaggle API token: https://www.kaggle.com/settings → Create New Token → downloads kaggle.json
pip install kaggle --break-system-packages
mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle --version  # should print 1.x

# 2. Download
cd Colorectal_cancer_detect/data
kaggle datasets download -d joangibert/tcga_coad_msi_mss_jpg -p . --unzip
# This may take 20–40 min (~12 GB). Pass -p . to land in data/

# 3. Verify
ls -lh | head          # expect MSIMUT_JPEG/, MSS_JPEG/
find MSIMUT_JPEG -type f | wc -l   # ~ many thousand
python3 -c "import json; print(len(json.load(open('patient_registry.json'))))"
# 326
```

### Option B: Browser

1. Visit `https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg`
2. Click `Download` → `tcga_coad_msi_mss_jpg.zip` (~12 GB)
3. Unzip: `unzip tcga_coad_msi_mss_jpg.zip -d data/`

### Option C: Already Preprocessed (use as-is)

Nothing to do. Confirm:

```bash
ls data/clean_tensors | wc -l   # 653
python3 -c "import torch; print(torch.load('data/clean_tensors/TCGA-AA-3877_vision.pt', weights_only=True).shape)"
# torch.Size([N, 1024])
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `403 Forbidden` from Kaggle | Token not set or `chmod 600` missed; re-copy `kaggle.json` |
| `Dataset not found` | Link moved — search Kaggle `tcga_coad_msi` or use GDC SVS + tile script |
| `MSIMUT_JPEG not found` after unzip | Zip may contain nested folder — `find data -type d -name "*MUT*"` to locate |
| `torch.load` fails | Use `weights_only=True` (PyTorch 2.9 requirement); tensors are CPU-mapped |
| App shows placeholder gallery | Add `data/MSIMUT_JPEG/` & `data/MSS_JPEG/` (see Option A) — optional |

## Provenance Notes

- Source cohort: TCGA Colorectal Adenocarcinoma (COAD), H&E stained, 10× magnification, Reinhard stain-normalized, Otsu foreground-filtered (>50% background removed).
- Label: `1=MSI, 0=MSS` (63 MSI / 263 MSS → pos_weight 4.17).
- Reports: ~9k pathology notes, filtered to 326 with PHI stripping, aligned by TCGA barcode.
- Tensors: Phikon-v2 (`owkin/phikon-v2`, DINOv2 ViT-L/16, 307M) and ClinicalBERT (BioBERT→MIMIC-III, 110M), frozen, `torch.no_grad()`, deterministic.

## Quick Link Card (for handoff slides)

```
Dataset: tcga_coad_msi_mss_jpg (TCGA-COAD, H&E + reports)
Link:    https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg
Size:    ~12 GB raw, 653 tensors (processed) already in repo
Patients: 326 (63 MSI / 263 MSS), Val: 66 isolated (50/16)
Patches: 192,314 @224×224, max 1000/patient (randperm cap)
Tensors: V [N×1024] (Phikon-v2) + T [1×768] (ClinicalBERT)
Contact: See PB05_Report_8thSem.pdf Ch.4.2 + data/Download_DataSet.pdf
```
