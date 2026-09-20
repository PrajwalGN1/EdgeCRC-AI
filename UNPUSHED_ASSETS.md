# Unpushed Heavy Assets — Where to Place Them

This repo excludes 10GB+ heavy files for GitHub's 100MB/1GB limits. Clone is lightweight (<50MB). To run the Streamlit app locally, restore the heavy assets as below.

## What is NOT pushed (via .gitignore)

| Path (relative to repo root) | Size | Why excluded | Where to get |
|------------------------------|------|--------------|--------------|
| `archive/` | 12G | Kaggle dump `MSIMUT_JPEG/` (75k) + `MSS_JPEG/` (117k) = 192k JPGs | Already in this project's `archive/` on local disk — copy from `/home/moki27/project/Colorectal_cancer_detect-20260919T075001Z-1-001/Colorectal_cancer_detect/archive/` or download below |
| `archive.zip` | 11G | Same as above zipped | Kaggle: https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg |
| `data/clean_tensors/*.pt` | 1.2G (653 files) | Frozen Phikon-v2 1024-D + ClinicalBERT 768-D tensors | 1) Copy from `data/clean_tensors/` on local disk, or 2) Regenerate: `python src/multimodal_dataset.py` needs `archive/` JPEGs + `patient_registry.json` + Phikon-v2/ClinicalBERT, or 3) Download from Kaggle + run feature extraction (see `data/DATASET.md`) |
| `data/MSIMUT_JPEG/*.jpg` | 2.3G | Symlink target `archive/MSIMUT_JPEG/` | Auto-wired via symlink `data/MSIMUT_JPEG -> archive/MSIMUT_JPEG` — after restoring `archive/`, run: `ln -s ../archive/MSIMUT_JPEG data/MSIMUT_JPEG && ln -s ../archive/MSS_JPEG data/MSS_JPEG` |
| `data/MSS_JPEG/*.jpg` | 3.6G | Same | Same as above |
| `models/*.pth` (optional) | 91M (4 × 21M + 2×5M) | Checkpoints — **currently PUSHED** (21M < 100MB limit) | If you uncomment `models/*.pth` in `.gitignore`, download from Releases: `models/fusion_model_v1_1.pth` (primary, best AUC 0.915) |
| `data/patient_registry.json` | 13M | Master index 326 patients — **PUSHED** (small enough) | Already in repo |
| `Unconfirmed*.crdownload` | 1.7G | Partial Chrome download | Delete |

## Quick restore (local machine that has the archive)

```bash
# From the original machine:
rsync -av --progress archive/ /path/to/clone/archive/
rsync -av --progress data/clean_tensors/ /path/to/clone/data/clean_tensors/
# Or recreate symlinks:
cd /path/to/clone
ln -s ../archive/MSIMUT_JPEG data/MSIMUT_JPEG
ln -s ../archive/MSS_JPEG data/MSS_JPEG
```

## Fresh download (new machine, no archive)

```bash
# 1. Kaggle raw JPEGs (192k, ~12GB)
pip install kaggle --break-system-packages
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
cd Colorectal_cancer_detect/data
kaggle datasets download -d joangibert/tcga_coad_msi_mss_jpg -p . --unzip
# → data/MSIMUT_JPEG/ + data/MSS_JPEG/ (or archive/ if you prefer)
# Move to archive/ for consistency:
mkdir -p ../archive && mv MSIMUT_JPEG MSS_JPEG ../archive/
ln -s ../archive/MSIMUT_JPEG MSIMUT_JPEG
ln -s ../archive/MSS_JPEG MSS_JPEG

# 2. Clean tensors (1.2G) — either copy or regenerate
# Option A: regenerate (needs Phikon-v2 + ClinicalBERT, ~30 min on CPU)
cd ../src
python multimodal_dataset.py  # writes data/clean_tensors/

# Option B: if you have a backup of clean_tensors/, just copy:
# rsync -av /backup/clean_tensors/ ../data/clean_tensors/
```

## Verify after restore

```bash
ls archive/MSIMUT_JPEG | wc -l  # 75039
ls archive/MSS_JPEG | wc -l     # 117273
ls data/clean_tensors | wc -l   # 653
ls data/clean_tensors/*.pt -lh | head -3
python3 -c "import torch; print(torch.load('data/clean_tensors/TCGA-AA-3877_vision.pt', weights_only=True).shape)"
# → torch.Size([N, 1024])  and text [1,768]

# Run app
streamlit run src/ui/app.py --server.port 8501
# → http://localhost:8501 (if port 8501 busy, try 8502)
```

## Why this split?

- GitHub hard limit: 100MB/file, soft 1GB/repo. Pushing 12G would break clone and CI.
- `data/patient_registry.json` + `models/*.pth` + `docs/` + `src/` = <150MB, clones in seconds.
- Heavy assets are reproducible from Kaggle + frozen encoders — no need to version them.

## For maintainers: If repo size grows, enable Git LFS

```bash
git lfs track "*.pth"
git lfs track "data/clean_tensors/*.pt"
git add .gitattributes
```

See also: `data/DATASET.md` (Kaggle link + 3 download options) + `docs/HANDOFF.md` (operational playbook).
