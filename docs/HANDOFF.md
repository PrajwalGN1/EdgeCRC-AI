# Technical Documentation — System Handoff

> **System documentation:** Operational playbook — what works, what's wired, what to touch (and what not to). Professional handover for deployment and maintenance.

---

## 0. TL;DR — What You Are Inheriting

A **production-ready Streamlit app** that loads frozen Phikon-v2 + ClinicalBERT tensors for any TCGA patient in the registry, runs `SummaryToPatchCrossAttention` (prob + attention), shows Top-5 patches, optionally renders Phikon-v2 cellular heatmaps, and drafts a 3-paragraph clinical summary via Ollama. **All training is done** — weights are checked in under `models/`. Just run it.

**Kaggle dataset (already preprocessed, but link for resync):**
`https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg` — see `data/Download_DataSet.pdf` and §6 below.

---

## 1. Current Working State ✅

| Capability | Status | File | Notes |
|------------|--------|------|-------|
| Cross-Attention model training | ✅ Done, best AUC 0.9150 | `src/cross_attentiion_train.py:45` | 30 epochs, AdamW, weighted BCE 4.17, τ=0.83 |
| Late Fusion baseline | ✅ Done, AUC 0.895 | `src/architectures/late_fusion_model.py:5` | Kept for ablation/comparison |
| Frozen tensor store | ✅ 653 files | `data/clean_tensors/*` | 326 patients, no re-encoding needed |
| Registry + validation contract | ✅ Locked | `data/patient_registry.json`, `results/actual_validation_keys.json` | 66 isolated val patients — **do not shuffle** |
| Evaluation (CM, ROC, report) | ✅ Generated | `results/*.png`, `*.txt`, `src/cross_attention_eval.py:60` | Run eval to reproduce |
| XAI Tier 1 (Top-5) | ✅ Wired | `src/generate_xai.py:16`, `src/ui/app.py:166` | Argsort, 5-col gallery |
| XAI Tier 2 (Phikon heatmap) | ✅ Wired (optional) | `src/generate_pixel_xai.py:53` | Requires `owkin/phikon-v2` cache; graceful fallback if offline |
| XAI Tier 3 (LLM scribe) | ✅ Wired (with fallback) | `src/ui/app.py:61` | Tries Ollama, else templated report — never crashes |
| Streamlit app | ✅ Production-fixed | `src/ui/app.py` (457 lines) | Path-robust, device-agnostic, placeholder gallery if JPEGs missing |
| Diagrams | ✅ 10 generated | `docs/diagrams/01..10_*.png` | Regenerate via `docs/generate_diagrams.py` |
| Docker | ✅ Ready | `Dockerfile`, `.streamlit/config.toml` | `streamlit run src/ui/app.py` |

**Verified smoke test (2026-09-19):**
```bash
python3 -c "from src.architectures.cross_attention_fusion import SummaryToPatchCrossAttention; ..."
# model ok 5,250,561 params, forward prob 0.60 attn [1×1×343] on TCGA sample
# Streamlit boots on :8502 — /_stcore/health returns 200
```

---

## 2. What Was Fixed for You (Don't Regress)

1. **Path fragility** — Legacy `app.py` used `../../data/...` relative to CWD; now uses `Path(__file__).resolve().parents[2]` so it works from any `streamlit run` invocation (project root or `src/ui/`). See `src/ui/app.py:17`.
2. **Missing JPEG handling** — `data/MSIMUT_JPEG` / `MSS_JPEG` may not be present in a clean checkout. App now shows colored placeholder + score instead of crashing, and histogram expander still works. Real JPEGs can be added later (see §6).
3. **Ollama offline** — Scribe now has `generate_clinical_report()` fallback templated text; Streamlit never shows raw error unless you want it. Sidebar probes `localhost:11434` and shows status.
4. **Weight candidates** — Tries `fusion_model_v1_1.pth`, `best_fusion_model.pth`, `fusion_model_v1_0.pth` in order with `weights_only=True` loading.
5. **Device routing** — Auto `cuda → mps (Apple M4) → cpu` with no hardcoding.
6. **Sidebar + session_state** — Persistent across reruns; dropdown+slider correctly rebuild gallery without re-inference.
7. **.streamlit/config.toml + Dockerfile** — Attached for Streamlit Cloud / HF Spaces / EC2.

**Keep `src/ui/app_legacy.py` as reference; do not delete until your team signs off.**

---

## 3. How to Run (3 Ways)

### A. Local (dev)
```bash
cd Colorectal_cancer_detect
pip install -r requirements.txt               # 75 pkgs, includes torch==2.9.1, streamlit==1.52.2
streamlit run src/ui/app.py --server.port 8501
# Optional: start Ollama for live scribe
ollama serve &
ollama pull qwen2.5-coder:7b
```

### B. Docker (prod parity)
```bash
docker build -t crc-msi .
docker run -p 8501:8501 crc-msi
# + optional --network host if Ollama is on host
```

### C. Streamlit Cloud / Hugging Face Spaces
- Push repo, set `Main file path: src/ui/app.py`
- Add `packages.txt` if needed: `libgl1`, `libglib2.0-0` already in Dockerfile; for cloud, Streamlit Cloud handles it.
- **Note:** Phikon-v2 heatmap (Tier 2) downloads ~1.2 GB `owkin/phikon-v2` on first run; mark `HAS_XAI` optional so build doesn't fail offline. Scribe fallback ensures deploy works without Ollama.

---

## 4. How to Retrain / Evaluate (If You Must)

**You probably don't need to retrain** — weights are final. But if you add patients or change hyperparameters:

```bash
cd src
python cross_attentiion_train.py          # trains Cross-Attention → models/fusion_model_v1_1.pth + results/actual_validation_keys.json + BCE_Loss_Trajectory.png
python cross_attention_eval.py            # eval on 66 isolated → Confusion_Matrix_C.png, ROC_Curve_C.png, Real_Cross_Attention_Matrix.png, report txt
python -c "from eval.verify_cross_attention import run_perturbation_test; run_perturbation_test()"  # perturbation proof
# Baseline (optional)
python late_fusion_train.py
python late_fusion_eval.py
```

**Critical contract:** `cross_attentiion_train.py:72` writes `actual_validation_keys.json` with `random_split` seed 42. **Eval filters strictly to that file** (`cross_attention_eval.py:77`). If you re-run training, evaluation **will shift** (different 66). Archive old `actual_validation_keys.json` before retraining if you need reproducibility.

**Hyperparameters to touch last** (report §4.6 justified):
- `pos_weight` 4.17, `threshold` 0.83, `max_patches` 1000, `lr` 1e-4 — all ablated; change only with clinical sign-off.

---

---

## 6. Dataset Resync — Kaggle Link & Setup

**Dataset:** `tcga_coad_msi_mss_jpg` — curated TCGA-COAD H&E patches + pathology reports, MSI/MSS labeled.

**Canonical link (from `data/Download_DataSet.pdf:2`):**
```
https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg
```
*(If the link 404s, search Kaggle for `tcga_coad_msi_mss` or query the TCGA-COAD cohort directly; the PDF's truncated `joangibert/tcga_coad_msi_mss_j` resolves to the above.)*

**What you have vs what's on Kaggle:**
- **Already in repo:** `data/clean_tensors/` (653 frozen tensors) + `data/patient_registry.json` (326 patients) — this is the **processed** form after Phikon-v2/ClinicalBERT encoding. You do **not** need raw JPEGs to run the app.
- **On Kaggle:** `MSIMUT_JPEG/`, `MSS_JPEG/` (192k patches), `clinical_reports/` (9236 → 326 aligned), `metadata.csv`.

**To fully resync raw JPEGs (only needed if you want gallery JPEGs, not tensors):**
```bash
# 1. Authenticate Kaggle
pip install kaggle --break-system-packages
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Download & extract into data/
cd Colorectal_cancer_detect/data
kaggle datasets download -d joangibert/tcga_coad_msi_mss_jpg -p . --unzip
# This creates data/MSIMUT_JPEG/ and data/MSS_JPEG/ (~12 GB)

# 3. Verify via registry
python3 -c "import json; r=json.load(open('patient_registry.json')); print(len(r), r[list(r.keys())[0]].keys())"
```

**If you skip raw JPEGs:** App shows placeholder gallery; all metrics remain valid because inference uses tensors, not JPEGs.

---

## 7. Known Limitations & Gotchas (Don't File Bugs For These)

1. **Truncation at 512 tokens** — ClinicalBERT caps at 512; long reports are cut. Diagnostic text beyond 512 is unseen (report §4.3.2). That's a known debt; Tier 2 fix is Longformer.
2. **Placeholder gallery** — If you don't download raw JPEGs, Top-5 shows colored boxes. It's intentional for lightweight deploy. Add `data/MSIMUT_JPEG/` to see real patches.
3. **Ollama is optional** — Scribe works offline via template. Don't require Ollama for CI.
4. **Batch size must be 1** — Changing it breaks variable-N handling. See `multimodal_dataset.py:77`.
5. **Weights are not deterministic across retrain** — `random_split` with seed 42 + `actual_validation_keys.json` contract means retrain changes val set.
6. **Report numbers vs eval scripts** — Small drift (e.g., 89.39% vs 81.82% tables in report Ch.5) due to different threshold choices (0.83 vs 0.70) and late vs cross. Always cite the **Cross-Attention τ=0.83** line.
7. **`cross_attentiion_train.py` filename has typo** (double `i`) — keep it; many scripts import that exact name. Don't rename without grep patch.
8. **`patient_registry.json` is ~5 MB, line 2 is a mega-JSON line** — don't pretty-print it; it will OOM naive formatters.
9. **Stain normalization is offline** — Reinhard was applied during patch curation, not live. New slides need matching normalization before encoding.
10. **Not FDA-cleared** — footer says "Not for clinical use without validation". Don't present as diagnostic device.

---

## 8. Directory Contract — Don't Move These

```
data/patient_registry.json          MUST stay — all loaders hardcode it
data/clean_tensors/*.pt             MUST stay — 653 files, models expect it
data/actual_validation_keys.json    Actually at results/actual_validation_keys.json — don't move
models/fusion_model_v1_1.pth        Primary weight — app tries 3 candidates but this is source of truth
src/ui/app.py                       Entry point — Streamlit Cloud points here
results/*                           Evaluation outputs — never commit raw .pth to GitHub LFS? Current .pth are small enough (<20 MB)
docs/diagrams/*.png                 Generated — reproducible, but commit them for report
```

---

---

## 10. Contact & Provenance

- **Mentor:** Mr Nageshwara Guptha
- **Report:** `PB05_Report_8thSem.pdf` (89 pages) — source of tables and figures
- **Stack:** PyTorch 2.9, HuggingFace Transformers, Phikon-v2, ClinicalBERT, Streamlit 1.52, Ollama, OpenCV

---

*This handoff was auto-generated from codebase inspection plus report extraction. Dates & metrics are from the actual `results/` artifacts on disk. When in doubt, trust the code (`src/`) over the report prose.*
