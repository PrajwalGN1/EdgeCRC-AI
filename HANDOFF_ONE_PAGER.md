# PB05 Handoff — One-Pager for Team Lead

**Multimodal Fusion for CRC MSI Detection (ClinicalBERT + Phikon-v2)** — RUAS, June 2026

## What to tell the next team in 3 minutes

This is a **working diagnostic pre-screen**: load any TCGA patient → see MSI probability + Top-5 patches + optional cellular heatmap + LLM report. Training done (89.39% acc, 0.915 AUC). Just deploy.

## Must-Know Numbers (cite these)

- **326 patients** (50 MSS/16 MSI in 66 val), 192k patches 224×224, max 1000/patient
- **Cross-Attention beats Late Fusion**: MSI F1 0.77 vs 0.50, recall 0.75 vs 0.44
- **Loss 4.17** is equilibrium (weighted), not error. **τ=0.83** is conservative (precision 0.80).
- **Kaggle**: `https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg` — already preprocessed in `data/clean_tensors/` (653 tensors).

## Run It Now

```bash
pip install -r requirements.txt
streamlit run src/ui/app.py   # → localhost:8501
# Optional: ollama serve & ollama pull qwen2.5-coder:7b  (scribe live, else template fallback)
```

Docker: `docker build -t crc-msi . && docker run -p 8501:8501 crc-msi`

## Docs to Read First

1. `README.md` — quick start, structure, all 10 diagrams
2. `docs/ARCHITECTURE.md` — 14-section deep dive with file:line refs
3. `docs/HANDOFF.md` — playbook: limits, roadmap Tier 1-3, gotchas
4. `data/DATASET.md` — Kaggle link + download steps
5. `PB05_Report_8thSem.pdf` Ch.4–5 — metrics justification

## Diagrams (10, in `docs/diagrams/`)

01 System Arch → 02 DFD → 03 Model Comp → 04 Sequence → 05 Deploy → 06 Component → 07 Train/Eval → 08 XAI → 09 UseCase/ER → 10 State/Flowchart

Regenerate: `python docs/generate_diagrams.py`

## What Was Fixed

App paths now absolute (`Path.resolve()`), missing JPEGs show placeholders (not crash), Ollama fallback templated, device auto (cuda→mps→cpu), weights multi-candidate, `.streamlit/config.toml` + `Dockerfile` attached. `app_legacy.py` kept.

## Scope Dial (pick 1–2 max next quarter)

- **Quick wins** (1–2 wks): batch CSV inference, threshold slider
- **Target** (1–2 mo): **ClinicalBERT 512→Longformer 4096** (real debt), heatmap cache
- **Defer** (3–6 mo): joint fine-tune, WSI stitching, federated

## Do Not Move / Do Not Touch Without Review

`data/patient_registry.json`, `data/clean_tensors/`, `results/actual_validation_keys.json`, `models/fusion_model_v1_1.pth`, `src/ui/app.py`, typo `cross_attentiion_train.py`

## Gotchas

- 512-token truncation loses tail of long reports → Longformer fix.
- `batch_size=1` is intentional (variable N).
- Not FDA-cleared; footer disclaimer required.

---

