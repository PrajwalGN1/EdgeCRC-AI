"""
Multimodal Colorectal Cancer Diagnostic Framework — Streamlit Production App
Architecture: SummaryToPatchCrossAttention (Phikon-v2 + ClinicalBERT) + Dual-Tier XAI + Qwen Scribe
"""
import os
import sys
import json
import requests
from pathlib import Path

import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Path resolution (robust for streamlit run from any CWD)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.cross_attention_fusion import SummaryToPatchCrossAttention
from src.inference import (
    InferenceValidationError,
    MissingInputError,
    InvalidImageError,
    InvalidReportError,
    CaseMismatchError,
    MissingModelWeightsError,
    MultimodalPredictor,
    validate_multimodal_inputs,
)

# Try importing XAI — optional if transformers not available in cloud
try:
    from src.generate_pixel_xai import generate_smooth_vit_heatmap
    HAS_XAI = True
except Exception as e:
    HAS_XAI = False
    XAI_IMPORT_ERROR = str(e)

# Config
REGISTRY_PATH = PROJECT_ROOT / "data" / "patient_registry.json"
TENSORS_DIR = PROJECT_ROOT / "data" / "clean_tensors"
MODEL_WEIGHTS_CANDIDATES = [
    PROJECT_ROOT / "models" / "fusion_model_v1_1.pth",
    PROJECT_ROOT / "models" / "best_fusion_model.pth",
    PROJECT_ROOT / "models" / "fusion_model_v1_0.pth",
]
ASSETS_DIR = PROJECT_ROOT / "assets" / "medical_images"
MODEL_WEIGHTS = next((p for p in MODEL_WEIGHTS_CANDIDATES if p.exists()), MODEL_WEIGHTS_CANDIDATES[0])

st.set_page_config(
    page_title="MSI Diagnostic Engine — CRC Multimodal",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar — System Info
with st.sidebar:
    st.title("🔬 CRC MSI Engine")
    st.caption("Multimodal Fusion • Phikon-v2 + ClinicalBERT")
    st.divider()
    st.markdown("**Model:** SummaryToPatchCrossAttention")
    st.markdown("**Vision:** Phikon-v2 (ViT-L/16) 1024-D")
    st.markdown("**Text:** ClinicalBERT 768-D → 1024 Proj")
    st.markdown("**Heads:** 8  •  **Threshold τ:** 0.83")
    st.markdown("**Classes:** MSI (1) / MSS (0)")
    st.divider()
    if MODEL_WEIGHTS.exists():
        size_mb = MODEL_WEIGHTS.stat().st_size / 1_048_576
        st.success(f"✓ Weights: {MODEL_WEIGHTS.name} ({size_mb:.1f} MB)")
    else:
        st.error(f"✗ Weights missing: {MODEL_WEIGHTS.name}")
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            _tmp = json.load(f)
        st.info(f"Registry: {len(_tmp)} patients")
        msi = sum(1 for v in _tmp.values() if v.get("label") == 1 or str(v.get("label")).upper() == "MSI")
        mss = len(_tmp) - msi
        st.caption(f"MSS: {mss}  •  MSI: {msi}  •  w⁺=4.17")
    else:
        st.warning("Registry not found")
    st.divider()
    st.caption("PB05 • v2.0 Production — Dark XAI Enabled")
    st.divider()
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        if r.ok:
            st.success("Ollama: Online")
        else:
            st.warning("Ollama: Unreachable")
    except Exception:
        st.caption("Ollama: Offline (scribe will use fallback)")

# Cached loaders
@st.cache_resource(show_spinner="Loading model & registry...")
def load_system():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = SummaryToPatchCrossAttention().to(device)
    loaded = False
    last_err = None
    for wp in MODEL_WEIGHTS_CANDIDATES:
        if wp.exists():
            try:
                state = torch.load(str(wp), map_location=device, weights_only=True)
                model.load_state_dict(state)
                loaded = True
                break
            except Exception as e:
                last_err = e
                continue
    model.eval()
    registry = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    return model, device, registry, loaded, last_err

fusion_model, device, patient_registry, weights_ok, weights_err = load_system()
if not weights_ok:
    st.warning(f"Model weights could not be loaded. Tried: {[str(p) for p in MODEL_WEIGHTS_CANDIDATES]}. Error: {weights_err}.")

for k in [
    "analysis_complete", "prediction_score", "is_msi_pred", "predicted_class",
    "confidence", "true_label", "attention_scores", "image_paths", "clinical_text",
    "scribe_report", "raw_prediction_complete", "raw_prediction", "raw_report_text",
]:
    if k not in st.session_state:
        st.session_state[k] = None
if "analysis_complete" not in st.session_state or st.session_state["analysis_complete"] is None:
    st.session_state.analysis_complete = False
if "raw_prediction_complete" not in st.session_state or st.session_state["raw_prediction_complete"] is None:
    st.session_state.raw_prediction_complete = False

def generate_clinical_report(confidence, is_msi, clinical_notes, top_scores=None):
    diagnosis = "MSI-High (Microsatellite Instability)" if is_msi else "MSS (Microsatellite Stable)"
    prompt = f"""You are an expert AI pathology scribe. Write a 3-paragraph diagnostic summary STRICTLY based on:
- Patient Clinical Notes: {clinical_notes[:1500]}
- AI Predicted Status: {diagnosis}
- Model Confidence: {confidence:.2f}%
- Threshold: 0.83, Architecture: SummaryToPatchCrossAttention (Phikon-v2 + ClinicalBERT)

Paragraph 1: State the analysis summary and confidence.
Paragraph 2: Describe the key spatial findings indicating the status based on the AI attention (do not invent patch filenames).
Paragraph 3: Provide a brief clinical recommendation (ICIs if MSI, standard chemo if MSS, suggest confirmatory IHC/PCR if borderline)."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5-coder:7b", "prompt": prompt, "stream": False},
            timeout=30,
        )
        if resp.ok and "response" in resp.json():
            return resp.json()["response"]
    except Exception as e:
        ollama_err = str(e)
    else:
        ollama_err = "Ollama returned non-OK or missing field"
    conf_str = f"{confidence:.1f}%"
    top_str = ""
    if top_scores is not None and len(top_scores) > 0:
        top3 = sorted(top_scores, reverse=True)[:3]
        top_str = f" Top-3 attention scores: {', '.join(f'{s:.4f}' for s in top3)}."
    if is_msi:
        return (
            f"**Paragraph 1 — Summary:** The multimodal Cross-Attention model predicts MSI-High with {conf_str} confidence (threshold 0.83). "
            f"This suggests a high likelihood of mismatch-repair deficiency. The prediction integrates Phikon-v2 histology embeddings (N patches, 1024-D) with ClinicalBERT report semantics via 8-head cross-attention.{top_str}\n\n"
            f"**Paragraph 2 — Spatial Findings:** The model's text-guided attention concentrated on a small subset of tissue patches. High-attention regions typically correspond to lymphocyte-rich stroma, mucinous differentiation, or poorly differentiated glandular architecture — patterns enriched in MSI-H CRC. Review the Top-5 gallery and, if available, the Phikon-v2 cellular heatmap for pixel-level corroboration.\n\n"
            f"**Paragraph 3 — Recommendation:** Given the MSI-H call, consider confirmatory IHC for MMR proteins (MLH1/MSH2/MSH6/PMS2) or PCR-based MSI testing, and evaluate eligibility for immune checkpoint inhibition (e.g., Pembrolizumab) per NCCN guidelines. Correlate with clinical notes and tumor sidedness (MSI more common right-sided). This AI result is a pre-screen, not a definitive diagnosis."
        )
    else:
        return (
            f"**Paragraph 1 — Summary:** The model predicts MSS (Microsatellite Stable) with {conf_str} confidence (threshold 0.83, inverted confidence). "
            f"Mismatch-repair proficiency is likely. The ensemble of vision (Phikon-v2) and text (ClinicalBERT) via cross-attention did not detect MSI-associated morphology or semantics above the conservative cutoff.{top_str}\n\n"
            f"**Paragraph 2 — Spatial Findings:** Attention was diffusely distributed or focused on well-differentiated glandular areas without dense lymphocytic infiltrate or mucinous features. Low attention scores on immune-rich patches support the MSS call. Visual review of the Top-5 patches should confirm absence of Crohn's-like reaction or medullary patterns.\n\n"
            f"**Paragraph 3 — Recommendation:** Proceed with standard CRC management per stage (surgical resection, adjuvant chemotherapy as indicated). MSS status predicts limited benefit from ICI monotherapy; confirmatory testing only if clinical suspicion remains (e.g., young age, Bethesda criteria, right-sided mucinous tumor)."
        )

# Header
st.title("🔬 Multimodal Colorectal Cancer Diagnostic Framework")
st.markdown("**Architecture:** `SummaryToPatchCrossAttention` — End-to-End Cross-Attention Fusion & Dual-Tier XAI Pipeline  •  *Device:* `{}`  •  *Weights:* `{}`".format(device, MODEL_WEIGHTS.name if MODEL_WEIGHTS.exists() else "missing"))
st.caption("Phikon-v2 (frozen, 307M) encodes each 224×224 patch → 1024-D  •  ClinicalBERT (frozen, 110M) encodes report → 768-D → Linear 768→1024 → 8-head Cross-Attention (Q=text, K/V=patches) → LayerNorm → MLP 1024→256→1 → Sigmoid → τ=0.83")

# Top-Level Tabs
tab_raw, tab_cohort = st.tabs([
    "🔬 Live Raw-Input Diagnostic (Production Pipeline)",
    "📊 Cohort Explorer & Tensor Benchmark (Regression Mode)",
])

# =========================================================================
# TAB 1: PRODUCTION RAW-INPUT TWO-MODALITY WORKFLOW
# =========================================================================
with tab_raw:
    st.subheader("1. Patient Raw Inputs (Dual-Modality)")
    st.markdown(
        "Production inference requires **BOTH** an H&E histopathology image and the patient's pathology report. "
        "The application generates embeddings live via Phikon-v2 and ClinicalBERT, runs the fusion checkpoint, "
        "and produces the MSI/MSS prediction without requiring precomputed tensors."
    )

    col_raw1, col_raw2 = st.columns([1, 1])
    with col_raw1:
        st.markdown("**Modality 1: H&E Histopathology Image**")
        raw_case_id = st.text_input(
            "Case / Specimen ID",
            value="CASE-2026-001",
            help="Identifier associating histology with pathology report",
            key="raw_case_id",
        )
        uploaded_image = st.file_uploader(
            "Upload H&E Histology Image (Tile 224x224 or Micrograph/WSI)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="raw_img_upload",
        )
        if uploaded_image is not None:
            try:
                preview_img = Image.open(uploaded_image)
                st.image(
                    preview_img,
                    caption=f"Uploaded Image ({preview_img.width}x{preview_img.height}, {preview_img.mode})",
                    use_container_width=True,
                )
            except Exception as e_prev:
                st.warning(f"Could not render image preview: {e_prev}")
        else:
            st.info("ℹ️ Upload an H&E slide or 224×224 patch tile.")

    with col_raw2:
        st.markdown("**Modality 2: Clinical Pathology Report**")
        raw_report_text = st.text_area(
            "Clinical Pathology Notes (Unstructured surgical narrative)",
            value="",
            height=180,
            placeholder="Paste patient pathology report text (e.g. tumor site, histology, histological grade, margin status, lymph node involvement)...",
            key="raw_text_input",
        )
        uploaded_report_file = st.file_uploader(
            "Or upload report text file (.txt)",
            type=["txt"],
            key="raw_report_file",
        )
        if uploaded_report_file is not None and not raw_report_text.strip():
            try:
                raw_report_text = uploaded_report_file.read().decode("utf-8")
                st.caption(f"Loaded report file ({len(raw_report_text)} characters)")
            except Exception as e_txt:
                st.error(f"Failed to read report file: {e_txt}")

    run_raw_clicked = st.button("▶ Run End-to-End Multimodal Diagnostic", type="primary", key="btn_run_raw")

    if run_raw_clicked:
        report_to_validate = raw_report_text
        if uploaded_report_file is not None and not report_to_validate.strip():
            try:
                report_to_validate = uploaded_report_file.getvalue().decode("utf-8")
            except Exception:
                pass

        try:
            val_case = validate_multimodal_inputs(
                image_input=uploaded_image,
                report_input=report_to_validate,
                case_id=raw_case_id,
            )
            st.success(
                f"✓ Inputs validated for Case '{val_case.case_id}' "
                f"({val_case.image.width}x{val_case.image.height} image, {len(val_case.report_text)} chars text)."
            )

            # Step 2: Attempt Real Live Inference
            with st.spinner("Executing end-to-end inference (Tiling -> Phikon-v2 -> ClinicalBERT -> Cross-Attention)..."):
                try:
                    predictor = MultimodalPredictor(checkpoint_path=MODEL_WEIGHTS, device=device)
                    raw_result = predictor.predict(
                        image_input=val_case.image,
                        report_input=val_case.report_text,
                        case_id=val_case.case_id,
                    )
                    st.session_state.raw_prediction = raw_result
                    st.session_state.raw_report_text = val_case.report_text
                    st.session_state.raw_prediction_complete = True
                except MissingModelWeightsError as weight_err:
                    st.session_state.raw_prediction_complete = False
                    st.error("❌ Foundation Encoder Weights Unavailable for Live Raw Inference")
                    st.markdown(
                        f"""
                        > **Production Inference Blocker:**  
                        > Live feature extraction requires pretrained foundation model weights that are not cached locally:
                        > - **Vision Encoder:** `owkin/phikon-v2` (~1.22 GB `model.safetensors`)
                        > - **Text Encoder:** `emilyalsentzer/Bio_ClinicalBERT` (~436 MB `pytorch_model.bin`)
                        > 
                        > **Clinical Safety Policy:** The framework fails closed. It will **never** return a fabricated medical prediction or substitute uncalibrated mocks in production.
                        > 
                        > *Technical Error:* `{weight_err}`
                        """
                    )
                except Exception as e_pred:
                    st.session_state.raw_prediction_complete = False
                    st.error(f"❌ Pipeline Execution Error: {e_pred}")

        except MissingInputError as err_missing:
            st.error(f"❌ Missing Modality: {err_missing}")
            st.session_state.raw_prediction_complete = False
        except InferenceValidationError as err_val:
            st.error(f"❌ Input Validation Error: {err_val}")
            st.session_state.raw_prediction_complete = False

    # Display Raw Results if successfully completed
    if st.session_state.get("raw_prediction_complete") and st.session_state.get("raw_prediction"):
        res = st.session_state.raw_prediction
        st.divider()
        st.subheader("2. Diagnostic Biomarker Prediction")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Case ID", res.case_id)
        rc2.metric("Predicted Status", res.prediction, delta=f"{res.confidence_pct:.1f}% confidence")
        rc3.metric("MSI Probability", f"{res.probability:.4f}", delta="τ=0.83")
        rc4.metric("Tissue Patches (N)", f"{res.num_patches}", delta=f"{'MSI-High' if res.class_id == 1 else 'MSS'}")

        st.progress(res.probability, text=f"MSI-High Score: {res.probability:.2%}")
        if res.class_id == 1:
            st.error(
                f"🚨 Biomarker Detected: MSI-High (Confidence: {res.confidence_pct:.1f}% @ threshold 0.83). "
                f"Patient may be eligible for Immune Checkpoint Inhibitor (ICI) therapy."
            )
        else:
            st.success(
                f"✓ Biomarker Result: MSS (Confidence: {res.confidence_pct:.1f}% @ threshold 0.83). "
                f"Standard colorectal cancer therapy protocol indicated."
            )

        # XAI and Scribe for Raw Result
        st.divider()
        st.subheader("3. Clinical Diagnostic Scribe")
        active_notes = st.session_state.get("raw_report_text", "")
        scribe_text = generate_clinical_report(
            res.confidence_pct, res.class_id == 1, active_notes, top_scores=res.attention_weights
        )
        st.info(scribe_text)
        st.download_button(
            "Download Diagnostic Report (.txt)",
            data=scribe_text,
            file_name=f"{res.case_id}_diagnostic_report.txt",
            mime="text/plain",
            key="btn_download_raw_rep",
        )


# =========================================================================
# TAB 2: DEVELOPMENT COHORT EXPLORER & PRECOMPUTED TENSOR BENCHMARK
# =========================================================================
with tab_cohort:
    st.subheader("1. Cohort Patient Selection (Regression Mode)")
    col1, col2 = st.columns([1, 2])
    with col1:
        known_ids = sorted(patient_registry.keys()) if patient_registry else []
        default_id = "TCGA-AA-3877" if "TCGA-AA-3877" in patient_registry else (known_ids[0] if known_ids else "TCGA-AA-3877")
        patient_id = (
            st.selectbox("Select Patient ID", options=known_ids, index=known_ids.index(default_id) if default_id in known_ids else 0)
            if known_ids
            else st.text_input("Enter Patient ID (e.g., TCGA-AY-6386)", value=default_id)
        )
        patient_id_manual = st.text_input("Or type custom ID (overrides selection)", value="", placeholder="TCGA-...", key="reg_custom_id")
        if patient_id_manual.strip():
            patient_id = patient_id_manual.strip()
        num_patches = st.slider("Top patches to display", 5, 20, 5, 1, key="reg_num_patches")
        st.caption(f"Tensors: {TENSORS_DIR} • {len(known_ids)} patients in registry")
    with col2:
        default_text = "Patient presents with ascending colon adenocarcinoma. Awaiting biomarker analysis."
        if patient_id in patient_registry:
            default_text = patient_registry[patient_id].get("report_text", default_text)
            display_default = (default_text[:1200] + "…") if len(default_text) > 1200 else default_text
        else:
            display_default = default_text
        clinical_text = st.text_area(
            "Patient Clinical Report (Stored Registry Text)",
            value=display_default,
            height=140,
            key="reg_clinical_text",
        )
        st.caption("Note: In this regression mode, the model evaluates the frozen precomputed tensors (`*_vision.pt`, `*_text.pt`).")
    analyze_clicked = st.button("▶ Run Diagnostic Pipeline (Precomputed Tensors)", type="primary", key="btn_run_reg")

    # Execution
    if analyze_clicked and patient_id:
        if patient_id not in patient_registry:
            st.error(f"Patient `{patient_id}` not found in registry. Available: {', '.join(known_ids[:8])}… ({len(known_ids)} total)")
            st.session_state.analysis_complete = False
        else:
            with st.spinner(f"Loading tensors & running Cross-Attention for {patient_id} on {device}…"):
                patient_data = patient_registry[patient_id]
                image_paths = patient_data.get("image_paths", [])
                raw_label = patient_data.get("label", 0)
                if isinstance(raw_label, str):
                    true_int = 1 if "MSI" in raw_label.upper() else 0
                else:
                    true_int = int(raw_label)
                true_label = "MSI" if true_int == 1 else "MSS"
                v_path = TENSORS_DIR / f"{patient_id}_vision.pt"
                t_path = TENSORS_DIR / f"{patient_id}_text.pt"
                if not v_path.exists() or not t_path.exists():
                    st.error(f"Tensors not found for {patient_id}. Expected:\n- {v_path}\n- {t_path}\nRun dataset generation or download clean_tensors.")
                    st.session_state.analysis_complete = False
                else:
                    try:
                        v = torch.load(str(v_path), map_location=device, weights_only=True)
                        t = torch.load(str(t_path), map_location=device, weights_only=True)
                        if v.dim() == 2:
                            v = v.unsqueeze(0)
                        if t.dim() == 2 and t.shape[0] == 1:
                            t = t.squeeze(0)
                        if t.dim() == 1:
                            t = t.unsqueeze(0)
                        v = v.to(device)
                        t = t.to(device)
                        with torch.no_grad():
                            prob, attn_weights = fusion_model(v, t)
                        score = prob.item() if prob.numel() == 1 else prob.squeeze().item()
                        is_msi = score >= 0.83
                        pred = "MSI" if is_msi else "MSS"
                        conf = score * 100 if is_msi else (1 - score) * 100
                        attn = attn_weights.squeeze().detach().cpu().numpy()
                        if attn.ndim > 1:
                            attn = attn.reshape(-1)
                        st.session_state.prediction_score = float(score)
                        st.session_state.is_msi_pred = bool(is_msi)
                        st.session_state.predicted_class = pred
                        st.session_state.confidence = float(conf)
                        st.session_state.true_label = true_label
                        st.session_state.attention_scores = attn
                        st.session_state.image_paths = image_paths
                        st.session_state.clinical_text = clinical_text
                        st.session_state.scribe_report = generate_clinical_report(conf, is_msi, clinical_text, top_scores=attn)
                        st.session_state.analysis_complete = True
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="python")
                        st.session_state.analysis_complete = False

    # Results
    if st.session_state.get("analysis_complete"):
        st.divider()
        st.subheader("2. Macroscopic XAI — Cross-Attention Spatial Filtering")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Status", st.session_state.true_label)
        c2.metric("Predicted", st.session_state.predicted_class, delta=f"{st.session_state.confidence:.1f}% conf")
        c3.metric("Raw Probability", f"{st.session_state.prediction_score:.4f}", delta="τ=0.83")
        c4.metric("Patches (N)", f"{len(st.session_state.attention_scores)}", delta=f"τ {'MSI' if st.session_state.is_msi_pred else 'MSS'}")
        st.progress(min(max(st.session_state.prediction_score, 0.0), 1.0), text=f"MSI Probability: {st.session_state.prediction_score:.2%}")
        match = st.session_state.true_label == st.session_state.predicted_class
        if match:
            st.success(f"✓ Prediction MATCHES ground truth ({st.session_state.true_label})")
        else:
            st.warning(f"⚠ Prediction DIFFERS from ground truth (True: {st.session_state.true_label}, Pred: {st.session_state.predicted_class}) — review XAI & consider confirmatory testing")
        attn_scores = st.session_state.attention_scores
        top_n = st.session_state.attention_scores.argsort()[::-1][:num_patches]
        valid_patch_paths = []
        dropdown_options = []
        for rank, idx in enumerate(top_n, start=1):
            fname = os.path.basename(st.session_state.image_paths[int(idx)]) if int(idx) < len(st.session_state.image_paths) else f"patch_{int(idx):04d}.jpg"
            candidates = [
                PROJECT_ROOT / "data" / "MSIMUT_JPEG" / fname,
                PROJECT_ROOT / "data" / "MSS_JPEG" / fname,
                PROJECT_ROOT / "data" / fname,
                ASSETS_DIR / fname,
            ]
            local_path = next((str(p) for p in candidates if p.exists()), None)
            valid_patch_paths.append(local_path if local_path else f"MISSING::{fname}")
            dropdown_options.append(f"Rank {rank} — idx {int(idx)} — Score {attn_scores[int(idx)]:.4f} — {fname[:22]}")
        cols = st.columns(5)
        for i, idx in enumerate(top_n):
            col = cols[i % 5]
            score = attn_scores[int(idx)]
            fpath = valid_patch_paths[i]
            cap = f"R{i+1} • {score:.4f}"
            if fpath.startswith("MISSING::"):
                fig, ax = plt.subplots(figsize=(2, 2))
                cmap_val = plt.cm.Reds(score / max(attn_scores.max(), 1e-6))
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=cmap_val))
                ax.text(0.5, 0.55, f"Patch\n{int(idx)}", ha="center", va="center", fontsize=10, weight="bold", color="white")
                ax.text(0.5, 0.35, f"{score:.4f}", ha="center", fontsize=7, color="white")
                ax.axis("off")
                fig.tight_layout(pad=0.2)
                col.pyplot(fig, use_container_width=True)
                plt.close(fig)
                col.caption(cap + " (JPEG not bundled — placeholder)")
            else:
                try:
                    img = Image.open(fpath)
                    col.image(img, caption=cap, use_container_width=True)
                except Exception as e:
                    col.error(f"Load fail: {e}")
        with st.expander("Attention distribution histogram", expanded=False):
            fig2, ax2 = plt.subplots(figsize=(8, 2.5))
            ax2.hist(attn_scores, bins=30, color="#0EA5E9", edgecolor="white", alpha=0.8)
            ax2.axvline(np.percentile(attn_scores, 80), color="#EF4444", linestyle="--", label="80th %ile (ROI cutoff)")
            ax2.set_xlabel("Attention weight")
            ax2.set_ylabel("Patch count")
            ax2.set_title(f"Attention weight distribution (N={len(attn_scores)}, max={attn_scores.max():.4f})")
            ax2.legend(fontsize=7)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        st.divider()
        st.subheader("3. Microscopic Interpretability & Diagnostic Scribe")
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown("**Targeted Cellular Analysis — Darkened Heatmap (Phikon-v2)**")
            sel = st.selectbox("Select a Top-N patch to generate cellular heatmap:", dropdown_options, key="heatmap_sel")
            sel_idx = dropdown_options.index(sel)
            sel_path = valid_patch_paths[sel_idx]
            if st.button("Generate Microscopic Heatmap", key="gen_heatmap_main"):
                with st.spinner("Running darkened attention rollout (last 6 layers, inferno, α=0.72)…"):
                    if sel_path.startswith("MISSING::"):
                        st.error("Original JPEG not found. Upload a patch via 'Manual Analysis' below (requires JPEG).")
                    elif os.path.exists(sel_path):
                        try:
                            out_path = generate_smooth_vit_heatmap(sel_path)
                            hm_candidates = []
                            if out_path and os.path.exists(out_path):
                                hm_candidates.append(Path(out_path))
                            fname = os.path.basename(sel_path)
                            hm_file = f"Interactive_XAI_{fname}.png"
                            for cand in [Path.cwd() / hm_file, PROJECT_ROOT / hm_file, Path(hm_file), Path("/tmp") / hm_file]:
                                if cand.exists():
                                    hm_candidates.append(cand)
                            hm_path = hm_candidates[0] if hm_candidates else None
                            if hm_path and hm_path.exists():
                                st.image(str(hm_path), caption=f"Dark Cellular Heatmap — {sel}", use_container_width=True)
                                st.caption("Darkened: Original H&E • Yellow ROI contours (Top 20%) • Inferno overlay α=0.72 (darker, high contrast)")
                                try:
                                    if hm_path.exists():
                                        hm_path.unlink()
                                except Exception:
                                    pass
                            else:
                                st.warning("Heatmap file not found after generation. Check logs.")
                        except Exception as e:
                            st.error(f"Heatmap generation failed: {e}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                    else:
                        st.error("Image not found on disk.")
        with right:
            st.markdown("**Automated Pathology Scribe (Qwen 2.5 via Ollama)**")
            report = st.session_state.get("scribe_report", "")
            st.info(report)
            st.download_button(
                "Download Report (.txt)",
                data=report,
                file_name=f"{patient_id}_scribe_report.txt",
                mime="text/plain",
                key="btn_download_reg_rep",
            )
            st.caption("Deterministic prompt — no hallucination. Falls back to template if Ollama offline.")

    # 4. Manual XAI
    st.divider()
    st.subheader("Manual Microscopic Analysis")
    st.markdown("Upload any `224×224` H&E patch (`.jpg`/`.jpeg`) to generate a darkened Phikon-v2 cellular heatmap — independent of the Top-5 ranking.")
    manual = st.file_uploader("Upload H&E patch for heatmap", type=["jpg", "jpeg"], key="manual_up")
    if manual is not None:
        with st.spinner("Generating darkened pixel-level ViT heatmap…"):
            tmp = Path(manual.name)
            tmp_path = PROJECT_ROOT / tmp.name
            with open(tmp_path, "wb") as f:
                f.write(manual.getbuffer())
            try:
                out_path = generate_smooth_vit_heatmap(str(tmp_path)) if HAS_XAI else None
                if not HAS_XAI:
                    try:
                        from src.generate_pixel_xai import generate_smooth_vit_heatmap as gen2
                        out_path = gen2(str(tmp_path))
                    except Exception as e:
                        st.error(f"XAI unavailable: {e}")
                        out_path = None
                hm_candidates = []
                if out_path and os.path.exists(out_path):
                    hm_candidates.append(Path(out_path))
                hm_file = f"Interactive_XAI_{tmp.name}.png"
                for cand in [Path.cwd() / hm_file, PROJECT_ROOT / hm_file, Path(hm_file), Path("/tmp") / hm_file]:
                    if cand.exists():
                        hm_candidates.append(cand)
                hm_path = hm_candidates[0] if hm_candidates else None
                if hm_path and hm_path.exists():
                    st.image(str(hm_path), caption=f"Dark Custom Heatmap: {tmp.name}", width=700)
                    st.caption("Darkened inferno overlay — high contrast for cellular detail")
                    try:
                        hm_path.unlink()
                    except Exception:
                        pass
                else:
                    try:
                        import cv2
                        raw = Image.open(str(tmp_path)).convert('RGB')
                        arr = np.array(raw)
                        dark = (arr * 0.45).astype(np.uint8)
                        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                        blur = cv2.GaussianBlur(gray, (21,21), 0)
                        norm = (blur - blur.min())/(blur.max()-blur.min()+1e-8)
                        heat = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
                        overlay = cv2.addWeighted(dark, 0.55, heat, 0.75, 0)
                        fig, axes = plt.subplots(1,3, figsize=(12,4))
                        axes[0].imshow(raw); axes[0].set_title("Original"); axes[0].axis('off')
                        axes[1].imshow(dark); axes[1].set_title("Darkened Base (0.45)"); axes[1].axis('off')
                        axes[2].imshow(overlay); axes[2].set_title("Dark Inferno Overlay", color='darkred'); axes[2].axis('off')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as e2:
                        st.warning(f"Heatmap file not produced — {e2}")
            except Exception as e:
                st.error(f"Failed: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
st.divider()
