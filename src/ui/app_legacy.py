import streamlit as st
import torch
import requests
import os
import sys
import json
import matplotlib.pyplot as plt
from PIL import Image

# --- PATHING FIX ---
# This forces Python to look at the root of your project so it can find the 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------

# Import your custom modules
from src.architectures.cross_attention_fusion import SummaryToPatchCrossAttention
from src.generate_pixel_xai import generate_smooth_vit_heatmap

st.set_page_config(page_title="MSI Diagnostic Engine", layout="wide")

# --- Configuration ---
REGISTRY_PATH = "../../data/patient_registry.json"
TENSORS_DIR = "../../data/clean_tensors"
MODEL_WEIGHTS = "../../models/fusion_model_v1_1.pth"


# --- 1. Load the PyTorch Model & Registry ---
@st.cache_resource
def load_system():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = SummaryToPatchCrossAttention().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval()
    except FileNotFoundError:
        st.warning(f"⚠️ Warning: {MODEL_WEIGHTS} not found.")

    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    return model, device, registry


fusion_model, device, patient_registry = load_system()

# --- Initialize Session State ---
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False


# --- 2. The LLM Scribe Integration (Qwen) ---
def generate_clinical_report(confidence, is_msi, clinical_notes):
    diagnosis = "MSI-High (Microsatellite Instability)" if is_msi else "MSS (Microsatellite Stable)"
    prompt = f"""
    You are an expert AI pathology scribe. 
    Write a 3-paragraph diagnostic summary based STRICTLY on the following metrics:
    - Patient Clinical Notes: {clinical_notes}
    - AI Predicted Status: {diagnosis}
    - Model Confidence: {confidence:.2f}%

    Paragraph 1: State the analysis summary and confidence.
    Paragraph 2: Describe the key spatial findings indicating the status based on the AI attention.
    Paragraph 3: Provide a brief clinical recommendation based on the notes and diagnosis.
    """
    try:
        response = requests.post('http://localhost:11434/api/generate',
                                 json={'model': 'qwen2.5-coder:7b', 'prompt': prompt, 'stream': False})
        return response.json()['response']
    except Exception as e:
        return f"Error connecting to Ollama: {e}. Ensure 'ollama serve' is running."


# --- 3. The Front-End Dashboard ---
st.title("🔬 Multimodal Colorectal Cancer Diagnostic Framework")
st.markdown("**Architecture:** End-to-End Cross-Attention Fusion & Automated XAI Pipeline")

# Input Section
st.subheader("1. Clinical Inputs")
col_input1, col_input2 = st.columns([1, 2])

with col_input1:
    patient_id = st.text_input("Enter Patient ID (e.g., TCGA-AY-6386)", value="TCGA-AY-6386")
    # Dynamic slider for patch display
    num_patches = st.slider("Number of Top Patches to Display", min_value=5, max_value=20, value=5, step=1)

with col_input2:
    # Auto-fetch report text if patient exists, otherwise use default
    default_text = "Patient presents with ascending colon adenocarcinoma. Awaiting biomarker analysis."
    if patient_id in patient_registry:
        default_text = patient_registry[patient_id].get("report_text", default_text)

    # Text area is populated with fetched text, but remains fully editable by the user
    clinical_text = st.text_area("Patient Clinical Report (Editable)", value=default_text, height=100)

analyze_button = st.button("Run Diagnostic Pipeline", type="primary")

# --- Execution Logic ---
if analyze_button and patient_id:
    if patient_id not in patient_registry:
        st.error(f"❌ Patient {patient_id} not found in the registry.")
        st.session_state.analysis_complete = False
    else:
        with st.spinner(f"Fetching data and executing Cross-Attention for {patient_id}..."):

            patient_data = patient_registry[patient_id]
            image_paths = patient_data["image_paths"]
            true_label_int = patient_data["label"]
            if isinstance(true_label_int, str):
                true_label_int = 1 if "MSI" in true_label_int.upper() else 0
            true_label = "MSI" if true_label_int == 1 else "MSS"

            vision_tensor_path = os.path.join(TENSORS_DIR, f"{patient_id}_vision.pt")
            text_tensor_path = os.path.join(TENSORS_DIR, f"{patient_id}_text.pt")

            try:
                vision_tensor = torch.load(vision_tensor_path, weights_only=True).unsqueeze(0).to(device)
                text_tensor = torch.load(text_tensor_path, weights_only=True).unsqueeze(0).to(device)

                with torch.no_grad():
                    prob, attention_weights = fusion_model(vision_tensor, text_tensor)

                # Save results to session state so they survive Streamlit reruns
                st.session_state.prediction_score = prob.item()
                st.session_state.is_msi_pred = st.session_state.prediction_score >= 0.83
                st.session_state.predicted_class = "MSI" if st.session_state.is_msi_pred else "MSS"
                st.session_state.confidence = st.session_state.prediction_score * 100 if st.session_state.is_msi_pred else (
                                                                                                                                       1 - st.session_state.prediction_score) * 100
                st.session_state.true_label = true_label
                st.session_state.attention_scores = attention_weights.squeeze().cpu().numpy()
                st.session_state.image_paths = image_paths
                st.session_state.clinical_text = clinical_text  # Save the customized text

                # Fetching LLM Report now so it doesn't re-trigger on dropdown change
                st.session_state.scribe_report = generate_clinical_report(st.session_state.confidence,
                                                                          st.session_state.is_msi_pred, clinical_text)

                st.session_state.analysis_complete = True

            except FileNotFoundError:
                st.error("❌ Tensors not found. Run dataset generation first.")
                st.session_state.analysis_complete = False

# --- Display Results (Reads from Session State) ---
if st.session_state.get('analysis_complete'):
    st.markdown("---")
    st.subheader("2. Macroscopic XAI: Cross-Attention Spatial Filtering")
    st.markdown(
        f"**True Status:** {st.session_state.true_label} | **Predicted:** {st.session_state.predicted_class} ({st.session_state.confidence:.2f}%)")

    # Get Top N indices dynamically based on the slider
    top_n_indices = st.session_state.attention_scores.argsort()[-num_patches:][::-1]

    # We will store valid local paths here to populate the dropdown later
    valid_patch_paths = []
    dropdown_options = []

    # Display images in rows of 5
    cols = st.columns(5)
    for i, idx in enumerate(top_n_indices):
        col = cols[i % 5]  # Wrap around to next row
        filename = os.path.basename(st.session_state.image_paths[idx])
        local_folder = "../../data/MSIMUT_JPEG" if st.session_state.true_label == "MSI" else "../../data/MSS_JPEG"
        local_img_path = os.path.join(local_folder, filename)

        valid_patch_paths.append(local_img_path)
        dropdown_options.append(f"Rank {i + 1} (Score: {st.session_state.attention_scores[idx]:.4f})")

        try:
            img = Image.open(local_img_path)
            col.image(img, caption=f"Rank {i + 1}\n{filename[:15]}...", use_container_width=True)
        except:
            col.error("Image Missing")

    st.markdown("---")
    st.subheader("3. Microscopic Interpretability & Diagnostic Scribe")

    res_col1, res_col2 = st.columns([1, 1.5])

    # Left Column: Dropdown Heatmap Selector
    with res_col1:
        st.markdown("**Targeted Cellular Analysis**")
        selected_dropdown = st.selectbox("Select a patch from the Top N to generate a cellular heatmap:",
                                         dropdown_options)

        # Find which index they selected
        selected_index = dropdown_options.index(selected_dropdown)
        selected_img_path = valid_patch_paths[selected_index]

        if st.button("Generate Microscopic Heatmap"):
            with st.spinner("Generating Pixel-Level ViT Heatmap..."):
                if os.path.exists(selected_img_path):
                    generate_smooth_vit_heatmap(selected_img_path)

                    filename_only = os.path.basename(selected_img_path)
                    heatmap_file = f"Interactive_XAI_{os.path.basename(selected_img_path)}.png"

                    if os.path.exists(heatmap_file):
                        st.image(heatmap_file, caption=f"Cellular Heatmap: {selected_dropdown}",
                                 use_container_width=True)
                        os.remove(heatmap_file)
                else:
                    st.error("Image not found on disk.")

    # Right Column: The LLM Report
    with res_col2:
        st.markdown("**Automated Qwen Pathology Scribe**")
        st.info(st.session_state.scribe_report)

# --- 4. Manual XAI Investigation (Optional) ---
st.markdown("---")
st.subheader("🔍 4. Manual Microscopic Analysis")
st.markdown("Upload any `.jpg` outside of the Top N from the patient's dataset to generate a custom Phikon-v2 heatmap.")

manual_patch = st.file_uploader("Upload Specific Patient Patch for Heatmap", type=["jpg", "jpeg"])

if manual_patch:
    with st.spinner("Generating custom pixel-level heatmap..."):
        temp_path = manual_patch.name
        with open(temp_path, "wb") as f:
            f.write(manual_patch.getbuffer())

        generate_smooth_vit_heatmap(temp_path)

        heatmap_file = f"Interactive_XAI_{os.path.basename(temp_path)}.png"
        if os.path.exists(heatmap_file):
            st.image(heatmap_file, caption=f"Custom Deep Heatmap: {temp_path}", width=600)
            os.remove(heatmap_file)

        if os.path.exists(temp_path):
            os.remove(temp_path)