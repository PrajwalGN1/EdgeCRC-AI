import os
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image

# --- THE CRITICAL FIX: USING YOUR NEW ARCHITECTURE ---
from architectures.cross_attention_fusion import SummaryToPatchCrossAttention

# --- Configuration ---
REGISTRY_PATH = "../data/patient_registry.json"
TENSORS_DIR = "../data/clean_tensors"
MODEL_WEIGHTS = "../models/fusion_model_v1_1.pth"


def generate_xai_report(patient_id):
    print(f"🚀 Initiating XAI Engine for Patient: {patient_id}")

    # 1. Load the Data Contract
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    if patient_id not in registry:
        print(f"❌ Error: Patient {patient_id} not found in registry.")
        return

    patient_data = registry[patient_id]
    image_paths = patient_data["image_paths"]

    # Apply the same Alphabetical Label Fix we used in the dataset
    raw_label = patient_data["label"]
    if isinstance(raw_label, str):
        true_label_int = 1 if "MSI" in raw_label.upper() else 0
    else:
        true_label_int = raw_label

    true_label = "MSI" if true_label_int == 1 else "MSS"

    # 2. Load Tensors
    vision_tensor_path = os.path.join(TENSORS_DIR, f"{patient_id}_vision.pt")
    text_tensor_path = os.path.join(TENSORS_DIR, f"{patient_id}_text.pt")

    try:
        # Load tensors and ensure they have the artificial batch dimension [1, N, 1024]
        vision_tensor = torch.load(vision_tensor_path, weights_only=True).unsqueeze(0)
        # Ensure text has the batch dimension [1, 1, 768]
        text_tensor = torch.load(text_tensor_path, weights_only=True).unsqueeze(0)
    except FileNotFoundError:
        print(f"❌ Error: Tensors for {patient_id} not found in {TENSORS_DIR}.")
        return

    # 3. Hardware Check (Optimized for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Executing on device: {device}")

    # --- INITIALIZE THE NEW CROSS-ATTENTION MODEL ---
    model = SummaryToPatchCrossAttention().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    vision_tensor = vision_tensor.to(device)
    text_tensor = text_tensor.to(device)

    # 4. Run Inference (Get Probability and Attention)
    with torch.no_grad():
        prob, attention_weights = model(vision_tensor, text_tensor)

    prediction_score = prob.item()
    predicted_class = "MSI" if prediction_score >= 0.83 else "MSS"

    # cross-attention outputs shape [1, 1, N]. We flatten it to a 1D list.
    attention_scores = attention_weights.squeeze().cpu().numpy()

    print("\n--- Diagnostic Results ---")
    print(f"True Class:      {true_label}")
    print(f"Predicted Class: {predicted_class} (Confidence: {prediction_score:.4f})")

    # 5. XAI Mapping: Find the Top 5 most important patches
    # argsort returns indices from lowest to highest. We take the last 5 and reverse them.
    top_5_indices = attention_scores.argsort()[-5:][::-1]

    print("\n🔬 Generating Visual Morphology Heatmap...")
    print("\n📍 --- TOP 5 PATCH FILE PATHS ---")

    # 6. Plotting the Results
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Patient {patient_id} | True: {true_label} | Pred: {predicted_class} ({prediction_score:.2f})",
                 fontsize=16, fontweight='bold')

    for i, idx in enumerate(top_5_indices):
        score = attention_scores[idx]

        # Extract just the filename from your friend's JSON path
        filename = os.path.basename(image_paths[idx])

        # Route it to the correct local Kaggle folder on your Mac
        local_folder = "../data/MSIMUT_JPEG" if true_label == "MSI" else "../data/MSS_JPEG"
        local_img_path = os.path.join(local_folder, filename)

        # PRINT THE EXACT PATH TO THE TERMINAL FOR EASY COPY-PASTING
        print(f"Rank {i + 1} [Score: {score:.4f}]: {local_img_path}")

        try:
            img = Image.open(local_img_path)
            axes[i].imshow(img)
            # Draw a red border to signify XAI high-attention
            for spine in axes[i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)
        except Exception as e:
            axes[i].text(0.5, 0.5, 'Image Missing\nCheck Folder', ha='center', va='center')

        axes[i].set_title(f"Rank {i + 1}\nAttn Score: {score:.4f}", color='darkred', fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    output_filename = f"../assets/medical_images/XAI_Report_{patient_id}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ XAI Report successfully saved as: {output_filename}")


if __name__ == "__main__":
    # We will use one of the TCGA IDs we verified earlier from your clean_tensors directory
    # Ensure this patient's actual JPEGs exist in the paths defined in patient_registry.json
    generate_xai_report("TCGA-AA-3877")