import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Added for matrix operations
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

# Import custom modules
from multimodal_dataset import MultimodalTCGADataset
from architectures.cross_attention_fusion import SummaryToPatchCrossAttention

# --- Configuration ---
REGISTRY = "../data/patient_registry.json"
TENSORS = "../data/clean_tensors"
MODEL_WEIGHTS = "../models/fusion_model_v1_1.pth"


def plot_cross_attention_matrix(attention_matrix, text_tokens, patch_labels,
                                save_path="../results/Real_Cross_Attention_Matrix.png"):
    """
    Generates a publication-ready heatmap of the cross-attention weights.
    """
    print("\n📊 Generating Cross-Attention Matrix diagram...")

    # 1. Figure Setup
    plt.figure(figsize=(12, 8), facecolor='white')
    sns.set_theme(style="white")

    # 2. Draw the Heatmap
    ax = sns.heatmap(
        attention_matrix,
        cmap="mako",
        cbar_kws={'label': 'Attention Weight (Softmax Probability)'},
        linewidths=0.5,
        linecolor='white',
        square=True,
        vmin=0.0,
        vmax=np.max(attention_matrix) * 1.2
    )

    # 3. Labeling & Styling
    ax.set_title("Summary-to-Patch Cross-Attention Matrix\n(Token vs. Patch Mapping)",
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xlabel("Vision Transformer Image Patches (N)", fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel("Clinical Summary Text Tokens (M)", fontsize=13, fontweight='bold', labelpad=15)

    ax.set_xticklabels(patch_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(text_tokens, rotation=0, fontsize=11, fontweight='medium')

    # 4. Save High-Res Output
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved publication-ready cross-attention matrix to '{save_path}'")


def evaluate_model():
    print("🔬 Initiating Sub-Phase: Clinical Diagnostic Evaluation\n")

    # Optimized for Apple Silicon M4 GPU acceleration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Executing on device: {device}")

    # 1. Load Dataset
    full_dataset = MultimodalTCGADataset(REGISTRY, TENSORS, max_patches=1000)

    # Load the isolated keys generated during training
    try:
        with open("../results/actual_validation_keys.json", "r") as f:
            val_keys = set(json.load(f))
        print(
            f"🔓 Found training data contract. Restricting evaluation pool to {len(val_keys)} unseen validation patients.")
    except FileNotFoundError:
        print("❌ Error: 'actual_validation_keys.json' not found. Please run train.py once with the new update.")
        return

    # Filter the active evaluation set down strictly to true validation records
    eval_patients = [p for p in full_dataset.valid_patients if p["id"] in val_keys]

    if len(eval_patients) == 0:
        print("❌ Error: No validation patients matched the JSON keys.")
        return

    # Assign the isolated validation records back to the dataset
    full_dataset.valid_patients = eval_patients

    # Load them sequentially (No shuffle, batch_size=1)
    eval_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    actual_eval_size = len(eval_patients)
    print(f"Data Loaded: Evaluating strictly on the {actual_eval_size} unexposed validation records.\n")

    # 2. Initialize Cross-Attention Model & Load Weights
    model = SummaryToPatchCrossAttention().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print(f"✅ Loaded Cross-Attention weights from {MODEL_WEIGHTS}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_WEIGHTS}. Did the training script finish?")
        return

    model.eval()  # STRICTLY set to evaluation mode (turns off dropout)

    # Tracking arrays
    true_labels = []
    pred_probs = []
    pred_classes = []

    # 3. Inference Loop
    print("\n⚙️ Running inference...")
    heatmap_generated = False  # NEW: Flag so we only generate ONE diagram

    with torch.no_grad():
        for batch in eval_loader:
            vision = batch["vision"].to(device)
            text = batch["text"].to(device)
            label = batch["label"].to(device).item()

            # NEW: Forward pass catches both probability and attention weights
            prob, attn_weights = model(vision, text)
            prob_val = prob.item()

            # ==========================================
            # DIAGRAM GENERATION (Just for the first patient)
            # ==========================================
            if not heatmap_generated and attn_weights is not None:
                print(f"📸 Capturing Cross-Attention Matrix for Patient 1...")

                # Move weights to CPU and convert to numpy
                matrix_np = attn_weights[0].cpu().numpy()

                # Slice the matrix safely (prevents index errors if sequences are short)
                slice_M = min(15, matrix_np.shape[0])
                slice_N = min(20, matrix_np.shape[1])
                matrix_slice = matrix_np[:slice_M, :slice_N]

                # Create placeholder labels (Update these if you want real words!)
                text_labels = [f"Token_{i}" for i in range(slice_M)]
                patch_labels = [f"Patch_{i}" for i in range(slice_N)]

                # Generate the plot
                plot_cross_attention_matrix(
                    attention_matrix=matrix_slice,
                    text_tokens=text_labels,
                    patch_labels=patch_labels,
                    save_path="../results/Real_Cross_Attention_Matrix.png"
                )
                heatmap_generated = True  # Flip the flag so it doesn't plot the rest
            # ==========================================

            # Store results
            true_labels.append(label)
            pred_probs.append(prob_val)
            pred_classes.append(1 if prob_val >= 0.83 else 0)

    # 4. Calculate Clinical Metrics
    print("\n📊 Calculating Metrics...")
    auc = roc_auc_score(true_labels, pred_probs)
    acc = accuracy_score(true_labels, pred_classes)

    # Target names mapping (0 = MSS, 1 = MSI)
    target_names = ['MSS (Class 0)', 'MSI (Class 1)']

    report_dict = classification_report(true_labels, pred_classes, target_names=target_names, zero_division=0)
    cm = confusion_matrix(true_labels, pred_classes)

    # 5. Terminal Output
    print("-" * 50)
    print(f"OVERALL ACCURACY : {acc:.4f}")
    print(f"ROC-AUC SCORE    : {auc:.4f}")
    print("-" * 50)
    print(report_dict)

    # 6. Save Text Report
    with open("../results/Cross_Attention_Model_Evaluation_Report.txt", "w") as f:
        f.write("=== MULTIMODAL CROSS-ATTENTION FUSION - EVALUATION REPORT ===\n\n")
        f.write(f"Evaluated on {actual_eval_size} isolated validation patients.\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC Score: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_dict)

    print("✅ Saved text report to 'Model_Evaluation_Report.txt'")

    # 7. Generate and Save Confusion Matrix Plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted MSS', 'Predicted MSI'],
                yticklabels=['Actual MSS', 'Actual MSI'],
                annot_kws={"size": 16})
    plt.title('Clinical Confusion Matrix (Cross-Attention)', fontsize=14, pad=15)
    plt.ylabel('True Biomarker Status', fontsize=12)
    plt.xlabel('AI Predicted Status', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/Confusion_Matrix_C.png', dpi=300)
    plt.close()
    print("✅ Saved visualization to 'Confusion_Matrix_C.png'")

    # 8. Generate and Save ROC Curve
    print("📈 Generating ROC Curve plot...")
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Cross-Attention (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('../results/ROC_Curve_C.png', dpi=300)
    plt.close()
    print("✅ Saved ROC Curve visualization to 'ROC_Curve_C.png'")


if __name__ == "__main__":
    evaluate_model()