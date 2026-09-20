import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix,roc_curve

# Import custom modules
from multimodal_dataset import MultimodalTCGADataset
from architectures.late_fusion_model import MultimodalFusionNetwork

# --- Configuration ---
REGISTRY = "../data/patient_registry.json"
TENSORS = "../data/clean_tensors"
MODEL_WEIGHTS = "../models/late_fusion_model_v1_0.pth"
EVAL_SIZE = 66


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

    # Set a manual seed so this "random" selection stays consistent for your testing
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

    # Grab 66 random patients
    eval_dataset, _ = random_split(full_dataset, [EVAL_SIZE, len(full_dataset) - EVAL_SIZE])
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    print(f"Data Loaded: Evaluating on {len(eval_dataset)} random patients.\n")

    # 2. Initialize Model & Load Weights
    model = MultimodalFusionNetwork().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print(f"✅ Loaded weights from {MODEL_WEIGHTS}")
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
    with torch.no_grad():
        for batch in eval_loader:
            vision = batch["vision"].to(device)
            text = batch["text"].to(device)
            label = batch["label"].to(device).item()

            # Forward pass
            prob, _ = model(vision, text)
            prob_val = prob.item()

            # Store results
            true_labels.append(label)
            pred_probs.append(prob_val)
            pred_classes.append(1 if prob_val >= 0.70 else 0)  # 0.5 Threshold

    # 4. Calculate Clinical Metrics
    print("\n📊 Calculating Metrics...")
    auc = roc_auc_score(true_labels, pred_probs)
    acc = accuracy_score(true_labels, pred_classes)

    # Target names mapping (0 = MSS, 1 = MSI)
    target_names = ['MSS (Class 0)', 'MSI (Class 1)']

    report_dict = classification_report(true_labels, pred_classes, target_names=target_names)
    cm = confusion_matrix(true_labels, pred_classes)

    # 5. Terminal Output
    print("-" * 50)
    print(f"OVERALL ACCURACY : {acc:.4f}")
    print(f"ROC-AUC SCORE    : {auc:.4f}")
    print("-" * 50)
    print(report_dict)

    # 6. Save Text Report
    with open("../results/Late_FusionModel_Evaluation_Report.txt", "w") as f:
        f.write("=== MULTIMODAL FUSION NETWORK - EVALUATION REPORT ===\n\n")
        f.write(f"Evaluated on {EVAL_SIZE} patients.\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC Score: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_dict)

    print("✅ Saved text report to 'Model_Evaluation_Report.txt'")

    # 7. Generate and Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted MSS', 'Predicted MSI'],
                yticklabels=['Actual MSS', 'Actual MSI'],
                annot_kws={"size": 16})

    plt.title('Clinical Confusion Matrix', fontsize=18, pad=20)
    plt.ylabel('True Biomarker Status', fontsize=14)
    plt.xlabel('AI Predicted Status', fontsize=14)

    # Save the plot
    plt.tight_layout()
    plt.savefig('../results/Confusion_Matrix_L.png', dpi=300)
    print("✅ Saved visualization to 'Confusion_Matrix_L.png'")

    # 8. Generate and Save ROC Curve
    print("📈 Generating ROC Curve plot...")
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Late-Fusion (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('../results/ROC_Curve_L.png', dpi=300)
    plt.close()
    print("✅ Saved ROC Curve visualization to 'ROC_Curve.png'")


if __name__ == "__main__":
    evaluate_model()
