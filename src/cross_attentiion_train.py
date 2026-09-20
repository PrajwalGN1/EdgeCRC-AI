import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import json
import random

# Import our custom modules
from multimodal_dataset import MultimodalTCGADataset
#from fusion_model import MultimodalFusionNetwork
from architectures.cross_attention_fusion import SummaryToPatchCrossAttention

# --- Configuration ---
REGISTRY = "../data/patient_registry.json"
TENSORS = "../data/clean_tensors"
EPOCHS = 30
LEARNING_RATE = 1e-4
SAVED_MODEL_NAME = "../models/fusion_model_v1_1.pth"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # Anchors random numbers for Apple Silicon GPU calculations
        torch.mps.manual_seed(seed)


def weighted_bce_loss(prob, label, pos_weight=4.17):
    """
    Custom Loss Function to fix Recall/F1.
    pos_weight = 263 (MSS) / 63 (MSI) = 4.17
    Penalizes the model 4.17x more when it misses an MSI case.
    """
    # Clamp probabilities to prevent PyTorch from crashing with log(0)
    prob = torch.clamp(prob, min=1e-7, max=1 - 1e-7)

    # Calculate the weighted loss
    loss = - (pos_weight * label * torch.log(prob) + (1 - label) * torch.log(1 - prob))
    return loss.mean()

def train_model():
    print("🚀 Initiating Sub-Phase 3.3: Multimodal Training Pipeline\n")
    
    # 1. Hardware Check (It will run fine on your Intel CPU!)
        # 1. Hardware Check (Optimized for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Training on device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Training on device: Apple Silicon MPS (M4 GPU)")
    else:
        device = torch.device('cpu')
        print("Training on device: CPU")
    print(f"Training on device: {device}\n")

    set_seed(42)

    # 2. Load Dataset and Split
    full_dataset = MultimodalTCGADataset(REGISTRY, TENSORS, max_patches=1000)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Data Split: {train_size} Training Patients | {val_size} Validation Patients")

    # === ADD THIS NEW BLOCK HERE TO LOCK THE VALIDATION KEYS ===
    val_patient_ids = [full_dataset.valid_patients[idx]["id"] for idx in val_dataset.indices]
    with open("../results/actual_validation_keys.json", "w") as f:
        json.dump(val_patient_ids, f)
    print("💾 Isolated validation patient keys saved to 'actual_validation_keys.json'")


    # Batch size MUST be 1 because we avoided zero-padding.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 3. Initialize Model, Loss, and Optimizer
    #model = MultimodalFusionNetwork().to(device)
    model = SummaryToPatchCrossAttention().to(device)
    # criterion = nn.BCELoss() # Binary Cross Entropy for 1/0 classification
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_auc = 0.0
    # === NEW: Arrays to track loss for the diagram ===
    history_train_loss = []
    history_val_loss = []

    # 4. The Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            vision = batch["vision"].to(device)
            text = batch["text"].to(device)
            label = batch["label"].to(device).float()
            
            optimizer.zero_grad()
            
            # Forward Pass
            prob, _ = model(vision, text)
            
            # Squeeze to ensure shapes match exactly: [1] vs [1]
            # loss = criterion(prob.squeeze(), label.squeeze())
            # New Loss Function to fix Recall/F1 by penalizing missed MSI cases more heavily:
            loss = weighted_bce_loss(prob.squeeze(), label.squeeze(), pos_weight=4.17)

            # Backward Pass (Learn)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Loop (Testing without learning)
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                vision = batch["vision"].to(device)
                text = batch["text"].to(device)
                label = batch["label"].to(device).float()
                
                prob, _ = model(vision, text)
                # loss = criterion(prob.squeeze(), label.squeeze())
                loss = weighted_bce_loss(prob.squeeze(), label.squeeze(), pos_weight=4.17)

                val_loss += loss.item()
                all_labels.append(label.cpu().item())
                all_probs.append(prob.cpu().item())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Medical Metrics
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.5 # Fallback if validation batch happens to have only one class
            
        val_preds = [1 if p >= 0.5 else 0 for p in all_probs]
        val_acc = accuracy_score(all_labels, val_preds)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        # === NEW: Store the metrics for plotting ===
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        # 6. Save the Best Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), SAVED_MODEL_NAME)
            print(f"   🌟 New Best Model Saved! (AUC: {best_val_auc:.4f})")

    print(f"\n✅ Training Complete. Best weights saved as {SAVED_MODEL_NAME}")

    print(f"\n✅ Training Complete. Best weights saved as {SAVED_MODEL_NAME}")

    # === NEW: Generate the Loss Trajectory Diagram ===
    import matplotlib.pyplot as plt

    print("📈 Generating Weighted BCE Loss Trajectory Diagram...")
    plt.figure(figsize=(10, 6))

    # Plotting the lines
    plt.plot(range(1, EPOCHS + 1), history_train_loss, label='Training Loss (Weighted BCE)', color='#1f77b4',
             linewidth=2.5, marker='o', markersize=4)
    plt.plot(range(1, EPOCHS + 1), history_val_loss, label='Validation Loss (Weighted BCE)', color='#d62728',
             linewidth=2.5, marker='s', markersize=4)

    # Styling the graph for reports
    plt.title('Training Convergence & Weighted BCE Loss Trajectory', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel('Loss Value', fontsize=13, fontweight='bold', labelpad=10)

    # Customize ticks
    plt.xticks(range(1, EPOCHS + 1, max(1, EPOCHS // 10)))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Save the output cleanly
    plt.tight_layout()
    plot_path = "../results/BCE_Loss_Trajectory.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved Loss Convergence diagram to '{plot_path}'")

if __name__ == "__main__":
    train_model()
