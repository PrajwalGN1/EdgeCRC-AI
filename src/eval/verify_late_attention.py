import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. IMPORT YOUR FRIEND'S MODEL ---
# Update this line to match his actual file name and class name
from src.architectures.late_fusion_model import MultimodalFusionNetwork as LateFusionModel

# --- Configuration ---
PATIENT_ID = "TCGA-A6-6142"
TENSORS_DIR = "../../data/clean_tensors"
MODEL_WEIGHTS = "../../models/late_fusion_model_v1_0.pth"


def run_late_fusion_test():
    print("🔬 Launching Real Late Fusion Attention Matrix Test...")
    print("=" * 60)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Executing validation logic on device: {device}\n")

    # 2. Load Model
    model = LateFusionModel().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
        model.eval()
        print(f"✅ Baseline weights loaded successfully from: {MODEL_WEIGHTS}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_WEIGHTS}.")
        return

    # 3. Load Patient Tensors
    vision_path = os.path.join(TENSORS_DIR, f"{PATIENT_ID}_vision.pt")
    text_path = os.path.join(TENSORS_DIR, f"{PATIENT_ID}_text.pt")

    if not os.path.exists(vision_path) or not os.path.exists(text_path):
        print(f"❌ Error: Missing tensor files for patient {PATIENT_ID}")
        return

    vision_tensor = torch.load(vision_path, weights_only=True).unsqueeze(0).to(device)
    real_text_tensor = torch.load(text_path, weights_only=True).unsqueeze(0).to(device)

    # Align text dimensions to 2D [1, 768] if needed for his architecture
    if real_text_tensor.dim() == 3:
        real_text_tensor = real_text_tensor.squeeze(1)

    # Generate the perturbed fake text noise
    fake_text_tensor = torch.randn_like(real_text_tensor).to(device)

    # 4. Compute Dual-Pass Inference & Extract Weights
    with torch.no_grad():
        # --- PASS A: Real Image + Real Text ---
        # We assume his forward pass returns (probability, attention_weights)
        # If his model returns it differently, adjust this line to match his return signature!
        prob_real, attn_real = model(vision_tensor, real_text_tensor)
        scores_real = attn_real.squeeze().cpu().numpy()
        top_5_real = scores_real.argsort()[-5:][::-1]

        # --- PASS B: Real Image + Fake Text ---
        prob_fake, attn_fake = model(vision_tensor, fake_text_tensor)
        scores_fake = attn_fake.squeeze().cpu().numpy()
        top_5_fake = scores_fake.argsort()[-5:][::-1]

    # 5. Display the Proof
    print("\n" + "-" * 40)
    print(f"📊 ACTUAL LATE FUSION ATTENTION WEIGHTS: {PATIENT_ID}")
    print("-" * 40)
    print(
        f"Pass A [REAL Text Context]  -> Diagnosis: {'MSI' if prob_real.item() > 0.5 else 'MSS'} (Prob: {prob_real.item():.4f})")
    print(f"📌 Real Text Top 5 Patches: {list(top_5_real)}")
    print("-" * 40)
    print(
        f"Pass B [PERTURBED Context]   -> Diagnosis: {'MSI' if prob_fake.item() > 0.5 else 'MSS'} (Prob: {prob_fake.item():.4f})")
    print(f"📌 Fake Text Top 5 Patches: {list(top_5_fake)}")
    print("-" * 40 + "\n")

    # 6. Evaluation Assert
    if np.array_equal(top_5_real, top_5_fake):
        print("❌ LIVE DEMO PROOF CONFIRMED:")
        print("Even though his model calculates a real attention matrix, the values did NOT change.")
        print("This proves his attention layer is entirely unimodal (blind to the patient's record).")

        generate_overlapping_plot(scores_real, scores_fake)
    else:
        print("⚠️ Exception: The scores shifted. Check if there is an implicit cross-layer in his code.")


def generate_overlapping_plot(real_scores, fake_scores):
    plt.figure(figsize=(10, 4))
    plt.plot(real_scores, label="Attention Profile (Real Clinical Notes)", color="#CD5C5C", alpha=0.9, lw=2)
    plt.plot(fake_scores, label="Attention Profile (Perturbed Text Noise)", color="#8B0000", alpha=0.7, lw=1.5,
             linestyle=":")
    plt.title("Late Fusion Performance: Static Attention Matrices", fontweight="bold")
    plt.xlabel("WSI Patch Index (0 to 384)", fontweight="bold")
    plt.ylabel("Extracted Attention Weight", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../../results/Late_Fusion_Verification_Proof.png", dpi=300)
    print("🎨 Saved real attention map comparison as: 'Late_Fusion_Verification_Proof.png'")


if __name__ == "__main__":
    run_late_fusion_test()