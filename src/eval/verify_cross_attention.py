import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.architectures.cross_attention_fusion import SummaryToPatchCrossAttention

# --- Configuration ---
PATIENT_ID = "TCGA-A6-6142"  # The verified patient ID from your dataset
TENSORS_DIR = "../../data/clean_tensors"
MODEL_WEIGHTS = "../../models/fusion_model_v1_1.pth"

def run_perturbation_test():
    print("🔬 Launching Cross-Attention Dynamic Perturbation Test...")
    print("=" * 60)

    # Hardware routing for your Apple Silicon M4
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Executing validation logic on device: {device}\n")

    # 1. Initialize and Load Your Cross-Attention Architecture
    model = SummaryToPatchCrossAttention().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
        model.eval()
        print(f"✅ Neural weights loaded successfully from: {MODEL_WEIGHTS}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_WEIGHTS}. Please verify the file name.")
        return

    # 2. Extract the REAL Patient Tensors
    vision_path = os.path.join(TENSORS_DIR, f"{PATIENT_ID}_vision.pt")
    text_path = os.path.join(TENSORS_DIR, f"{PATIENT_ID}_text.pt")

    if not os.path.exists(vision_path) or not os.path.exists(text_path):
        print(f"❌ Error: Missing tensor files for patient {PATIENT_ID} in {TENSORS_DIR}/")
        return

    # Load tensors and align shapes to [Batch=1, Sequence, Embedding]
    vision_tensor = torch.load(vision_path, weights_only=True).unsqueeze(0).to(device)
    real_text_tensor = torch.load(text_path, weights_only=True).unsqueeze(0).to(device)

    # 3. Synthesize the Perturbed "Fake" Text Tensor
    # Generates uniform random noise with the exact same mathematical properties [1, 1, 768]
    # to simulate a completely uncorrelated, different clinical report context
    fake_text_tensor = torch.randn_like(real_text_tensor).to(device)

    # 4. Compute Dual-Pass Inference
    with torch.no_grad():
        # Pass A: Real Image + Real Text
        prob_real, attn_real = model(vision_tensor, real_text_tensor)
        scores_real = attn_real.squeeze().cpu().numpy()
        top_5_real = scores_real.argsort()[-5:][::-1]

        # Pass B: Real Image + Fake (Perturbed) Text
        prob_fake, attn_fake = model(vision_tensor, fake_text_tensor)
        scores_fake = attn_fake.squeeze().cpu().numpy()
        top_5_fake = scores_fake.argsort()[-5:][::-1]

    # 5. Display the Architectural Proof
    print("\n" + "-" * 40)
    print(f"📊 METRIC RESULTS FOR PATIENT: {PATIENT_ID}")
    print("-" * 40)
    print(
        f"Pass A [REAL Text Context]  -> Diagnosis: {'MSI' if prob_real.item() > 0.5 else 'MSS'} (Prob: {prob_real.item():.4f})")
    print(f"📌 Top 5 Patch Focus Indices: {list(top_5_real)}")
    print("-" * 40)
    print(
        f"Pass B [PERTURBED Context]   -> Diagnosis: {'MSI' if prob_fake.item() > 0.5 else 'MSS'} (Prob: {prob_fake.item():.4f})")
    print(f"📌 Top 5 Patch Focus Indices: {list(top_5_fake)}")
    print("-" * 40 + "\n")

    # 6. Evaluation Assert Statement
    if np.array_equal(top_5_real, top_5_fake):
        print("❌ CRITICAL VERIFICATION FAILED:")
        print(
            "The attention matrix did not shift. The model is behaving like late fusion (ignoring text semantic context).")
    else:
        print("✅ CRITICAL VERIFICATION PASSED:")
        print("The structural patch rankings physically shifted! This proves that the multi-head attention blocks")
        print("are actively using the text embedding vector to search and weigh the visual landscape.")

        # 7. Bonus: Generate a quick validation chart for your slides!
        generate_proof_plot(scores_real, scores_fake)


def generate_proof_plot(real_scores, fake_scores):
    plt.figure(figsize=(10, 4))
    plt.plot(real_scores, label="Attention Profile (Real Clinical Notes)", color="#104E8B", alpha=0.8, lw=1.5)
    plt.plot(fake_scores, label="Attention Profile (Perturbed Text Noise)", color="#FFD700", alpha=0.6, lw=1.5,
             linestyle="--")
    plt.title("Cross-Attention Diagnostic Profile Shift", fontweight="bold")
    plt.xlabel("WSI Patch Index (0 to 384)", fontweight="bold")
    plt.ylabel("Mathematical Attention Weight", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../../results/Cross_Attention_Verification_Proof.png", dpi=300)
    print("🎨 Generated visualization asset: 'Cross_Attention_Verification_Proof.png'")


if __name__ == "__main__":
    run_perturbation_test()