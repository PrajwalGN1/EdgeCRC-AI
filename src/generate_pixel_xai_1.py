import os
# --- ENFORCE STRICT LOCAL-FIRST POLICY ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModel
import logging


def get_interactive_input():
    print("\n" + "=" * 50)
    print("🔬 ADVANCED MULTIMODAL XAI ENGINE v2.0 (Microscopic)")
    print("=" * 50)
    print("Select the tissue class folder:")
    print("  [1] MSI (MSIMUT_JPEG)")
    print("  [2] MSS (MSS_JPEG)")

    folder_choice = input("\nEnter 1 or 2: ").strip()
    if folder_choice == '1':
        target_folder = "../data/MSIMUT_JPEG"
    elif folder_choice == '2':
        target_folder = "../data/MSS_JPEG"
    else:
        print("❌ Invalid choice. Defaulting to MSS_JPEG.")
        target_folder = "MSS_JPEG"

    print(f"\n📂 Selected Folder: {target_folder}")
    img_filename = input("📝 Enter the exact image filename (e.g., TCGA-CM-6674_patch_45.jpg): ").strip()

    full_path = os.path.join(target_folder, img_filename)
    return full_path


def generate_smooth_vit_heatmap(img_path):
    print(f"\n🚀 Initializing Advanced Phikon-v2 Engine for: {img_path}")

    try:
        raw_image = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Error: Could not find image at {img_path}. Please check the filename and folder.")
        return

    # Hardware Check Optimized for Apple Silicon M4 GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Executing on device: {device}")

    # Suppress verbose HuggingFace warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)

    print("Loading Phikon-v2 foundation model...")
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2", output_attentions=True).to(device)
    model.eval()

    inputs = processor(images=raw_image, return_tensors="pt").to(device)

    print("Extracting deep attention matrices...")
    with torch.no_grad():
        outputs = model(**inputs)

    # ---------------------------------------------------------
    # THE MATHEMATICAL FIX: Deep Layer Averaging
    # ---------------------------------------------------------
    # outputs.attentions contains all 24 layers.
    # We grab the last 6 layers to capture broad semantic structure.
    stacked_attentions = torch.stack(outputs.attentions[-6:])  # Shape: [6, Batch, Heads, Seq, Seq]

    # 1. Average across the 6 layers
    layer_avg = torch.mean(stacked_attentions, dim=0)

    # 2. Average across the 16 Attention Heads
    head_avg = torch.mean(layer_avg[0], dim=0)

    # 3. Extract the [CLS] token's attention to the image patches
    cls_attention = head_avg[0, 1:].cpu().numpy()

    # Reshape into a 2D grid (14x14 for standard ViT)
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    attention_grid = cls_attention.reshape(grid_size, grid_size)

    # ---------------------------------------------------------
    # THE VISUAL FIX: Gaussian Smoothing & Dynamic Thresholding
    # ---------------------------------------------------------
    # Normalize the raw 14x14 grid first
    attention_grid = (attention_grid - np.min(attention_grid)) / (
                np.max(attention_grid) - np.min(attention_grid) + 1e-8)

    # Apply a Gaussian Blur to create organic, continuous blobs
    smoothed_grid = cv2.GaussianBlur(attention_grid, (3, 3), 0)

    # Re-normalize after blurring
    smoothed_grid = (smoothed_grid - np.min(smoothed_grid)) / (np.max(smoothed_grid) - np.min(smoothed_grid) + 1e-8)

    # DYNAMIC THRESHOLDING: Dynamically find the 70th percentile of attention intensity
    # This guarantees the top 30% of the tissue is always highlighted, preventing blank heatmaps.
    dynamic_threshold = np.percentile(smoothed_grid, 70)
    smoothed_grid[smoothed_grid < dynamic_threshold] = 0

    # Resize cleanly to match the original 224x224 image dimensions
    heatmap = cv2.resize(smoothed_grid, (raw_image.size[0], raw_image.size[1]), interpolation=cv2.INTER_CUBIC)

    # --- Visual Rendering ---
    print("Generating visual output...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    filename_only = os.path.basename(img_path)
    fig.suptitle(f"Microscopic Pixel-Level Attention\nTarget: {filename_only}", fontsize=16, fontweight='bold', y=0.98)

    # 1. Original Image
    axes[0].imshow(raw_image)
    axes[0].set_title("Original H&E Tissue Patch", fontweight='bold', pad=15)
    axes[0].axis('off')

    # 2. The Smoothed B&W Attention
    axes[1].imshow(heatmap, cmap='gray')
    axes[1].set_title("Smoothed ViT Attention Map", fontweight='bold', pad=15)
    axes[1].axis('off')

    # 3. The Heatmap Overlay
    axes[2].imshow(raw_image)
    # Using 'jet' colormap for the classic deep blue to deep red gradient
    axes[2].imshow(heatmap, cmap='jet', alpha=0.45)
    axes[2].set_title("Cellular Proximity Heatmap Overlay", fontweight='bold', color='darkred', pad=15)
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])


    output_name = f"Interactive_XAI_{os.path.basename(img_path)}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ SUCCESS! Report saved safely as: {output_name}")


if __name__ == "__main__":
    target_img_path = get_interactive_input()
    if target_img_path:
        generate_smooth_vit_heatmap(target_img_path)